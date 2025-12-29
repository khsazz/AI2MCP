"""GNN Reasoner HTTP Service.

Provides REST API for GNN inference, allowing the MCP server
to query scene understanding from a separate service.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from gnn_reasoner.graph_builder import SemanticGraphBuilder, WorldGraph, GraphNode, GraphEdge
from gnn_reasoner.model.scene_gnn import SceneGNN, SCENE_CLASSES

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = structlog.get_logger()


class GNNReasonerService:
    """HTTP service for GNN-based scene reasoning."""

    def __init__(
        self,
        model_path: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8081,
    ):
        """Initialize GNN service.
        
        Args:
            model_path: Path to trained model weights (optional)
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.graph_builder = SemanticGraphBuilder()
        self.model: SceneGNN | None = None

        if TORCH_AVAILABLE:
            try:
                self.model = SceneGNN()
                if model_path:
                    self.model.load_state_dict(torch.load(model_path, weights_only=True))
                self.model.eval()
                logger.info("GNN model loaded", model_path=model_path)
            except Exception as e:
                logger.warning("Failed to load GNN model", error=str(e))

    async def health(self, request: Request) -> Response:
        """Health check endpoint."""
        return JSONResponse({
            "status": "ok",
            "model_loaded": self.model is not None,
            "scene_classes": SCENE_CLASSES,
        })

    async def process_graph(self, request: Request) -> Response:
        """Process world graph and return scene understanding.
        
        Expected JSON body:
        {
            "nodes": [...],
            "edges": [...],
        }
        """
        try:
            data = await request.json()
            graph = self._parse_graph(data)

            result = self._analyze_graph(graph)
            return JSONResponse(result)

        except Exception as e:
            logger.error("Graph processing failed", error=str(e))
            return JSONResponse(
                {"error": str(e)},
                status_code=400,
            )

    async def process_sensor_data(self, request: Request) -> Response:
        """Build graph from sensor data and analyze.
        
        Expected JSON body:
        {
            "robot_pose": [x, y, theta],
            "scan_obstacles": [...],  # Optional
            "detections": {...},       # Optional
        }
        """
        try:
            data = await request.json()

            robot_pose = tuple(data["robot_pose"])
            scan_obstacles = data.get("scan_obstacles")

            graph = self.graph_builder.build_graph(
                robot_pose=robot_pose,
                scan_obstacles=scan_obstacles,
            )

            result = self._analyze_graph(graph)
            result["graph"] = graph.to_dict()
            
            return JSONResponse(result)

        except Exception as e:
            logger.error("Sensor processing failed", error=str(e))
            return JSONResponse(
                {"error": str(e)},
                status_code=400,
            )

    def _parse_graph(self, data: dict) -> WorldGraph:
        """Parse JSON data into WorldGraph."""
        nodes = []
        for n in data.get("nodes", []):
            nodes.append(GraphNode(
                id=n["id"],
                node_type=n.get("type", "obstacle"),
                position=tuple(n.get("position", [0, 0])),
                attributes=n.get("attributes", {}),
            ))

        edges = []
        for e in data.get("edges", []):
            edges.append(GraphEdge(
                source=e["source"],
                target=e["target"],
                relation=e.get("relation", "near"),
                weight=e.get("weight", 1.0),
                attributes=e.get("attributes", {}),
            ))

        return WorldGraph(nodes=nodes, edges=edges)

    def _analyze_graph(self, graph: WorldGraph) -> dict[str, Any]:
        """Analyze graph and return scene understanding."""
        result: dict[str, Any] = {
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
        }

        # Heuristic analysis (always available)
        result["heuristic"] = self._heuristic_analysis(graph)

        # GNN analysis (if model available)
        if self.model is not None and TORCH_AVAILABLE:
            try:
                data = SceneGNN.from_world_graph(graph, self.model)
                scene_class, confidence = self.model.predict_scene(data)
                
                result["gnn"] = {
                    "scene_class": scene_class,
                    "confidence": round(confidence, 3),
                }
            except Exception as e:
                logger.warning("GNN inference failed", error=str(e))
                result["gnn"] = {"error": str(e)}

        return result

    def _heuristic_analysis(self, graph: WorldGraph) -> dict[str, Any]:
        """Simple rule-based scene analysis."""
        num_obstacles = sum(1 for n in graph.nodes if n.node_type == "obstacle")
        num_objects = sum(1 for n in graph.nodes if n.node_type == "object")

        # Find robot node
        robot = next((n for n in graph.nodes if n.node_type == "robot"), None)

        # Count blocking relations
        blocking_count = sum(1 for e in graph.edges if e.relation == "blocking")
        near_count = sum(1 for e in graph.edges if e.relation == "near")

        # Heuristic scene classification
        if blocking_count >= 2:
            scene = "cluttered"
        elif num_obstacles == 0 and num_objects == 0:
            scene = "open_space"
        elif near_count >= 4:
            scene = "room"
        elif num_obstacles >= 1 and near_count <= 2:
            scene = "corridor"
        else:
            scene = "unknown"

        # Navigation assessment
        can_move = blocking_count == 0
        risk_level = min(1.0, blocking_count * 0.3 + near_count * 0.1)

        return {
            "scene_class": scene,
            "obstacle_count": num_obstacles,
            "object_count": num_objects,
            "blocking_relations": blocking_count,
            "near_relations": near_count,
            "navigation": {
                "can_move_forward": can_move,
                "risk_level": round(risk_level, 2),
                "recommendation": "proceed" if can_move else "avoid",
            },
        }

    def create_app(self) -> Starlette:
        """Create Starlette application."""
        return Starlette(
            debug=True,
            routes=[
                Route("/health", self.health),
                Route("/process_graph", self.process_graph, methods=["POST"]),
                Route("/process_sensors", self.process_sensor_data, methods=["POST"]),
            ],
        )


def main() -> None:
    """Entry point for GNN reasoner service."""
    service = GNNReasonerService()
    app = service.create_app()
    
    logger.info("Starting GNN Reasoner Service", host=service.host, port=service.port)
    uvicorn.run(app, host=service.host, port=service.port, log_level="info")


if __name__ == "__main__":
    main()

