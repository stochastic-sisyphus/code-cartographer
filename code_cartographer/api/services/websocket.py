"""
WebSocket Service
=================
Real-time communication for analysis progress and updates.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, project_id: str):
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            project_id: Project identifier to subscribe to
        """
        await websocket.accept()

        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()

        self.active_connections[project_id].add(websocket)
        logger.info(f"Client connected to project {project_id}")

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to Code Warp House"
        })

    def disconnect(self, websocket: WebSocket, project_id: str):
        """
        Remove a WebSocket connection.

        Args:
            websocket: The WebSocket connection
            project_id: Project identifier
        """
        if project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)

            if not self.active_connections[project_id]:
                del self.active_connections[project_id]

        logger.info(f"Client disconnected from project {project_id}")

    async def send_progress(
        self,
        project_id: str,
        progress: int,
        total: int,
        message: str
    ):
        """
        Send progress update to all connected clients for a project.

        Args:
            project_id: Project identifier
            progress: Current progress count
            total: Total items to process
            message: Progress message
        """
        if project_id not in self.active_connections:
            return

        data = {
            "type": "progress",
            "progress": progress,
            "total": total,
            "percentage": int((progress / total) * 100) if total > 0 else 0,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

        await self._broadcast(project_id, data)

    async def send_complete(self, project_id: str, data: Dict):
        """
        Send completion message to all connected clients.

        Args:
            project_id: Project identifier
            data: Completion data
        """
        message = {
            "type": "complete",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        await self._broadcast(project_id, message)

    async def send_error(self, project_id: str, error: str):
        """
        Send error message to all connected clients.

        Args:
            project_id: Project identifier
            error: Error message
        """
        message = {
            "type": "error",
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        await self._broadcast(project_id, message)

    async def _broadcast(self, project_id: str, message: Dict):
        """
        Broadcast a message to all connections for a project.

        Args:
            project_id: Project identifier
            message: Message to broadcast
        """
        if project_id not in self.active_connections:
            return

        # Create a copy of the set to avoid modification during iteration
        connections = self.active_connections[project_id].copy()

        for connection in connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(connection, project_id)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                self.disconnect(connection, project_id)


# Global connection manager instance
manager = ConnectionManager()
