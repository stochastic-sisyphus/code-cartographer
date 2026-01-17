"""
FastAPI Server for Code Warp House
===================================
Main application entry point for the web API.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifecycle management for the FastAPI app."""
    logger.info("Starting Code Warp House server")
    yield
    logger.info("Shutting down Code Warp House server")


# Create FastAPI application
app = FastAPI(
    title="Code Warp House API",
    description="Immersive temporal code visualization and analysis platform",
    version="0.3.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serves the web interface."""
    return """
    <html>
        <head>
            <title>Code Warp House</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 20px;
                    background: #0a0a0a;
                    color: #e0e0e0;
                }
                h1 {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }
                a {
                    color: #667eea;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
                .endpoint {
                    background: #1a1a1a;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 3px solid #667eea;
                }
                .method {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 12px;
                    margin-right: 10px;
                }
                .get { background: #28a745; color: white; }
                .post { background: #007bff; color: white; }
                .ws { background: #ffc107; color: black; }
            </style>
        </head>
        <body>
            <h1>🌀 Code Warp House</h1>
            <p>Immersive temporal code visualization platform</p>

            <h2>API Documentation</h2>
            <p>Interactive API docs: <a href="/docs">/docs</a></p>
            <p>Alternative docs: <a href="/redoc">/redoc</a></p>

            <h2>Quick Start</h2>
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/api/v1/health</code> - Health check
            </div>
            <div class="endpoint">
                <span class="method post">POST</span>
                <code>/api/v1/projects/analyze</code> - Analyze a codebase
            </div>
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/api/v1/projects</code> - List analyzed projects
            </div>

            <h2>Status</h2>
            <p>Server is running and ready to accept requests.</p>
            <p>Frontend web interface coming soon...</p>
        </body>
    </html>
    """


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "code-warp-house",
        "version": "0.3.0"
    }


# Import and include routers
from code_cartographer.api.routes import analysis, temporal
from code_cartographer.api.services.websocket import manager
from fastapi import WebSocket, WebSocketDisconnect

app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(temporal.router, prefix="/api/v1", tags=["temporal"])


@app.websocket("/api/v1/ws/analysis/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time analysis updates."""
    await manager.connect(websocket, project_id)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            # Echo back for now (can be extended for commands)
            await websocket.send_json({
                "type": "echo",
                "message": f"Received: {data}",
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket, project_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, project_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
