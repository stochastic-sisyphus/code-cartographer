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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# Get paths to web assets
_module_path = Path(__file__).parent.parent
_static_path = _module_path / "web" / "static"
_templates_path = _module_path / "web" / "templates"

# Create templates directory if it doesn't exist
_templates_path.mkdir(parents=True, exist_ok=True)
_static_path.mkdir(parents=True, exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory=str(_templates_path))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifecycle management for the FastAPI app."""
    logger.info("Starting Code Warp House server")
    logger.info(f"Static files: {_static_path}")
    logger.info(f"Templates: {_templates_path}")
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

# Mount static files
app.mount("/static", StaticFiles(directory=str(_static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint - serves the web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


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
