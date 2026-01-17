"""
Analysis API Routes
===================
Endpoints for code analysis operations.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from code_cartographer.core.analyzer import ProjectAnalyzer
from code_cartographer.core.dependency_analyzer import DependencyAnalyzer
from code_cartographer.core.variant_analyzer import VariantAnalyzer
from code_cartographer.api.models.schemas import (
    ProjectAnalysisRequest,
    ProjectAnalysisResponse,
    FileMetadataResponse,
    DefinitionResponse,
    DependencyGraphResponse,
    ComplexityMetrics,
    ErrorResponse,
    ProjectListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory cache for projects (in production, use Redis or similar)
_project_cache: Dict[str, Dict] = {}
_cache_dir = Path.home() / ".code-cartographer" / "cache" / "api"
_cache_dir.mkdir(parents=True, exist_ok=True)


def _get_project_id(project_path: str) -> str:
    """Generate a unique project ID from path."""
    return hashlib.sha256(project_path.encode()).hexdigest()[:16]


def _cache_analysis(project_id: str, analysis: Dict) -> None:
    """Cache analysis results to disk."""
    cache_file = _cache_dir / f"{project_id}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Cached analysis for project {project_id}")
    except Exception as e:
        logger.error(f"Failed to cache analysis: {e}")


def _load_cached_analysis(project_id: str) -> Optional[Dict]:
    """Load cached analysis from disk."""
    cache_file = _cache_dir / f"{project_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    return None


@router.post("/projects/analyze", response_model=Dict)
async def analyze_project(request: ProjectAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze a code project.

    Creates a new analysis or returns cached results if available.
    """
    try:
        project_path = Path(request.project_path).resolve()

        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"Project path not found: {project_path}")

        if not project_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {project_path}")

        # Generate project ID
        project_id = _get_project_id(str(project_path))

        # Check cache
        cached = _load_cached_analysis(project_id)
        if cached:
            logger.info(f"Returning cached analysis for {project_id}")
            return {
                "project_id": project_id,
                "status": "cached",
                "analysis": cached,
                "message": "Analysis retrieved from cache"
            }

        # Perform analysis
        logger.info(f"Starting analysis for {project_path}")
        analyzer = ProjectAnalyzer(
            root=project_path,
            exclude_patterns=request.exclude_patterns or []
        )

        analysis_result = analyzer.execute()

        # Add metadata
        analysis_data = {
            "project_id": project_id,
            "project_path": str(project_path),
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_result": analysis_result
        }

        # Cache in background
        background_tasks.add_task(_cache_analysis, project_id, analysis_data)

        # Store in memory
        _project_cache[project_id] = analysis_data

        return {
            "project_id": project_id,
            "status": "complete",
            "analysis": analysis_data,
            "message": "Analysis complete"
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects")
async def list_projects():
    """List all analyzed projects."""
    projects = []

    # Load from cache directory
    for cache_file in _cache_dir.glob("*.json"):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                projects.append({
                    "project_id": data.get("project_id"),
                    "project_path": data.get("project_path"),
                    "analysis_timestamp": data.get("analysis_timestamp"),
                })
        except Exception as e:
            logger.warning(f"Failed to load project from {cache_file}: {e}")

    return {"projects": projects}


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get analysis results for a specific project."""
    # Try memory cache first
    if project_id in _project_cache:
        return _project_cache[project_id]

    # Try disk cache
    cached = _load_cached_analysis(project_id)
    if cached:
        _project_cache[project_id] = cached  # Load into memory
        return cached

    raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")


@router.get("/projects/{project_id}/files")
async def get_project_files(project_id: str):
    """Get list of files in a project."""
    project_data = await get_project(project_id)
    analysis = project_data.get("analysis_result", {})
    files = analysis.get("files", [])

    return {
        "project_id": project_id,
        "total_files": len(files),
        "files": files
    }


@router.get("/projects/{project_id}/files/{file_path:path}")
async def get_file_details(project_id: str, file_path: str):
    """Get detailed information about a specific file."""
    project_data = await get_project(project_id)
    analysis = project_data.get("analysis_result", {})
    files = analysis.get("files", [])

    # Find the file
    for file_data in files:
        if file_data.get("path") == file_path:
            return file_data

    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")


@router.get("/projects/{project_id}/dependencies")
async def get_dependencies(project_id: str):
    """Get dependency graph for a project."""
    project_data = await get_project(project_id)
    project_path = project_data.get("project_path")

    if not project_path:
        raise HTTPException(status_code=400, detail="Project path not found in cache")

    try:
        analyzer = DependencyAnalyzer(root=Path(project_path))
        dep_data = analyzer.analyze()

        # Convert to graph format
        nodes = []
        edges = []

        # Build import graph nodes and edges
        import_graph = dep_data.get("import_graph", {})
        for source, targets in import_graph.items():
            nodes.append({"id": source, "type": "module"})
            for target in targets:
                nodes.append({"id": target, "type": "module"})
                edges.append({"source": source, "target": target, "type": "import"})

        # Deduplicate nodes
        unique_nodes = {node["id"]: node for node in nodes}
        nodes = list(unique_nodes.values())

        return {
            "project_id": project_id,
            "nodes": nodes,
            "edges": edges,
            "metrics": dep_data.get("metrics", {})
        }

    except Exception as e:
        logger.error(f"Failed to analyze dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/variants")
async def get_variants(project_id: str, threshold: float = 0.8):
    """Get code variants (similar code blocks) in a project."""
    project_data = await get_project(project_id)
    project_path = project_data.get("project_path")

    if not project_path:
        raise HTTPException(status_code=400, detail="Project path not found in cache")

    try:
        analyzer = VariantAnalyzer(
            root=Path(project_path),
            semantic_threshold=threshold
        )
        variant_data = analyzer.analyze()

        return {
            "project_id": project_id,
            "threshold": threshold,
            "variants": variant_data.get("variant_groups", [])
        }

    except Exception as e:
        logger.error(f"Failed to analyze variants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project from cache."""
    # Remove from memory
    if project_id in _project_cache:
        del _project_cache[project_id]

    # Remove from disk
    cache_file = _cache_dir / f"{project_id}.json"
    if cache_file.exists():
        cache_file.unlink()

    return {"status": "deleted", "project_id": project_id}
