"""
Temporal Analysis API Routes
=============================
Endpoints for git history and temporal code analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from code_cartographer.core.git_analyzer import GitAnalyzer
from code_cartographer.core.temporal_analyzer import TemporalAnalyzer
from code_cartographer.api.models.schemas import (
    CommitResponse,
    TemporalDataResponse,
    ComplexityTrendResponse,
    RefactoringEventResponse,
    TemporalMetricsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Cache for temporal analyzers
_temporal_cache: Dict[str, TemporalAnalyzer] = {}


def _get_temporal_analyzer(project_path: Path) -> TemporalAnalyzer:
    """Get or create a temporal analyzer for a project."""
    path_str = str(project_path)

    if path_str not in _temporal_cache:
        try:
            analyzer = TemporalAnalyzer(repo_path=project_path)
            _temporal_cache[path_str] = analyzer
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Not a git repository: {e}")

    return _temporal_cache[path_str]


@router.get("/projects/{project_id}/timeline")
async def get_timeline(
    project_id: str,
    max_commits: int = Query(default=100, ge=1, le=1000),
    file_pattern: Optional[str] = None
):
    """
    Get commit timeline for a project.

    Args:
        project_id: Project identifier
        max_commits: Maximum number of commits to retrieve
        file_pattern: Optional file pattern to filter commits (e.g., ".py")
    """
    # Load project to get path
    from code_cartographer.api.routes.analysis import get_project

    try:
        project_data = await get_project(project_id)
        project_path = Path(project_data.get("project_path"))
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    try:
        analyzer = _get_temporal_analyzer(project_path)
        commits = analyzer.git_analyzer.get_commit_history(
            max_commits=max_commits,
            file_pattern=file_pattern
        )

        # Convert to response format
        commit_responses = [
            {
                "hash": c.hash,
                "short_hash": c.short_hash,
                "timestamp": c.timestamp.isoformat(),
                "author": c.author,
                "author_email": c.author_email,
                "message": c.message,
                "files_changed": c.files_changed,
                "insertions": c.insertions,
                "deletions": c.deletions,
            }
            for c in commits
        ]

        return {
            "project_id": project_id,
            "total_commits": len(commit_responses),
            "commits": commit_responses
        }

    except Exception as e:
        logger.error(f"Failed to get timeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/commits")
async def list_commits(project_id: str, max_commits: int = Query(default=50, ge=1, le=500)):
    """Get list of commits for a project."""
    return await get_timeline(project_id, max_commits=max_commits)


@router.get("/projects/{project_id}/commits/{commit_hash}")
async def get_commit_details(project_id: str, commit_hash: str):
    """Get detailed information about a specific commit."""
    from code_cartographer.api.routes.analysis import get_project

    try:
        project_data = await get_project(project_id)
        project_path = Path(project_data.get("project_path"))
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    try:
        analyzer = _get_temporal_analyzer(project_path)

        # Get commit from history
        commits = analyzer.git_analyzer.get_commit_history(max_commits=1000)
        commit = next((c for c in commits if c.hash.startswith(commit_hash)), None)

        if not commit:
            raise HTTPException(status_code=404, detail=f"Commit not found: {commit_hash}")

        return {
            "hash": commit.hash,
            "short_hash": commit.short_hash,
            "timestamp": commit.timestamp.isoformat(),
            "author": commit.author,
            "author_email": commit.author_email,
            "message": commit.message,
            "files_changed": commit.files_changed,
            "insertions": commit.insertions,
            "deletions": commit.deletions,
            "analysis_data": commit.analysis_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get commit details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/commits/{commit_hash}/analysis")
async def analyze_at_commit(project_id: str, commit_hash: str):
    """Analyze code at a specific commit."""
    from code_cartographer.api.routes.analysis import get_project

    try:
        project_data = await get_project(project_id)
        project_path = Path(project_data.get("project_path"))
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    try:
        analyzer = _get_temporal_analyzer(project_path)

        # Analyze at commit with caching
        analysis = analyzer._analyze_commit_cached(commit_hash)

        if not analysis:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze commit {commit_hash}"
            )

        return {
            "commit_hash": commit_hash,
            "analysis": analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze at commit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/evolution/complexity")
async def get_complexity_evolution(
    project_id: str,
    max_commits: int = Query(default=50, ge=5, le=200),
    strategy: str = Query(default="uniform", regex="^(uniform|major|all)$")
):
    """
    Get complexity evolution over time.

    Args:
        project_id: Project identifier
        max_commits: Maximum commits to analyze
        strategy: Sampling strategy (uniform, major, all)
    """
    from code_cartographer.api.routes.analysis import get_project

    try:
        project_data = await get_project(project_id)
        project_path = Path(project_data.get("project_path"))
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    try:
        analyzer = _get_temporal_analyzer(project_path)

        # Run temporal analysis
        temporal_data = analyzer.analyze_evolution(
            max_commits=max_commits,
            sample_strategy=strategy
        )

        # Convert complexity trends
        trends = [
            {
                "file_path": trend.file_path,
                "timeline": [
                    {"timestamp": ts.isoformat(), "complexity": comp}
                    for ts, comp in trend.timeline
                ],
                "trend_direction": trend.trend_direction,
                "max_complexity": trend.max_complexity,
                "min_complexity": trend.min_complexity,
                "current_complexity": trend.current_complexity,
            }
            for trend in temporal_data.complexity_trends
        ]

        return {
            "project_id": project_id,
            "total_commits_analyzed": temporal_data.total_commits_analyzed,
            "complexity_trends": trends
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze complexity evolution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/evolution/refactorings")
async def get_refactoring_events(
    project_id: str,
    max_commits: int = Query(default=100, ge=1, le=500)
):
    """Get detected refactoring events."""
    from code_cartographer.api.routes.analysis import get_project

    try:
        project_data = await get_project(project_id)
        project_path = Path(project_data.get("project_path"))
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    try:
        analyzer = _get_temporal_analyzer(project_path)
        events = analyzer.git_analyzer.detect_refactoring_events(max_commits=max_commits)

        event_responses = [
            {
                "commit_hash": e.commit_hash,
                "event_type": e.event_type,
                "affected_definitions": e.affected_definitions,
                "before_hash": e.before_hash,
                "after_hash": e.after_hash,
                "confidence": e.confidence,
                "description": e.description,
            }
            for e in events
        ]

        return {
            "project_id": project_id,
            "total_events": len(event_responses),
            "refactoring_events": event_responses
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to detect refactoring events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/evolution/hotspots")
async def get_hotspots(
    project_id: str,
    max_commits: int = Query(default=100, ge=1, le=500)
):
    """Get file hotspots (frequently changed files)."""
    from code_cartographer.api.routes.analysis import get_project

    try:
        project_data = await get_project(project_id)
        project_path = Path(project_data.get("project_path"))
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    try:
        analyzer = _get_temporal_analyzer(project_path)
        commits = analyzer.git_analyzer.get_commit_history(max_commits=max_commits)
        metrics = analyzer.git_analyzer.build_temporal_complexity_graph(commits)

        return {
            "project_id": project_id,
            "hotspots": [
                {"file_path": path, "change_count": count}
                for path, count in metrics.hotspots
            ],
            "file_churn": metrics.file_churn,
            "contributor_stats": metrics.contributor_stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze hotspots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/files/{file_path:path}/history")
async def get_file_history(
    project_id: str,
    file_path: str,
    max_commits: int = Query(default=50, ge=1, le=200)
):
    """Get commit history for a specific file."""
    from code_cartographer.api.routes.analysis import get_project

    try:
        project_data = await get_project(project_id)
        project_path = Path(project_data.get("project_path"))
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    try:
        analyzer = _get_temporal_analyzer(project_path)
        commits = analyzer.git_analyzer.get_file_history(file_path, max_commits=max_commits)

        commit_responses = [
            {
                "hash": c.hash,
                "short_hash": c.short_hash,
                "timestamp": c.timestamp.isoformat(),
                "author": c.author,
                "message": c.message,
            }
            for c in commits
        ]

        return {
            "project_id": project_id,
            "file_path": file_path,
            "total_commits": len(commit_responses),
            "commits": commit_responses
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
