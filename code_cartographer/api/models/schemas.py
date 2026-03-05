"""
Pydantic schemas for API validation and serialization.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# Request Models

class ProjectAnalysisRequest(BaseModel):
    """Request to analyze a project."""
    project_path: str = Field(..., description="Path to the project directory")
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="Patterns to exclude from analysis"
    )
    max_commits: Optional[int] = Field(
        default=100,
        description="Maximum number of commits to analyze for temporal data"
    )
    include_temporal: bool = Field(
        default=False,
        description="Whether to include git history analysis"
    )


# Response Models

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ComplexityMetrics(BaseModel):
    """Code complexity metrics."""
    cyclomatic: Optional[int] = None
    cognitive: Optional[int] = None
    halstead: Optional[Dict[str, Any]] = None
    maintainability_index: Optional[float] = None


class DefinitionResponse(BaseModel):
    """Code definition (function, class, etc.)."""
    name: str
    type: str  # 'function', 'class', 'method'
    file_path: str
    line_start: int
    line_end: int
    complexity: Optional[ComplexityMetrics] = None
    calls: Optional[List[str]] = None
    docstring: Optional[str] = None


class FileMetadataResponse(BaseModel):
    """File metadata and metrics."""
    path: str
    lines_of_code: int
    complexity: Optional[ComplexityMetrics] = None
    definitions: List[DefinitionResponse]
    imports: List[str]
    hash: Optional[str] = None


class ProjectAnalysisResponse(BaseModel):
    """Complete project analysis results."""
    project_path: str
    analysis_timestamp: datetime
    total_files: int
    total_lines: int
    files: List[FileMetadataResponse]
    dependencies: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None


class DependencyGraphResponse(BaseModel):
    """Dependency graph data."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None


class CommitResponse(BaseModel):
    """Git commit information."""
    hash: str
    short_hash: str
    timestamp: datetime
    author: str
    author_email: str
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    analysis_data: Optional[Dict[str, Any]] = None


class RefactoringEventResponse(BaseModel):
    """Refactoring event information."""
    commit_hash: str
    event_type: str
    affected_definitions: List[str]
    before_hash: str
    after_hash: str
    confidence: float
    description: Optional[str] = None


class ComplexityTrendResponse(BaseModel):
    """Complexity trend for a file."""
    file_path: str
    timeline: List[Dict[str, Any]]  # [{timestamp, complexity}]
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    max_complexity: float
    min_complexity: float
    current_complexity: float


class TemporalMetricsResponse(BaseModel):
    """Temporal metrics."""
    complexity_timeline: List[Dict[str, Any]]
    file_churn: Dict[str, int]
    hotspots: List[Dict[str, Any]]
    contributor_stats: Dict[str, int]


class TemporalDataResponse(BaseModel):
    """Temporal analysis results."""
    repository_path: str
    analysis_start: datetime
    analysis_end: datetime
    total_commits_analyzed: int
    commit_snapshots: List[CommitResponse]
    complexity_trends: List[ComplexityTrendResponse]
    refactoring_events: List[RefactoringEventResponse]
    temporal_metrics: TemporalMetricsResponse


class VariantResponse(BaseModel):
    """Code variant information."""
    variant_id: str
    files: List[str]
    similarity_score: float
    definition_type: str
    normalized_code: Optional[str] = None


class ProjectListResponse(BaseModel):
    """List of analyzed projects."""
    projects: List[Dict[str, Any]]


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # 'progress', 'complete', 'error'
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
