"""
Temporal Analysis Module
========================
Orchestrates multi-commit code analysis to track evolution over time.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from code_cartographer.core.analyzer import ProjectAnalyzer
from code_cartographer.core.git_analyzer import (
    CommitSnapshot,
    GitAnalyzer,
    RefactoringEvent,
    TemporalMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ComplexityTrend:
    """Represents complexity evolution over time."""

    file_path: str
    timeline: List[Tuple[datetime, float]]  # (timestamp, complexity)
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    max_complexity: float
    min_complexity: float
    current_complexity: float


@dataclass
class VariantEvolution:
    """Tracks how code variants change over time."""

    variant_group_id: str
    first_seen: datetime
    last_seen: datetime
    lifecycle: str  # 'created', 'merged', 'split', 'active'
    member_count_timeline: List[Tuple[datetime, int]]


@dataclass
class TemporalData:
    """Complete temporal analysis data."""

    repository_path: str
    analysis_start: datetime
    analysis_end: datetime
    total_commits_analyzed: int
    commit_snapshots: List[CommitSnapshot]
    complexity_trends: List[ComplexityTrend]
    variant_evolution: List[VariantEvolution]
    refactoring_events: List[RefactoringEvent]
    temporal_metrics: TemporalMetrics
    cache_dir: Optional[Path] = None


class TemporalAnalyzer:
    """Orchestrates multi-commit analysis for temporal code visualization."""

    def __init__(
        self,
        repo_path: Path,
        git_analyzer: Optional[GitAnalyzer] = None,
        project_analyzer: Optional[ProjectAnalyzer] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the TemporalAnalyzer.

        Args:
            repo_path: Path to the git repository
            git_analyzer: Optional GitAnalyzer instance
            project_analyzer: Optional ProjectAnalyzer instance
            cache_dir: Optional directory for caching analysis results
        """
        self.repo_path = Path(repo_path).resolve()
        self.git_analyzer = git_analyzer or GitAnalyzer(repo_path)
        self.project_analyzer = project_analyzer

        # Setup caching
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.repo_path / ".code-cartographer" / "cache"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized TemporalAnalyzer with cache at {self.cache_dir}")

    def analyze_evolution(
        self,
        commit_range: Optional[Tuple[str, str]] = None,
        max_commits: int = 100,
        sample_strategy: str = "uniform",
    ) -> TemporalData:
        """
        Analyze code evolution across multiple commits.

        Args:
            commit_range: Optional tuple of (start_commit, end_commit)
            max_commits: Maximum number of commits to analyze
            sample_strategy: Strategy for sampling commits ('uniform', 'major', 'all')

        Returns:
            TemporalData object with complete analysis
        """
        logger.info(f"Starting temporal analysis with {max_commits} max commits")
        analysis_start = datetime.now()

        # Get commit history
        commits = self.git_analyzer.get_commit_history(max_commits=max_commits)

        if not commits:
            logger.warning("No commits found in repository")
            return TemporalData(
                repository_path=str(self.repo_path),
                analysis_start=analysis_start,
                analysis_end=datetime.now(),
                total_commits_analyzed=0,
                commit_snapshots=[],
                complexity_trends=[],
                variant_evolution=[],
                refactoring_events=[],
                temporal_metrics=TemporalMetrics(
                    complexity_timeline=[],
                    file_churn={},
                    hotspots=[],
                    contributor_stats={},
                ),
            )

        # Sample commits based on strategy
        sampled_commits = self._sample_commits(commits, max_commits, sample_strategy)
        logger.info(f"Sampled {len(sampled_commits)} commits for analysis")

        # Analyze each sampled commit
        for commit in sampled_commits:
            analysis_data = self._analyze_commit_cached(commit.hash)
            commit.analysis_data = analysis_data

        # Detect refactoring events
        refactoring_events = self.git_analyzer.detect_refactoring_events(
            max_commits=max_commits
        )

        # Build temporal metrics
        temporal_metrics = self.git_analyzer.build_temporal_complexity_graph(
            sampled_commits
        )

        # Detect complexity trends
        complexity_trends = self.detect_complexity_trends(sampled_commits)

        # Track variant lifecycle (placeholder for now)
        variant_evolution = []

        analysis_end = datetime.now()
        duration = (analysis_end - analysis_start).total_seconds()
        logger.info(f"Temporal analysis completed in {duration:.2f} seconds")

        return TemporalData(
            repository_path=str(self.repo_path),
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            total_commits_analyzed=len(sampled_commits),
            commit_snapshots=sampled_commits,
            complexity_trends=complexity_trends,
            variant_evolution=variant_evolution,
            refactoring_events=refactoring_events,
            temporal_metrics=temporal_metrics,
            cache_dir=self.cache_dir,
        )

    def detect_complexity_trends(
        self, commits: List[CommitSnapshot]
    ) -> List[ComplexityTrend]:
        """
        Detect complexity trends for files over time.

        Args:
            commits: List of commit snapshots with analysis data

        Returns:
            List of ComplexityTrend objects
        """
        file_complexity_map: Dict[str, List[Tuple[datetime, float]]] = {}

        # Build timeline for each file
        for commit in commits:
            if not commit.analysis_data:
                continue

            files = commit.analysis_data.get("files", [])
            for file_data in files:
                if not isinstance(file_data, dict):
                    continue

                file_path = file_data.get("path")
                metrics = file_data.get("metrics", {})

                if file_path and isinstance(metrics, dict):
                    complexity = metrics.get("cyclomatic")
                    if complexity is not None:
                        if file_path not in file_complexity_map:
                            file_complexity_map[file_path] = []
                        file_complexity_map[file_path].append(
                            (commit.timestamp, complexity)
                        )

        # Create trends for each file
        trends = []
        for file_path, timeline in file_complexity_map.items():
            if len(timeline) < 2:
                continue  # Need at least 2 points for a trend

            # Sort by timestamp
            timeline.sort(key=lambda x: x[0])

            complexities = [c for _, c in timeline]
            max_c = max(complexities)
            min_c = min(complexities)
            current_c = complexities[-1]

            # Determine trend direction
            first_half_avg = sum(complexities[: len(complexities) // 2]) / (
                len(complexities) // 2
            )
            second_half_avg = sum(complexities[len(complexities) // 2 :]) / (
                len(complexities) - len(complexities) // 2
            )

            if second_half_avg > first_half_avg * 1.2:
                direction = "increasing"
            elif second_half_avg < first_half_avg * 0.8:
                direction = "decreasing"
            else:
                direction = "stable"

            trend = ComplexityTrend(
                file_path=file_path,
                timeline=timeline,
                trend_direction=direction,
                max_complexity=max_c,
                min_complexity=min_c,
                current_complexity=current_c,
            )
            trends.append(trend)

        logger.info(f"Detected {len(trends)} complexity trends")
        return trends

    def track_variant_lifecycle(
        self, commits: List[CommitSnapshot]
    ) -> List[VariantEvolution]:
        """
        Track how code variants evolve over time.

        Args:
            commits: List of commit snapshots with variant analysis

        Returns:
            List of VariantEvolution objects
        """
        # TODO: Implement variant tracking logic
        # This requires running VariantAnalyzer at each commit
        # and tracking variant groups across commits
        logger.info("Variant lifecycle tracking not yet implemented")
        return []

    def _sample_commits(
        self, commits: List[CommitSnapshot], max_commits: int, strategy: str
    ) -> List[CommitSnapshot]:
        """
        Sample commits based on the specified strategy.

        Args:
            commits: Full list of commits
            max_commits: Maximum number of commits to sample
            strategy: Sampling strategy ('uniform', 'major', 'all')

        Returns:
            Sampled list of commits
        """
        if len(commits) <= max_commits:
            return commits

        if strategy == "all":
            return commits[:max_commits]

        elif strategy == "uniform":
            # Sample uniformly across the timeline
            step = len(commits) // max_commits
            return [commits[i * step] for i in range(max_commits)]

        elif strategy == "major":
            # Sample major commits (high line changes or refactoring keywords)
            scored_commits = []
            for commit in commits:
                score = commit.insertions + commit.deletions

                # Boost score for refactoring keywords
                msg_lower = commit.message.lower()
                refactoring_keywords = [
                    "refactor",
                    "restructure",
                    "major",
                    "breaking",
                    "release",
                ]
                if any(kw in msg_lower for kw in refactoring_keywords):
                    score *= 2

                scored_commits.append((score, commit))

            # Sort by score and take top N
            scored_commits.sort(reverse=True, key=lambda x: x[0])
            return [commit for _, commit in scored_commits[:max_commits]]

        else:
            logger.warning(f"Unknown sampling strategy: {strategy}, using uniform")
            return self._sample_commits(commits, max_commits, "uniform")

    def _analyze_commit_cached(self, commit_hash: str) -> Optional[Dict]:
        """
        Analyze a commit with caching support.

        Args:
            commit_hash: The commit hash to analyze

        Returns:
            Analysis results dictionary
        """
        cache_file = self.cache_dir / f"{commit_hash}.json"

        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    logger.debug(f"Loading cached analysis for {commit_hash[:7]}")
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache for {commit_hash[:7]}: {e}")

        # Perform analysis
        def analyze_at_commit(repo_path: Path) -> Dict:
            if not self.project_analyzer:
                self.project_analyzer = ProjectAnalyzer(root=repo_path)
            return self.project_analyzer.execute()

        result = self.git_analyzer.analyze_at_commit(commit_hash, analyze_at_commit)

        # Cache the result
        if result:
            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f, indent=2)
                logger.debug(f"Cached analysis for {commit_hash[:7]}")
            except IOError as e:
                logger.warning(f"Failed to cache analysis for {commit_hash[:7]}: {e}")

        return result

    def clear_cache(self) -> None:
        """Clear all cached analysis results."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared analysis cache")

    def get_cache_size(self) -> int:
        """Get the number of cached analyses."""
        if not self.cache_dir.exists():
            return 0
        return len(list(self.cache_dir.glob("*.json")))
