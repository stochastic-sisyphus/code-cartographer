"""
Git History Analysis Module
===========================
Analyzes code evolution through git history to enable temporal code visualization.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import git
    from git import Repo

    _HAS_GIT = True
except ImportError:
    _HAS_GIT = False

logger = logging.getLogger(__name__)


@dataclass
class CommitSnapshot:
    """Represents a snapshot of the codebase at a specific commit."""

    hash: str
    short_hash: str
    timestamp: datetime
    author: str
    author_email: str
    message: str
    files_changed: List[str]
    insertions: int = 0
    deletions: int = 0
    analysis_data: Optional[Dict] = None


@dataclass
class RefactoringEvent:
    """Represents a detected refactoring event in the git history."""

    commit_hash: str
    event_type: str  # 'rename', 'split', 'merge', 'move', 'extract'
    affected_definitions: List[str]
    before_hash: str
    after_hash: str
    confidence: float = 1.0
    description: Optional[str] = None


@dataclass
class TemporalMetrics:
    """Aggregated metrics over time."""

    complexity_timeline: List[Tuple[datetime, float]]
    file_churn: Dict[str, int]
    hotspots: List[Tuple[str, int]]  # (file_path, change_count)
    contributor_stats: Dict[str, int]


class GitAnalyzer:
    """Analyzes code evolution through git history."""

    def __init__(self, repo_path: Path):
        """
        Initialize the GitAnalyzer.

        Args:
            repo_path: Path to the git repository root

        Raises:
            ImportError: If gitpython is not installed
            git.exc.InvalidGitRepositoryError: If path is not a git repository
        """
        if not _HAS_GIT:
            raise ImportError(
                "GitPython is required for git analysis. "
                "Install with: pip install gitpython"
            )

        self.repo_path = Path(repo_path).resolve()
        try:
            self.repo = Repo(self.repo_path)
        except git.exc.InvalidGitRepositoryError as e:
            raise ValueError(f"Not a git repository: {repo_path}") from e

        if self.repo.bare:
            raise ValueError(f"Cannot analyze bare repository: {repo_path}")

        logger.info(f"Initialized GitAnalyzer for {repo_path}")

    def get_commit_history(
        self,
        max_commits: int = 100,
        branch: str = "HEAD",
        file_pattern: Optional[str] = None,
    ) -> List[CommitSnapshot]:
        """
        Extract commit history with metadata.

        Args:
            max_commits: Maximum number of commits to retrieve
            branch: Branch or ref to analyze
            file_pattern: Optional file pattern to filter commits

        Returns:
            List of CommitSnapshot objects, newest first
        """
        commits = []

        try:
            commit_iter = self.repo.iter_commits(branch, max_count=max_commits)

            for commit in commit_iter:
                # Get files changed in this commit
                files_changed = []
                insertions = 0
                deletions = 0

                if commit.parents:
                    parent = commit.parents[0]
                    diffs = parent.diff(commit)

                    for diff in diffs:
                        # Handle renamed files
                        if diff.renamed_file:
                            files_changed.append(f"{diff.rename_from} -> {diff.rename_to}")
                        elif diff.a_path:
                            files_changed.append(diff.a_path)
                        elif diff.b_path:
                            files_changed.append(diff.b_path)

                    # Get stats
                    stats = commit.stats.total
                    insertions = stats.get("insertions", 0)
                    deletions = stats.get("deletions", 0)

                # Filter by file pattern if provided
                if file_pattern:
                    files_changed = [
                        f for f in files_changed if f.endswith(file_pattern)
                    ]
                    if not files_changed:
                        continue

                snapshot = CommitSnapshot(
                    hash=commit.hexsha,
                    short_hash=commit.hexsha[:7],
                    timestamp=datetime.fromtimestamp(commit.committed_date),
                    author=commit.author.name,
                    author_email=commit.author.email,
                    message=commit.message.strip(),
                    files_changed=files_changed,
                    insertions=insertions,
                    deletions=deletions,
                )

                commits.append(snapshot)

            logger.info(f"Retrieved {len(commits)} commits from {branch}")
            return commits

        except git.exc.GitCommandError as e:
            logger.error(f"Git command failed: {e}")
            return []

    def analyze_at_commit(
        self, commit_hash: str, analyzer_func=None
    ) -> Optional[Dict]:
        """
        Checkout and analyze code at a specific commit.

        Args:
            commit_hash: The commit hash to analyze
            analyzer_func: Optional function to run analysis (receives repo_path)

        Returns:
            Analysis results dictionary or None if checkout fails
        """
        try:
            # Store current HEAD
            original_head = self.repo.head.commit

            # Checkout the target commit
            self.repo.git.checkout(commit_hash)
            logger.info(f"Checked out commit {commit_hash[:7]}")

            # Run analysis if function provided
            result = None
            if analyzer_func:
                result = analyzer_func(self.repo_path)

            # Return to original HEAD
            self.repo.git.checkout(original_head.hexsha)
            logger.info(f"Returned to commit {original_head.hexsha[:7]}")

            return result

        except git.exc.GitCommandError as e:
            logger.error(f"Failed to checkout commit {commit_hash}: {e}")
            # Try to return to original state
            try:
                self.repo.git.checkout(original_head.hexsha)
            except:
                logger.warning("Failed to return to original commit")
            return None

    def detect_refactoring_events(
        self, max_commits: int = 50
    ) -> List[RefactoringEvent]:
        """
        Detect potential refactoring events in git history.

        Args:
            max_commits: Number of commits to analyze

        Returns:
            List of detected refactoring events
        """
        events = []
        commits = self.get_commit_history(max_commits=max_commits)

        for i, commit in enumerate(commits):
            if i == 0:
                continue  # Skip first commit (no parent to compare)

            # Look for refactoring patterns in commit message
            msg_lower = commit.message.lower()
            refactoring_keywords = {
                "rename": ["rename", "renamed"],
                "move": ["move", "moved", "relocate"],
                "split": ["split", "divide", "separate"],
                "merge": ["merge", "combine", "consolidate"],
                "extract": ["extract", "refactor"],
            }

            for event_type, keywords in refactoring_keywords.items():
                if any(kw in msg_lower for kw in keywords):
                    event = RefactoringEvent(
                        commit_hash=commit.hash,
                        event_type=event_type,
                        affected_definitions=[],
                        before_hash=commits[i - 1].hash,
                        after_hash=commit.hash,
                        confidence=0.7,  # Based on commit message only
                        description=commit.message.split("\n")[0],
                    )
                    events.append(event)

            # Detect renames from file changes
            for file_change in commit.files_changed:
                if "->" in file_change:
                    old_path, new_path = file_change.split(" -> ")
                    event = RefactoringEvent(
                        commit_hash=commit.hash,
                        event_type="rename",
                        affected_definitions=[old_path, new_path],
                        before_hash=commits[i - 1].hash,
                        after_hash=commit.hash,
                        confidence=1.0,  # File rename is certain
                        description=f"Renamed {old_path} to {new_path}",
                    )
                    events.append(event)

        logger.info(f"Detected {len(events)} refactoring events")
        return events

    def build_temporal_complexity_graph(
        self, commits: List[CommitSnapshot]
    ) -> TemporalMetrics:
        """
        Build temporal metrics from commit history.

        Args:
            commits: List of commit snapshots with analysis_data

        Returns:
            TemporalMetrics object with aggregated data
        """
        complexity_timeline = []
        file_churn: Dict[str, int] = {}
        contributor_stats: Dict[str, int] = {}

        for commit in commits:
            # Track contributor activity
            contributor_stats[commit.author] = (
                contributor_stats.get(commit.author, 0) + 1
            )

            # Track file churn
            for file_path in commit.files_changed:
                # Skip arrow notation from renames
                if "->" not in file_path:
                    file_churn[file_path] = file_churn.get(file_path, 0) + 1

            # Extract complexity if analysis data available
            if commit.analysis_data:
                avg_complexity = self._extract_avg_complexity(commit.analysis_data)
                if avg_complexity is not None:
                    complexity_timeline.append((commit.timestamp, avg_complexity))

        # Sort file churn to find hotspots
        hotspots = sorted(file_churn.items(), key=lambda x: x[1], reverse=True)[:20]

        return TemporalMetrics(
            complexity_timeline=complexity_timeline,
            file_churn=file_churn,
            hotspots=hotspots,
            contributor_stats=contributor_stats,
        )

    def _extract_avg_complexity(self, analysis_data: Dict) -> Optional[float]:
        """Extract average cyclomatic complexity from analysis data."""
        try:
            files = analysis_data.get("files", [])
            if not files:
                return None

            total_complexity = 0
            count = 0

            for file_data in files:
                if isinstance(file_data, dict):
                    metrics = file_data.get("metrics", {})
                    if metrics and isinstance(metrics, dict):
                        cc = metrics.get("cyclomatic")
                        if cc is not None:
                            total_complexity += cc
                            count += 1

            return total_complexity / count if count > 0 else None

        except Exception as e:
            logger.warning(f"Failed to extract complexity: {e}")
            return None

    def get_file_history(self, file_path: str, max_commits: int = 50) -> List[CommitSnapshot]:
        """
        Get commit history for a specific file.

        Args:
            file_path: Path to the file relative to repo root
            max_commits: Maximum number of commits to retrieve

        Returns:
            List of commits that modified this file
        """
        commits = []

        try:
            commit_iter = self.repo.iter_commits("HEAD", paths=file_path, max_count=max_commits)

            for commit in commit_iter:
                snapshot = CommitSnapshot(
                    hash=commit.hexsha,
                    short_hash=commit.hexsha[:7],
                    timestamp=datetime.fromtimestamp(commit.committed_date),
                    author=commit.author.name,
                    author_email=commit.author.email,
                    message=commit.message.strip(),
                    files_changed=[file_path],
                )
                commits.append(snapshot)

            logger.info(f"Retrieved {len(commits)} commits for {file_path}")
            return commits

        except git.exc.GitCommandError as e:
            logger.error(f"Failed to get file history: {e}")
            return []

    def get_current_branch(self) -> str:
        """Get the name of the current branch."""
        try:
            return self.repo.active_branch.name
        except TypeError:
            # Detached HEAD state
            return "HEAD"

    def is_dirty(self) -> bool:
        """Check if repository has uncommitted changes."""
        return self.repo.is_dirty()

    def get_remote_url(self) -> Optional[str]:
        """Get the URL of the remote origin."""
        try:
            return self.repo.remotes.origin.url
        except (AttributeError, ValueError):
            return None
