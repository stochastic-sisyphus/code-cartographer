"""
Tests for GitAnalyzer module.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from code_cartographer.core.git_analyzer import (
    GitAnalyzer,
    CommitSnapshot,
    RefactoringEvent,
    TemporalMetrics,
)


@pytest.fixture
def mock_repo():
    """Create a mock git repository."""
    repo = MagicMock()
    repo.bare = False
    repo.working_dir = "/fake/repo"
    return repo


@pytest.fixture
def mock_commit():
    """Create a mock commit object."""
    commit = MagicMock()
    commit.hexsha = "abcdef1234567890" * 2 + "abcdef12"  # 40 chars
    commit.committed_date = 1704067200  # 2024-01-01
    commit.author.name = "Test Author"
    commit.author.email = "test@example.com"
    commit.message = "Test commit message"
    commit.stats.total = {"insertions": 10, "deletions": 5}
    commit.parents = []
    return commit


class TestGitAnalyzer:
    """Test suite for GitAnalyzer."""

    def test_init_requires_gitpython(self, tmp_path):
        """Test that GitAnalyzer requires gitpython to be installed."""
        # GitPython should be installed for tests
        try:
            analyzer = GitAnalyzer(tmp_path / "non_existent")
        except ValueError as e:
            assert "Not a git repository" in str(e)

    def test_init_with_invalid_repo(self, tmp_path):
        """Test initialization with non-git directory."""
        non_git_dir = tmp_path / "not_a_repo"
        non_git_dir.mkdir()

        with pytest.raises(ValueError, match="Not a git repository"):
            GitAnalyzer(non_git_dir)

    @patch("code_cartographer.core.git_analyzer.Repo")
    def test_get_commit_history(self, mock_repo_class, mock_commit, tmp_path):
        """Test retrieving commit history."""
        # Setup mock
        mock_repo = Mock()
        mock_repo.bare = False
        mock_repo.iter_commits.return_value = [mock_commit]
        mock_repo_class.return_value = mock_repo

        analyzer = GitAnalyzer(tmp_path)
        commits = analyzer.get_commit_history(max_commits=10)

        assert len(commits) == 1
        assert commits[0].hash == mock_commit.hexsha
        assert commits[0].author == "Test Author"
        assert commits[0].insertions == 10
        assert commits[0].deletions == 5

    @patch("code_cartographer.core.git_analyzer.Repo")
    def test_get_commit_history_with_file_pattern(
        self, mock_repo_class, mock_commit, tmp_path
    ):
        """Test filtering commits by file pattern."""
        # Setup mock commit with Python file change
        mock_diff = Mock()
        mock_diff.renamed_file = False
        mock_diff.a_path = "test.py"
        mock_diff.b_path = None

        mock_parent = Mock()
        mock_parent.diff.return_value = [mock_diff]
        mock_commit.parents = [mock_parent]

        mock_repo = Mock()
        mock_repo.bare = False
        mock_repo.iter_commits.return_value = [mock_commit]
        mock_repo_class.return_value = mock_repo

        analyzer = GitAnalyzer(tmp_path)
        commits = analyzer.get_commit_history(max_commits=10, file_pattern=".py")

        assert len(commits) == 1
        assert "test.py" in commits[0].files_changed

    @patch("code_cartographer.core.git_analyzer.Repo")
    def test_detect_refactoring_events(self, mock_repo_class, tmp_path):
        """Test detection of refactoring events from commit messages."""
        # Create commits with refactoring keywords
        commit1 = Mock()
        commit1.hexsha = "abc" * 13 + "a"
        commit1.committed_date = 1704067200
        commit1.author.name = "Author 1"
        commit1.author.email = "author1@example.com"
        commit1.message = "Refactor authentication module"
        commit1.stats.total = {"insertions": 50, "deletions": 30}
        commit1.parents = []

        commit2 = Mock()
        commit2.hexsha = "def" * 13 + "d"
        commit2.committed_date = 1704153600
        commit2.author.name = "Author 2"
        commit2.author.email = "author2@example.com"
        commit2.message = "Add new feature"
        commit2.stats.total = {"insertions": 20, "deletions": 0}
        commit2.parents = []

        # Setup mock for file changes
        mock_diff1 = Mock()
        mock_diff1.renamed_file = False
        mock_diff1.a_path = "auth.py"
        mock_diff1.b_path = None

        mock_parent1 = Mock()
        mock_parent1.diff.return_value = [mock_diff1]
        commit2.parents = [mock_parent1]

        mock_diff2 = Mock()
        mock_diff2.renamed_file = False
        mock_diff2.a_path = "feature.py"
        mock_diff2.b_path = None

        mock_parent2 = Mock()
        mock_parent2.diff.return_value = [mock_diff2]

        mock_repo = Mock()
        mock_repo.bare = False
        mock_repo.iter_commits.return_value = [commit2, commit1]
        mock_repo_class.return_value = mock_repo

        analyzer = GitAnalyzer(tmp_path)
        events = analyzer.detect_refactoring_events(max_commits=10)

        # Should detect the refactor keyword in commit1
        refactor_events = [e for e in events if e.event_type == "extract"]
        assert len(refactor_events) >= 1
        assert any("Refactor" in e.description for e in refactor_events)

    @patch("code_cartographer.core.git_analyzer.Repo")
    def test_detect_file_rename(self, mock_repo_class, tmp_path):
        """Test detection of file renames."""
        commit1 = Mock()
        commit1.hexsha = "abc" * 13 + "a"
        commit1.committed_date = 1704067200
        commit1.author.name = "Author"
        commit1.author.email = "author@example.com"
        commit1.message = "Rename file"
        commit1.stats.total = {"insertions": 0, "deletions": 0}
        commit1.parents = []

        commit2 = Mock()
        commit2.hexsha = "def" * 13 + "d"
        commit2.committed_date = 1704153600
        commit2.author.name = "Author"
        commit2.author.email = "author@example.com"
        commit2.message = "Regular commit"
        commit2.stats.total = {"insertions": 0, "deletions": 0}
        commit2.parents = []

        # Setup file rename
        mock_diff = Mock()
        mock_diff.renamed_file = True
        mock_diff.rename_from = "old_name.py"
        mock_diff.rename_to = "new_name.py"

        mock_parent = Mock()
        mock_parent.diff.return_value = [mock_diff]
        commit2.parents = [mock_parent]

        mock_repo = Mock()
        mock_repo.bare = False
        mock_repo.iter_commits.return_value = [commit2, commit1]
        mock_repo_class.return_value = mock_repo

        analyzer = GitAnalyzer(tmp_path)
        events = analyzer.detect_refactoring_events(max_commits=10)

        rename_events = [e for e in events if e.event_type == "rename"]
        assert len(rename_events) >= 1
        assert rename_events[0].confidence == 1.0  # File rename is certain

    @patch("code_cartographer.core.git_analyzer.Repo")
    def test_build_temporal_complexity_graph(self, mock_repo_class, tmp_path):
        """Test building temporal metrics from commits."""
        # Create commit snapshots with analysis data
        commit1 = CommitSnapshot(
            hash="abc123",
            short_hash="abc123"[:7],
            timestamp=datetime(2024, 1, 1),
            author="Author 1",
            author_email="author1@example.com",
            message="Commit 1",
            files_changed=["file1.py", "file2.py"],
            insertions=10,
            deletions=5,
            analysis_data={
                "files": [
                    {"path": "file1.py", "metrics": {"cyclomatic": 5}},
                    {"path": "file2.py", "metrics": {"cyclomatic": 10}},
                ]
            },
        )

        commit2 = CommitSnapshot(
            hash="def456",
            short_hash="def456"[:7],
            timestamp=datetime(2024, 1, 2),
            author="Author 2",
            author_email="author2@example.com",
            message="Commit 2",
            files_changed=["file1.py"],
            insertions=5,
            deletions=2,
            analysis_data={
                "files": [
                    {"path": "file1.py", "metrics": {"cyclomatic": 7}},
                ]
            },
        )

        mock_repo = Mock()
        mock_repo.bare = False
        mock_repo_class.return_value = mock_repo

        analyzer = GitAnalyzer(tmp_path)
        metrics = analyzer.build_temporal_complexity_graph([commit1, commit2])

        assert len(metrics.complexity_timeline) == 2
        assert metrics.file_churn["file1.py"] == 2
        assert metrics.file_churn["file2.py"] == 1
        assert len(metrics.hotspots) > 0
        assert metrics.contributor_stats["Author 1"] == 1
        assert metrics.contributor_stats["Author 2"] == 1

    @patch("code_cartographer.core.git_analyzer.Repo")
    def test_get_current_branch(self, mock_repo_class, tmp_path):
        """Test getting current branch name."""
        mock_repo = Mock()
        mock_repo.bare = False
        mock_repo.active_branch.name = "main"
        mock_repo_class.return_value = mock_repo

        analyzer = GitAnalyzer(tmp_path)
        branch = analyzer.get_current_branch()

        assert branch == "main"

    @patch("code_cartographer.core.git_analyzer.Repo")
    def test_is_dirty(self, mock_repo_class, tmp_path):
        """Test checking for uncommitted changes."""
        mock_repo = Mock()
        mock_repo.bare = False
        mock_repo.is_dirty.return_value = True
        mock_repo_class.return_value = mock_repo

        analyzer = GitAnalyzer(tmp_path)
        assert analyzer.is_dirty() is True

    @patch("code_cartographer.core.git_analyzer.Repo")
    def test_get_remote_url(self, mock_repo_class, tmp_path):
        """Test getting remote origin URL."""
        mock_repo = Mock()
        mock_repo.bare = False
        mock_repo.remotes.origin.url = "https://github.com/user/repo.git"
        mock_repo_class.return_value = mock_repo

        analyzer = GitAnalyzer(tmp_path)
        url = analyzer.get_remote_url()

        assert url == "https://github.com/user/repo.git"


class TestCommitSnapshot:
    """Test suite for CommitSnapshot dataclass."""

    def test_commit_snapshot_creation(self):
        """Test creating a CommitSnapshot."""
        snapshot = CommitSnapshot(
            hash="abc123",
            short_hash="abc123"[:7],
            timestamp=datetime(2024, 1, 1),
            author="Test Author",
            author_email="test@example.com",
            message="Test commit",
            files_changed=["file1.py"],
            insertions=10,
            deletions=5,
        )

        assert snapshot.hash == "abc123"
        assert snapshot.author == "Test Author"
        assert snapshot.insertions == 10


class TestRefactoringEvent:
    """Test suite for RefactoringEvent dataclass."""

    def test_refactoring_event_creation(self):
        """Test creating a RefactoringEvent."""
        event = RefactoringEvent(
            commit_hash="abc123",
            event_type="rename",
            affected_definitions=["old_function"],
            before_hash="before123",
            after_hash="after123",
            confidence=0.9,
            description="Renamed function",
        )

        assert event.event_type == "rename"
        assert event.confidence == 0.9
