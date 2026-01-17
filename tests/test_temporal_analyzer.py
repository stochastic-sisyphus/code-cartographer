"""
Tests for TemporalAnalyzer module.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from code_cartographer.core.temporal_analyzer import (
    TemporalAnalyzer,
    ComplexityTrend,
    VariantEvolution,
    TemporalData,
)
from code_cartographer.core.git_analyzer import CommitSnapshot, GitAnalyzer


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_git_analyzer():
    """Create a mock GitAnalyzer."""
    analyzer = Mock(spec=GitAnalyzer)
    return analyzer


@pytest.fixture
def sample_commits():
    """Create sample commit snapshots."""
    return [
        CommitSnapshot(
            hash="abc123",
            short_hash="abc123"[:7],
            timestamp=datetime(2024, 1, 1),
            author="Author 1",
            author_email="author1@example.com",
            message="Initial commit",
            files_changed=["file1.py"],
            insertions=100,
            deletions=0,
            analysis_data={
                "files": [
                    {"path": "file1.py", "metrics": {"cyclomatic": 5}},
                ]
            },
        ),
        CommitSnapshot(
            hash="def456",
            short_hash="def456"[:7],
            timestamp=datetime(2024, 1, 2),
            author="Author 2",
            author_email="author2@example.com",
            message="Add feature",
            files_changed=["file1.py", "file2.py"],
            insertions=50,
            deletions=10,
            analysis_data={
                "files": [
                    {"path": "file1.py", "metrics": {"cyclomatic": 7}},
                    {"path": "file2.py", "metrics": {"cyclomatic": 3}},
                ]
            },
        ),
        CommitSnapshot(
            hash="ghi789",
            short_hash="ghi789"[:7],
            timestamp=datetime(2024, 1, 3),
            author="Author 1",
            author_email="author1@example.com",
            message="Refactor code",
            files_changed=["file1.py"],
            insertions=20,
            deletions=15,
            analysis_data={
                "files": [
                    {"path": "file1.py", "metrics": {"cyclomatic": 4}},
                ]
            },
        ),
    ]


class TestTemporalAnalyzer:
    """Test suite for TemporalAnalyzer."""

    @patch("code_cartographer.core.temporal_analyzer.GitAnalyzer")
    def test_init(self, mock_git_class, tmp_path, temp_cache_dir):
        """Test TemporalAnalyzer initialization."""
        mock_git_class.return_value = Mock(spec=GitAnalyzer)

        analyzer = TemporalAnalyzer(
            repo_path=tmp_path, cache_dir=temp_cache_dir
        )

        assert analyzer.repo_path == tmp_path
        assert analyzer.cache_dir == temp_cache_dir
        assert analyzer.cache_dir.exists()

    @patch("code_cartographer.core.temporal_analyzer.GitAnalyzer")
    def test_sample_commits_uniform(
        self, mock_git_class, tmp_path, sample_commits
    ):
        """Test uniform commit sampling."""
        mock_git = Mock(spec=GitAnalyzer)
        mock_git_class.return_value = mock_git

        analyzer = TemporalAnalyzer(repo_path=tmp_path, git_analyzer=mock_git)

        # Sample 2 commits from 3
        sampled = analyzer._sample_commits(sample_commits, 2, "uniform")

        assert len(sampled) == 2
        assert sampled[0].hash == "abc123"
        assert sampled[1].hash == "def456"

    @patch("code_cartographer.core.temporal_analyzer.GitAnalyzer")
    def test_sample_commits_all(self, mock_git_class, tmp_path, sample_commits):
        """Test 'all' sampling strategy."""
        mock_git = Mock(spec=GitAnalyzer)
        mock_git_class.return_value = mock_git

        analyzer = TemporalAnalyzer(repo_path=tmp_path, git_analyzer=mock_git)
        sampled = analyzer._sample_commits(sample_commits, 2, "all")

        assert len(sampled) == 2

    @patch("code_cartographer.core.temporal_analyzer.GitAnalyzer")
    def test_sample_commits_major(self, mock_git_class, tmp_path, sample_commits):
        """Test 'major' sampling strategy."""
        mock_git = Mock(spec=GitAnalyzer)
        mock_git_class.return_value = mock_git

        analyzer = TemporalAnalyzer(repo_path=tmp_path, git_analyzer=mock_git)
        sampled = analyzer._sample_commits(sample_commits, 2, "major")

        assert len(sampled) == 2
        # Should prioritize commit with most changes (initial commit)
        assert sampled[0].hash == "abc123"

    @patch("code_cartographer.core.temporal_analyzer.GitAnalyzer")
    def test_detect_complexity_trends(
        self, mock_git_class, tmp_path, sample_commits
    ):
        """Test complexity trend detection."""
        mock_git = Mock(spec=GitAnalyzer)
        mock_git_class.return_value = mock_git

        analyzer = TemporalAnalyzer(repo_path=tmp_path, git_analyzer=mock_git)
        trends = analyzer.detect_complexity_trends(sample_commits)

        assert len(trends) > 0

        # Check file1.py trend (5 -> 7 -> 4)
        file1_trend = next(t for t in trends if t.file_path == "file1.py")
        assert file1_trend.max_complexity == 7
        assert file1_trend.min_complexity == 4
        assert file1_trend.current_complexity == 4
        assert len(file1_trend.timeline) == 3

    @patch("code_cartographer.core.temporal_analyzer.GitAnalyzer")
    def test_analyze_evolution_no_commits(self, mock_git_class, tmp_path):
        """Test analyze_evolution with no commits."""
        mock_git = Mock(spec=GitAnalyzer)
        mock_git.get_commit_history.return_value = []
        mock_git_class.return_value = mock_git

        analyzer = TemporalAnalyzer(repo_path=tmp_path, git_analyzer=mock_git)
        result = analyzer.analyze_evolution()

        assert result.total_commits_analyzed == 0
        assert len(result.commit_snapshots) == 0
        assert len(result.complexity_trends) == 0

    @patch("code_cartographer.core.temporal_analyzer.GitAnalyzer")
    @patch("code_cartographer.core.temporal_analyzer.ProjectAnalyzer")
    def test_analyze_evolution_with_commits(
        self, mock_project_analyzer, mock_git_class, tmp_path, sample_commits
    ):
        """Test analyze_evolution with sample commits."""
        mock_git = Mock(spec=GitAnalyzer)
        mock_git.get_commit_history.return_value = sample_commits
        mock_git.detect_refactoring_events.return_value = []
        mock_git.build_temporal_complexity_graph.return_value = Mock(
            complexity_timeline=[],
            file_churn={},
            hotspots=[],
            contributor_stats={},
        )
        mock_git_class.return_value = mock_git

        analyzer = TemporalAnalyzer(repo_path=tmp_path, git_analyzer=mock_git)

        # Mock the cache to avoid actual git operations
        analyzer._analyze_commit_cached = Mock(
            return_value={"files": [{"path": "test.py", "metrics": {"cyclomatic": 5}}]}
        )

        result = analyzer.analyze_evolution(max_commits=5)

        assert result.total_commits_analyzed > 0
        assert len(result.commit_snapshots) > 0
        assert result.repository_path == str(tmp_path)

    @patch("code_cartographer.core.temporal_analyzer.GitAnalyzer")
    def test_cache_operations(self, mock_git_class, tmp_path, temp_cache_dir):
        """Test cache operations."""
        mock_git = Mock(spec=GitAnalyzer)
        mock_git_class.return_value = mock_git

        analyzer = TemporalAnalyzer(
            repo_path=tmp_path,
            git_analyzer=mock_git,
            cache_dir=temp_cache_dir,
        )

        # Test cache size
        assert analyzer.get_cache_size() == 0

        # Create a fake cache file
        cache_file = temp_cache_dir / "test_hash.json"
        cache_file.write_text(json.dumps({"test": "data"}))

        assert analyzer.get_cache_size() == 1

        # Clear cache
        analyzer.clear_cache()
        assert analyzer.get_cache_size() == 0


class TestComplexityTrend:
    """Test suite for ComplexityTrend dataclass."""

    def test_complexity_trend_creation(self):
        """Test creating a ComplexityTrend."""
        timeline = [
            (datetime(2024, 1, 1), 5.0),
            (datetime(2024, 1, 2), 7.0),
            (datetime(2024, 1, 3), 4.0),
        ]

        trend = ComplexityTrend(
            file_path="test.py",
            timeline=timeline,
            trend_direction="decreasing",
            max_complexity=7.0,
            min_complexity=4.0,
            current_complexity=4.0,
        )

        assert trend.file_path == "test.py"
        assert len(trend.timeline) == 3
        assert trend.trend_direction == "decreasing"
        assert trend.max_complexity == 7.0


class TestTemporalData:
    """Test suite for TemporalData dataclass."""

    def test_temporal_data_creation(self):
        """Test creating a TemporalData object."""
        now = datetime.now()

        data = TemporalData(
            repository_path="/fake/repo",
            analysis_start=now,
            analysis_end=now,
            total_commits_analyzed=5,
            commit_snapshots=[],
            complexity_trends=[],
            variant_evolution=[],
            refactoring_events=[],
            temporal_metrics=Mock(
                complexity_timeline=[],
                file_churn={},
                hotspots=[],
                contributor_stats={},
            ),
        )

        assert data.repository_path == "/fake/repo"
        assert data.total_commits_analyzed == 5
        assert len(data.commit_snapshots) == 0
