"""
Temporal Analysis Module
========================
Analyzes code evolution over time through git history to enable dynamic visualization
of how codebases grow, change, and evolve.
"""

import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CodeChange:
    """Represents a change to a code element."""
    
    timestamp: datetime
    commit_hash: str
    author: str
    change_type: str  # 'added', 'modified', 'deleted'
    lines_added: int
    lines_removed: int
    complexity_delta: Optional[float] = None


@dataclass
class CodeElement:
    """Represents a code element tracked over time."""
    
    name: str
    element_type: str  # 'function', 'class', 'module'
    file_path: str
    first_seen: datetime
    last_modified: datetime
    changes: List[CodeChange] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    
@dataclass
class TemporalSnapshot:
    """A snapshot of the codebase at a specific point in time."""
    
    timestamp: datetime
    commit_hash: str
    elements: Dict[str, CodeElement]
    relationships: List[Tuple[str, str, str]]  # (from, to, relationship_type)
    

class TemporalAnalyzer:
    """Analyzes code evolution through git history."""
    
    def __init__(self, repo_path: Path):
        """
        Initialize the temporal analyzer.
        
        Args:
            repo_path: Path to the git repository
        """
        self.repo_path = Path(repo_path)
        self.snapshots: List[TemporalSnapshot] = []
        
    def analyze_git_history(
        self, 
        max_commits: int = 100,
        file_patterns: Optional[List[str]] = None
    ) -> List[TemporalSnapshot]:
        """
        Analyze git history to build temporal evolution data.
        
        Args:
            max_commits: Maximum number of commits to analyze
            file_patterns: Optional list of file patterns to include (e.g., ['*.py'])
            
        Returns:
            List of temporal snapshots
        """
        commits = self._get_commits(max_commits)
        
        for commit_hash, timestamp, author in commits:
            elements = self._analyze_commit(commit_hash, file_patterns)
            relationships = self._extract_relationships(elements)
            
            snapshot = TemporalSnapshot(
                timestamp=timestamp,
                commit_hash=commit_hash,
                elements=elements,
                relationships=relationships
            )
            self.snapshots.append(snapshot)
            
        return self.snapshots
    
    def _get_commits(self, max_commits: int) -> List[Tuple[str, datetime, str]]:
        """
        Get commit history from git.
        
        Returns:
            List of (commit_hash, timestamp, author) tuples
        """
        try:
            result = subprocess.run(
                [
                    'git', 'log',
                    f'-{max_commits}',
                    '--pretty=format:%H|%at|%an',
                    '--reverse'
                ],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) >= 3:
                    commit_hash = parts[0]
                    timestamp = datetime.fromtimestamp(int(parts[1]))
                    author = parts[2]
                    commits.append((commit_hash, timestamp, author))
                    
            return commits
        except subprocess.CalledProcessError:
            return []
    
    def _analyze_commit(
        self, 
        commit_hash: str,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, CodeElement]:
        """
        Analyze a specific commit to extract code elements.
        
        Args:
            commit_hash: Git commit hash
            file_patterns: Optional file patterns to filter
            
        Returns:
            Dictionary of code elements
        """
        # This is a simplified implementation
        # In practice, this would parse the actual code at this commit
        elements = {}
        
        try:
            # Get list of files in this commit
            result = subprocess.run(
                ['git', 'ls-tree', '-r', '--name-only', commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = result.stdout.strip().split('\n')
            
            # Filter by patterns if provided
            if file_patterns:
                filtered_files = []
                for file in files:
                    if any(file.endswith(pattern.replace('*', '')) 
                          for pattern in file_patterns):
                        filtered_files.append(file)
                files = filtered_files
            
            # Create basic elements for each file
            for file_path in files:
                if file_path:
                    element = CodeElement(
                        name=Path(file_path).stem,
                        element_type='module',
                        file_path=file_path,
                        first_seen=datetime.now(),
                        last_modified=datetime.now()
                    )
                    elements[file_path] = element
                    
        except subprocess.CalledProcessError:
            pass
            
        return elements
    
    def _extract_relationships(
        self, 
        elements: Dict[str, CodeElement]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between code elements.
        
        Args:
            elements: Dictionary of code elements
            
        Returns:
            List of relationships as (from, to, type) tuples
        """
        relationships = []
        
        # Build dependency relationships based on imports
        # This is simplified - real implementation would parse AST
        for element_id, element in elements.items():
            for dep in element.dependencies:
                relationships.append((element_id, dep, 'depends_on'))
                
        return relationships
    
    def get_evolution_timeline(self) -> List[Dict]:
        """
        Get timeline data for visualization.
        
        Returns:
            List of timeline events
        """
        timeline = []
        
        for snapshot in self.snapshots:
            event = {
                'timestamp': snapshot.timestamp.isoformat(),
                'commit': snapshot.commit_hash[:8],
                'element_count': len(snapshot.elements),
                'relationship_count': len(snapshot.relationships)
            }
            timeline.append(event)
            
        return timeline
    
    def calculate_code_velocity(self, window_size: int = 10) -> List[Dict]:
        """
        Calculate the rate of change in the codebase.
        
        Args:
            window_size: Number of commits to include in velocity calculation
            
        Returns:
            List of velocity measurements
        """
        velocities = []
        
        for i in range(window_size, len(self.snapshots)):
            window_snapshots = self.snapshots[i-window_size:i]
            
            # Calculate changes
            elements_added = 0
            elements_modified = 0
            
            for j in range(1, len(window_snapshots)):
                prev = window_snapshots[j-1]
                curr = window_snapshots[j]
                
                prev_elements = set(prev.elements.keys())
                curr_elements = set(curr.elements.keys())
                
                elements_added += len(curr_elements - prev_elements)
                elements_modified += len(curr_elements & prev_elements)
            
            velocity = {
                'timestamp': self.snapshots[i].timestamp.isoformat(),
                'commit': self.snapshots[i].commit_hash[:8],
                'added': elements_added,
                'modified': elements_modified,
                'total_change': elements_added + elements_modified
            }
            velocities.append(velocity)
            
        return velocities
    
    def get_interaction_patterns(self) -> Dict[str, Dict]:
        """
        Analyze how code elements interact over time.
        
        Returns:
            Dictionary of interaction patterns
        """
        patterns = defaultdict(lambda: {
            'frequency': 0,
            'strength': 0.0,
            'last_interaction': None
        })
        
        for snapshot in self.snapshots:
            for from_elem, to_elem, rel_type in snapshot.relationships:
                key = f"{from_elem}:{to_elem}"
                patterns[key]['frequency'] += 1
                patterns[key]['last_interaction'] = snapshot.timestamp.isoformat()
                
        # Calculate interaction strength
        max_freq = max((p['frequency'] for p in patterns.values()), default=1)
        for key in patterns:
            patterns[key]['strength'] = patterns[key]['frequency'] / max_freq
            
        return dict(patterns)
