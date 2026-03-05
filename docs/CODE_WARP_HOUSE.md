# Code Warp House - User Guide

## Overview

Code Warp House transforms Code Cartographer from a static analysis tool into an immersive, temporal code visualization platform. It enables you to explore your codebase's evolution through git history, visualize dependencies, track complexity trends, and detect refactoring patterns.

## Quick Start

### 1. Start the Server

```bash
# Using the CLI
python -m code_cartographer serve

# With custom host/port
python -m code_cartographer serve --host 0.0.0.0 --port 8080

# With auto-reload for development
python -m code_cartographer serve --reload
```

### 2. Access the Web Interface

Open your browser to `http://localhost:8000`

### 3. Analyze a Project

1. Click "Analyze Project" in the sidebar
2. Enter the path to your project (e.g., `/Users/you/code/myproject`)
3. Click "Analyze" and wait for the analysis to complete
4. Explore your codebase through the interactive interface

## Features

### 🕒 Temporal Analysis

Navigate through your codebase's history:

- **Timeline View**: Browse commits chronologically
- **Complexity Evolution**: Track how code complexity changes over time
- **Refactoring Events**: Automatically detect renames, splits, merges, and extractions
- **File Hotspots**: Identify frequently changed files that may need attention

### 📊 Static Analysis

Comprehensive code analysis:

- **Project Metrics**: Total files, lines of code, definitions
- **File Analysis**: Cyclomatic complexity, maintainability index
- **Definition Tracking**: Functions, classes, methods with complexity scores

### 🔗 Dependency Visualization

Understand code relationships:

- **Import Graph**: See which modules depend on each other
- **Call Graph**: Trace function and method calls
- **Dependency Metrics**: Coupling and cohesion analysis

### 🔍 Code Variants

Find and manage code duplication:

- **Similarity Detection**: Semantic analysis using sentence transformers
- **Variant Groups**: Grouped by similarity threshold
- **Merge Suggestions**: Refactoring opportunities

## API Reference

### REST Endpoints

#### Health Check
```
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "service": "code-warp-house",
  "version": "0.3.0"
}
```

#### Analyze Project
```
POST /api/v1/projects/analyze
```

Request:
```json
{
  "project_path": "/path/to/project",
  "exclude_patterns": ["**/node_modules/**", "**/__pycache__/**"],
  "max_commits": 100,
  "include_temporal": true
}
```

Response:
```json
{
  "project_id": "abc123def456",
  "status": "complete",
  "analysis": {
    "project_path": "/path/to/project",
    "analysis_timestamp": "2024-01-15T10:30:00",
    "analysis_result": { ... }
  }
}
```

#### List Projects
```
GET /api/v1/projects
```

#### Get Project Details
```
GET /api/v1/projects/{project_id}
```

#### Get Timeline
```
GET /api/v1/projects/{project_id}/timeline?max_commits=50
```

#### Get Complexity Evolution
```
GET /api/v1/projects/{project_id}/evolution/complexity?max_commits=20&strategy=uniform
```

Parameters:
- `max_commits`: Number of commits to analyze (default: 50)
- `strategy`: Sampling strategy - `uniform`, `major`, or `all`

#### Get Refactoring Events
```
GET /api/v1/projects/{project_id}/evolution/refactorings
```

#### Get Hotspots
```
GET /api/v1/projects/{project_id}/evolution/hotspots
```

#### Get File History
```
GET /api/v1/projects/{project_id}/files/{file_path}/history
```

### WebSocket

Real-time updates during analysis:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/analysis/project_id');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'connected':
      console.log('Connected to analysis stream');
      break;
    case 'progress':
      console.log(`Progress: ${data.percentage}% - ${data.message}`);
      break;
    case 'complete':
      console.log('Analysis complete:', data.data);
      break;
    case 'error':
      console.error('Analysis error:', data.error);
      break;
  }
};
```

## Architecture

### Backend (Python + FastAPI)

```
code_cartographer/
├── api/
│   ├── server.py              # FastAPI application
│   ├── routes/
│   │   ├── analysis.py        # Project analysis endpoints
│   │   └── temporal.py        # Git history endpoints
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   └── services/
│       └── websocket.py       # WebSocket connection manager
├── core/
│   ├── analyzer.py            # Static code analysis
│   ├── git_analyzer.py        # Git history extraction
│   └── temporal_analyzer.py   # Multi-commit analysis
└── web/
    ├── static/
    │   ├── app.js             # Frontend JavaScript
    │   └── style.css          # UI styles
    └── templates/
        └── index.html         # Main HTML template
```

### Frontend (Vanilla JavaScript)

The web interface uses vanilla JavaScript for simplicity and performance:

- **No build step required**: Works directly in the browser
- **Responsive design**: Mobile-friendly layout
- **Real-time updates**: WebSocket integration for progress
- **Interactive visualizations**: Timeline, complexity charts, dependency graphs

## Performance

### Caching

Analysis results are cached in `.code-cartographer/cache/` to improve performance:

- **Disk cache**: JSON files per commit hash
- **Memory cache**: Active projects kept in RAM
- **Smart invalidation**: Re-analyze only when needed

### Sampling Strategies

For large repositories, use commit sampling:

- **Uniform**: Evenly distributed commits across timeline
- **Major**: Prioritize commits with significant changes or refactoring keywords
- **All**: Analyze every commit (may be slow for large repos)

### Recommendations

- Start with `max_commits=20` for initial exploration
- Use `uniform` strategy for large repos
- Increase `max_commits` progressively as needed
- Exclude build artifacts and dependencies with `exclude_patterns`

## Examples

### Python CLI

```python
from pathlib import Path
from code_cartographer.core.git_analyzer import GitAnalyzer
from code_cartographer.core.temporal_analyzer import TemporalAnalyzer

# Analyze git history
git_analyzer = GitAnalyzer(Path("/path/to/repo"))
commits = git_analyzer.get_commit_history(max_commits=50)

# Detect refactoring events
events = git_analyzer.detect_refactoring_events()

# Temporal analysis
temporal = TemporalAnalyzer(Path("/path/to/repo"))
data = temporal.analyze_evolution(max_commits=20, sample_strategy="uniform")

# Access complexity trends
for trend in data.complexity_trends:
    print(f"{trend.file_path}: {trend.trend_direction}")
    print(f"  Current: {trend.current_complexity}")
    print(f"  Max: {trend.max_complexity}")
```

### JavaScript API Client

```javascript
class CodeWarpAPI {
    constructor(baseURL = '/api/v1') {
        this.baseURL = baseURL;
    }

    async analyzeProject(projectPath) {
        const response = await fetch(`${this.baseURL}/projects/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_path: projectPath })
        });
        return response.json();
    }

    async getTimeline(projectId, maxCommits = 50) {
        const response = await fetch(
            `${this.baseURL}/projects/${projectId}/timeline?max_commits=${maxCommits}`
        );
        return response.json();
    }

    async getComplexityEvolution(projectId) {
        const response = await fetch(
            `${this.baseURL}/projects/${projectId}/evolution/complexity`
        );
        return response.json();
    }
}

// Usage
const api = new CodeWarpAPI();
const result = await api.analyzeProject('/path/to/project');
const timeline = await api.getTimeline(result.project_id);
```

## Troubleshooting

### "Not a git repository" Error

The temporal analysis features require a git repository. Ensure:
- The project is initialized with `git init`
- There are committed changes (`git log` shows commits)
- You're analyzing the repository root (contains `.git/`)

### Slow Analysis

For large repositories:
1. Reduce `max_commits` to 20-50
2. Use `uniform` or `major` sampling strategy
3. Add exclude patterns for build artifacts
4. Check disk space for cache directory

### WebSocket Connection Failed

Ensure:
- Server is running on the correct port
- No firewall blocking WebSocket connections
- Browser supports WebSockets (all modern browsers do)

### Empty Timeline

If timeline shows no commits:
- Verify it's a git repository
- Check `git log` shows commits
- Ensure file pattern filter isn't excluding all files

## Next Steps

### Planned Features

1. **3D Visualizations**
   - Three.js-based complexity terrain
   - Interactive dependency constellation
   - Immersive code exploration

2. **Advanced Analytics**
   - Technical debt tracking
   - Code quality trends
   - Contributor analysis

3. **Collaboration Features**
   - Share analysis results
   - Team dashboards
   - Export reports

### Contributing

To extend Code Warp House:

1. **Add new visualizations**: Create components in `web/static/`
2. **New API endpoints**: Add routes in `api/routes/`
3. **Analysis features**: Extend core analyzers in `core/`
4. **Tests**: Add tests in `tests/`

See `CONTRIBUTING.md` for detailed guidelines.

## Support

- **Documentation**: [GitHub Wiki](https://github.com/stochastic-sisyphus/code-cartographer/wiki)
- **Issues**: [GitHub Issues](https://github.com/stochastic-sisyphus/code-cartographer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/stochastic-sisyphus/code-cartographer/discussions)

## License

MIT License - See LICENSE file for details
