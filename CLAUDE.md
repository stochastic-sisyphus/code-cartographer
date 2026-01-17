# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install
pip install -e .              # Basic install
pip install -e ".[dev]"       # With dev dependencies

# Run analysis
python -m code_cartographer analyze -d /path/to/project -o output.json
python -m code_cartographer variants -d /path/to/project --semantic-threshold 0.8

# Apply variant merges (creates backups by default)
python -m code_cartographer variants -d /path/to/project --apply-merges
python -m code_cartographer variants -d /path/to/project --apply-merges --no-backup

# Code Warp House (Web Interface)
python -m code_cartographer serve                    # Start web server
python -m code_cartographer serve --port 8080        # Custom port
python -m code_cartographer serve --reload           # Auto-reload

# Tests
pytest                        # All tests with coverage
pytest tests/test_analyzer.py # Single file
pytest tests/test_git_analyzer.py     # Git analysis tests
pytest tests/api/test_server.py       # API tests

# Linting & Formatting
black src/ tests/
isort src/ tests/
mypy src/
flake8 src/ tests/
pre-commit run --all-files    # Run all hooks
```

## Architecture

```
code_cartographer/
├── cli.py                    # CLI (analyze, variants, serve commands)
├── core/
│   ├── analyzer.py           # ProjectAnalyzer, CodeInspector - AST-based analysis
│   ├── variant_analyzer.py   # VariantAnalyzer, CodeNormalizer, SemanticAnalyzer
│   ├── dependency_analyzer.py # DependencyAnalyzer - import/call graph
│   ├── variable_analyzer.py  # VariableAnalyzer - scope and usage tracking
│   ├── git_analyzer.py       # GitAnalyzer - git history extraction
│   ├── temporal_analyzer.py  # TemporalAnalyzer - multi-commit analysis
│   ├── reporter.py           # ReportGenerator - markdown output
│   └── visualizer.py         # CodeVisualizer - Graphviz output
├── api/                      # Code Warp House backend
│   ├── server.py             # FastAPI application
│   ├── routes/
│   │   ├── analysis.py       # Project analysis endpoints
│   │   └── temporal.py       # Git history endpoints
│   ├── models/
│   │   └── schemas.py        # Pydantic models
│   └── services/
│       └── websocket.py      # WebSocket connection manager
└── web/                      # Code Warp House frontend
    ├── static/
    │   ├── app.js            # Frontend JavaScript
    │   └── style.css         # UI styles
    └── templates/
        └── index.html        # Main HTML template
```

### Core Classes

#### Static Analysis
- **ProjectAnalyzer**: Entry point for full codebase analysis. Uses `CodeInspector` (AST visitor) to extract definitions, calls, complexity metrics.
- **VariantAnalyzer**: Detects similar code blocks using normalization + semantic embeddings (sentence-transformers). Can auto-merge variants with backup.
- **CodeNormalizer**: Strips docstrings, standardizes variable names, normalizes literals for comparison.
- **DependencyAnalyzer**: Builds import graph and call graph across files.

#### Temporal Analysis (Code Warp House)
- **GitAnalyzer**: Extracts git commit history, detects refactoring events (rename, move, split, merge, extract), tracks file changes and contributors.
- **TemporalAnalyzer**: Orchestrates multi-commit analysis with caching. Samples commits (uniform/major/all), detects complexity trends, tracks variant lifecycle.

#### Web API
- **FastAPI Server**: REST API with WebSocket support for real-time analysis updates.
- **ConnectionManager**: Manages WebSocket connections for progress streaming.

### Analysis Flow

1. Walk Python files (respecting exclude patterns)
2. Parse AST, extract definitions with `CodeInspector`
3. Compute SHA-256 hashes of function/class bodies
4. Optional: radon for cyclomatic complexity / maintainability index
5. Variant detection via normalized code + cosine similarity on embeddings
6. Output JSON or Markdown report

## Key Dependencies

### Static Analysis
- **sentence-transformers**: Semantic similarity for variant detection
- **nltk**: Tokenization for code analysis
- **radon** (optional): Cyclomatic complexity, maintainability index (CC > 10 or MI < 65 flags risk)
- **graphviz** (optional): Dependency graph visualization

### Code Warp House
- **gitpython**: Git repository interaction and history extraction
- **fastapi**: Modern web framework for REST API
- **uvicorn**: ASGI server for FastAPI
- **pydantic**: Data validation and serialization
- **websockets**: Real-time WebSocket communication
- **jinja2**: HTML templating

## Code Style

- Line length: 88 (black)
- Type hints required (mypy strict)
- Import sorting: isort with black profile

---

# Code Warp House - Extensive Technical Breakdown

## Overview

Code Warp House is the immersive temporal code visualization platform built on top of Code Cartographer. It adds:
- Git history analysis and temporal tracking
- FastAPI REST + WebSocket backend
- Modern web interface for interactive exploration
- Real-time analysis progress updates
- Complexity evolution tracking
- Refactoring event detection

**Status**: ✅ Complete and Operational (v0.3.0)
**Lines of Code**: ~3,000 new lines
**Test Coverage**: 38% overall, 72-81% on new modules

---

## Component Deep Dive

### 1. Git Analysis Layer (`core/git_analyzer.py`)

**Purpose**: Extract and analyze git repository history

**Key Classes**:

```python
class GitAnalyzer:
    """Main git history analyzer"""

    def __init__(self, repo_path: Path)
        # Initializes GitPython Repo, validates repository

    def get_commit_history(max_commits=100, branch="HEAD", file_pattern=None) -> List[CommitSnapshot]
        # Extracts commit metadata (hash, author, timestamp, files changed)
        # Handles renamed files, calculates insertions/deletions

    def detect_refactoring_events(max_commits=50) -> List[RefactoringEvent]
        # Pattern matching on commit messages for refactoring keywords
        # Detects file renames from diff metadata
        # Returns events with confidence scores (0.7-1.0)

    def build_temporal_complexity_graph(commits) -> TemporalMetrics
        # Aggregates file churn, hotspots, contributor stats
        # Extracts complexity timeline from analysis_data

    def get_file_history(file_path, max_commits=50) -> List[CommitSnapshot]
        # File-specific commit history
```

**Data Structures**:

```python
@dataclass
class CommitSnapshot:
    hash: str                    # Full SHA
    short_hash: str              # First 7 chars
    timestamp: datetime          # Commit time
    author: str                  # Author name
    author_email: str            # Author email
    message: str                 # Commit message
    files_changed: List[str]     # Modified files (includes "old -> new" for renames)
    insertions: int              # Lines added
    deletions: int               # Lines removed
    analysis_data: Optional[Dict] # ProjectAnalyzer output at this commit
```

**Refactoring Detection Logic**:
- **Keyword matching**: "rename", "move", "split", "merge", "extract", "refactor"
- **File rename detection**: Parses GitPython diff objects for `renamed_file=True`
- **Confidence scoring**: 1.0 for file renames (certain), 0.7 for keyword-based (heuristic)

**Performance Considerations**:
- Use `max_commits` to limit history depth
- File pattern filtering reduces unnecessary processing
- Caching prevents re-analysis of same commits

### 2. Temporal Analysis Layer (`core/temporal_analyzer.py`)

**Purpose**: Orchestrate multi-commit analysis with intelligent sampling

**Key Classes**:

```python
class TemporalAnalyzer:
    """Multi-commit analysis orchestrator"""

    def __init__(self, repo_path, git_analyzer=None, project_analyzer=None, cache_dir=None)
        # Sets up analyzers and caching directory
        # Default cache: .code-cartographer/cache/

    def analyze_evolution(commit_range=None, max_commits=100, sample_strategy="uniform") -> TemporalData
        # Main entry point for temporal analysis
        # Samples commits, runs ProjectAnalyzer at each
        # Detects trends, builds metrics

    def detect_complexity_trends(commits) -> List[ComplexityTrend]
        # Tracks file-level complexity over time
        # Determines trend direction (increasing/decreasing/stable)
        # Calculates max, min, current complexity

    def _sample_commits(commits, max_commits, strategy) -> List[CommitSnapshot]
        # Uniform: Even distribution across timeline
        # Major: Prioritizes high-change commits + refactoring keywords
        # All: Every commit (may be slow)

    def _analyze_commit_cached(commit_hash) -> Optional[Dict]
        # Checks disk cache first (.code-cartographer/cache/{hash}.json)
        # Runs ProjectAnalyzer if cache miss
        # Stores result to cache
```

**Sampling Strategies**:

1. **Uniform** (default):
   - Evenly spaced commits across timeline
   - Best for large repos, gives overview
   - Example: 100 commits → sample every 10th

2. **Major**:
   - Scores commits by: insertions + deletions
   - 2x boost for refactoring keywords
   - Sorts by score, takes top N
   - Best for finding significant changes

3. **All**:
   - No sampling, analyze every commit
   - Use only for small repos or specific ranges
   - Can be very slow (10+ minutes for large repos)

**Caching Architecture**:
```
.code-cartographer/
└── cache/
    ├── {commit_hash_1}.json  # Full ProjectAnalyzer output
    ├── {commit_hash_2}.json
    └── ...
```

**Complexity Trend Detection**:
- Groups timeline points by file
- Requires ≥2 commits for trend
- Calculates first-half vs second-half average
- Thresholds: >20% increase → "increasing", >20% decrease → "decreasing", else "stable"

### 3. FastAPI Backend (`api/`)

**Purpose**: REST + WebSocket API for web interface

#### Server Configuration (`api/server.py`)

```python
app = FastAPI(
    title="Code Warp House API",
    version="0.3.0",
    lifespan=lifespan  # Async context manager for startup/shutdown
)

# CORS enabled for frontend access
# Static files mounted at /static
# Templates at /templates
```

**Lifecycle Management**:
- `lifespan()` context manager logs startup/shutdown
- Future: Could add connection pooling, cache warmup, etc.

#### API Routes

**Analysis Routes** (`api/routes/analysis.py`):

```python
POST /api/v1/projects/analyze
    Request: ProjectAnalysisRequest (project_path, exclude_patterns, max_commits, include_temporal)
    Response: { project_id, status, analysis }
    Logic:
        1. Validate path exists and is directory
        2. Generate project_id = SHA256(path)[:16]
        3. Check disk cache
        4. Run ProjectAnalyzer
        5. Cache result (background task)
        6. Return analysis data

GET /api/v1/projects
    Returns: List of all cached projects
    Scans: .code-cartographer/cache/api/*.json

GET /api/v1/projects/{project_id}
    Returns: Full analysis for project
    Checks: Memory cache → disk cache → 404

GET /api/v1/projects/{project_id}/files
    Returns: List of files with metrics
    Extracts: analysis_result.files

GET /api/v1/projects/{project_id}/dependencies
    Logic:
        1. Load project path from cache
        2. Run DependencyAnalyzer
        3. Convert to graph format (nodes, edges)
        4. Return with metrics

GET /api/v1/projects/{project_id}/variants?threshold=0.8
    Logic:
        1. Load project path
        2. Run VariantAnalyzer with threshold
        3. Return variant groups

DELETE /api/v1/projects/{project_id}
    Removes: Memory cache + disk cache file
```

**Temporal Routes** (`api/routes/temporal.py`):

```python
GET /api/v1/projects/{project_id}/timeline?max_commits=100&file_pattern=.py
    Logic:
        1. Get project path from cache
        2. Create/reuse TemporalAnalyzer
        3. Call git_analyzer.get_commit_history()
        4. Format and return commits

GET /api/v1/projects/{project_id}/commits/{commit_hash}
    Returns: Detailed commit info
    Searches: Timeline for matching hash (prefix match)

GET /api/v1/projects/{project_id}/commits/{commit_hash}/analysis
    Logic:
        1. Call temporal_analyzer._analyze_commit_cached()
        2. Returns full ProjectAnalyzer output at that commit

GET /api/v1/projects/{project_id}/evolution/complexity?max_commits=50&strategy=uniform
    Logic:
        1. Run temporal_analyzer.analyze_evolution()
        2. Extract complexity_trends
        3. Format timeline points as {timestamp, complexity}

GET /api/v1/projects/{project_id}/evolution/refactorings
    Calls: git_analyzer.detect_refactoring_events()
    Returns: List of RefactoringEvent objects

GET /api/v1/projects/{project_id}/evolution/hotspots
    Logic:
        1. Get commit history
        2. Build temporal metrics
        3. Return file_churn, hotspots, contributor_stats

GET /api/v1/projects/{project_id}/files/{file_path}/history
    Calls: git_analyzer.get_file_history()
    Returns: Commits that modified specific file
```

**WebSocket** (`api/services/websocket.py`):

```python
class ConnectionManager:
    active_connections: Dict[str, Set[WebSocket]]

    async def connect(websocket, project_id)
        # Accept connection, send welcome message

    async def send_progress(project_id, progress, total, message)
        # Broadcast progress update to all clients

    async def send_complete(project_id, data)
        # Notify completion

    async def send_error(project_id, error)
        # Broadcast error

WS /api/v1/ws/analysis/{project_id}
    Protocol:
        Client connects → Receives {"type": "connected"}
        Server sends → {"type": "progress", "percentage": 50, "message": "..."}
        On complete → {"type": "complete", "data": {...}}
        On error → {"type": "error", "error": "..."}
```

**Caching Strategy**:
- **Memory**: `_project_cache` dict (project_id → analysis_data)
- **Disk**: `.code-cartographer/cache/api/{project_id}.json`
- **Temporal**: `.code-cartographer/cache/{commit_hash}.json`
- **Invalidation**: Manual (DELETE endpoint) or re-analyze

### 4. Web Frontend (`web/`)

**Architecture**: Vanilla JavaScript SPA (no build tools)

#### HTML Structure (`templates/index.html`)

```html
<body>
    <div class="app-container">
        <div class="sidebar">
            <!-- Project list -->
        </div>
        <div class="main-content">
            <!-- Project view (tabs) -->
        </div>
    </div>

    <!-- Dialogs -->
    <div id="analyzeDialog">...</div>
    <div id="progressOverlay">...</div>
    <div id="toast">...</div>
</body>
```

#### JavaScript Architecture (`static/app.js`)

```javascript
class CodeWarpHouse {
    constructor()
        this.apiBase = '/api/v1'
        this.currentProject = null
        this.websocket = null

    async init()
        this.setupEventListeners()
        await this.loadProjects()

    // Core Methods
    async loadProjects()
        // Fetch GET /api/v1/projects
        // Render project cards

    async handleAnalyze(event)
        // POST /api/v1/projects/analyze
        // Show progress overlay
        // Handle WebSocket updates (future)
        // Redirect to project view

    async loadProject(projectId)
        // Parallel fetch: project data + files
        // Try to load timeline (catch if not git repo)
        // Render tabs

    // Tab Rendering
    renderProjectView(project, files)
        // Header with stats
        // Tab navigation
        // Tab content containers

    renderFilesTab(files)
        // List of files with metrics

    renderTimeline(data)
        // Commit list with metadata

    renderDependencyGraph(data)
        // Placeholder + data dump (TODO: D3 visualization)

    renderComplexityChart(data)
        // Trends with inline bar charts

    // Async Data Loading
    async loadDependencies(projectId)
        // GET /api/v1/projects/{id}/dependencies

    async loadComplexityEvolution(projectId)
        // GET /api/v1/projects/{id}/evolution/complexity

    // UI Helpers
    showTab(tabName)
    showProgress(message)
    showError(message)
    showSuccess(message)
}
```

#### CSS Architecture (`static/style.css`)

**Design System**:
```css
:root {
    --bg-primary: #0a0a0a;      /* Deep black background */
    --bg-secondary: #1a1a1a;    /* Card backgrounds */
    --bg-tertiary: #2a2a2a;     /* Hover states */
    --text-primary: #e0e0e0;    /* Main text */
    --text-secondary: #a0a0a0;  /* Secondary text */
    --accent-primary: #667eea;  /* Purple gradient start */
    --accent-secondary: #764ba2; /* Purple gradient end */
    --success: #10b981;
    --error: #ef4444;
    --border: #333;
}
```

**Layout**:
- Grid-based app container (sidebar + main)
- Flexbox for header, stats, tabs
- Responsive breakpoints at 768px
- Mobile: Stacked layout

**Components**:
- `.project-card`: Hover effects, transform
- `.tab`: Active state with border-bottom
- `.timeline-point`: Height-based bar chart
- `.toast`: Animated slide-in notifications
- `.dialog-overlay`: Modal with backdrop

---

## API Reference Quick Guide

### Authentication
**Current**: None (local development)
**Future**: Consider API keys, OAuth for deployment

### Response Format
All endpoints return JSON. Errors use:
```json
{
    "detail": "Error message",
    "status_code": 404
}
```

### Common Parameters
- `max_commits`: Limit history depth (default: 50-100)
- `strategy`: Commit sampling ("uniform", "major", "all")
- `threshold`: Similarity threshold for variants (0.0-1.0)
- `exclude_patterns`: List of glob patterns to ignore

### Rate Limiting
**Current**: None
**Future**: Implement for production deployment

---

## Testing Strategy

### Unit Tests
```bash
# Core analyzers
pytest tests/test_analyzer.py           # Static analysis
pytest tests/test_git_analyzer.py       # Git operations
pytest tests/test_temporal_analyzer.py  # Temporal analysis

# API
pytest tests/api/test_server.py         # Endpoint tests
```

### Integration Tests
```bash
# Full workflow
pytest tests/test_functionality.py      # End-to-end scenarios
```

### Test Data
- `examples/mini_repo/`: Small Python project for testing
- Mock git repos created in `tmp_path` fixtures
- Mocked GitPython objects for unit tests

### Coverage Goals
- **Critical paths**: >80% (git_analyzer, temporal_analyzer achieved 72-81%)
- **API routes**: >60% (currently 23-40%, needs improvement)
- **Overall**: >50% (currently 38%)

### Testing Challenges
1. **Git operations**: Require actual git repos or complex mocking
2. **WebSocket**: Need async test client setup
3. **File I/O**: Temporary directories, cleanup
4. **Sentence transformers**: Slow model loading (mock in tests)

---

## Development Workflow

### Adding a New Endpoint

1. **Define Pydantic schemas** (`api/models/schemas.py`):
```python
class NewFeatureRequest(BaseModel):
    param1: str
    param2: int = 10

class NewFeatureResponse(BaseModel):
    result: List[Dict[str, Any]]
```

2. **Create route handler**:
```python
# api/routes/analysis.py or temporal.py
@router.get("/projects/{project_id}/new-feature")
async def get_new_feature(project_id: str, param: int = Query(default=10)):
    try:
        # Load project
        project = await get_project(project_id)

        # Call core analyzer
        result = some_analyzer.analyze(...)

        # Format response
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

3. **Add to router** (in `server.py` if new router):
```python
app.include_router(new_router, prefix="/api/v1", tags=["new-feature"])
```

4. **Write tests**:
```python
# tests/api/test_new_feature.py
def test_new_feature_success(client):
    response = client.get("/api/v1/projects/test_id/new-feature")
    assert response.status_code == 200
    assert "result" in response.json()
```

5. **Update frontend** (`web/static/app.js`):
```javascript
async loadNewFeature(projectId) {
    const response = await fetch(`${this.apiBase}/projects/${projectId}/new-feature`);
    const data = await response.json();
    this.renderNewFeature(data);
}
```

### Adding a New Analyzer

1. **Create analyzer class** (`core/new_analyzer.py`):
```python
class NewAnalyzer:
    def __init__(self, root: Path):
        self.root = root

    def analyze(self) -> Dict[str, Any]:
        # Analysis logic
        return {"findings": [...]}
```

2. **Write tests** (`tests/test_new_analyzer.py`):
```python
def test_new_analyzer(tmp_path):
    # Setup test files
    test_file = tmp_path / "test.py"
    test_file.write_text("# test code")

    # Run analyzer
    analyzer = NewAnalyzer(tmp_path)
    result = analyzer.analyze()

    # Assert
    assert "findings" in result
```

3. **Integrate with API**:
```python
# api/routes/analysis.py
from code_cartographer.core.new_analyzer import NewAnalyzer

@router.get("/projects/{project_id}/new-analysis")
async def get_new_analysis(project_id: str):
    project = await get_project(project_id)
    analyzer = NewAnalyzer(Path(project["project_path"]))
    return analyzer.analyze()
```

### Frontend Development

**No Build Step!** Just edit and refresh:

1. Edit `web/static/app.js` or `web/static/style.css`
2. Refresh browser (Ctrl+R)
3. Changes appear immediately

**For auto-reload during development**:
```bash
python -m code_cartographer serve --reload
```

---

## Troubleshooting Guide

### Common Issues

#### 1. "Not a git repository" Error
**Symptom**: Temporal endpoints return 400 error
**Cause**: Project directory isn't a git repository
**Fix**:
```bash
cd /path/to/project
git init
git add .
git commit -m "Initial commit"
```

#### 2. Empty Timeline
**Symptom**: Timeline shows "No commit history available"
**Causes**:
- No commits in repository
- File pattern excludes all files
- Git branch has no commits
**Fix**: Verify `git log` shows commits, remove file_pattern filter

#### 3. Slow Analysis
**Symptom**: Analysis takes >5 minutes
**Causes**:
- Large repository (>10k files)
- Many commits (>1000)
- No caching
**Fix**:
- Reduce `max_commits` to 20-50
- Use `uniform` or `major` sampling
- Add exclude patterns for node_modules, build dirs
- Clear and rebuild cache

#### 4. WebSocket Connection Failed
**Symptom**: No real-time updates in UI
**Cause**: Browser WebSocket blocked or server issue
**Fix**:
- Check browser console for errors
- Verify server logs for WebSocket upgrade
- Disable browser extensions (ad blockers)

#### 5. ModuleNotFoundError
**Symptom**: Import errors for new dependencies
**Cause**: Dependencies not installed
**Fix**:
```bash
pip install -e .
# Or install specific package
pip install gitpython fastapi uvicorn
```

#### 6. Stale Cache
**Symptom**: Old analysis results shown
**Cause**: Cached data not invalidated
**Fix**:
```bash
# Clear all caches
rm -rf .code-cartographer/cache
# Or use API
curl -X DELETE http://localhost:8000/api/v1/projects/{project_id}
```

### Debugging Tips

**Enable debug logging**:
```python
# Add to server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check FastAPI logs**:
```bash
# Server output shows all requests
python -m code_cartographer serve
# Look for 200, 404, 500 status codes
```

**Browser DevTools**:
- Network tab: Check API responses
- Console tab: Check JavaScript errors
- Application tab: Clear localStorage if needed

**Test specific endpoint**:
```bash
# Use curl
curl -X POST http://localhost:8000/api/v1/projects/analyze \
  -H "Content-Type: application/json" \
  -d '{"project_path": "/path/to/project"}'
```

---

## Next Steps - Detailed Roadmap

### Phase 1: Visualization Enhancements (High Priority)

#### 1.1 D3.js Force-Directed Dependency Graph
**Goal**: Interactive, physics-based dependency visualization

**Implementation**:
1. Add D3.js to static assets:
```html
<!-- templates/index.html -->
<script src="https://d3js.org/d3.v7.min.js"></script>
```

2. Create `web/static/visualizations/dependency-graph.js`:
```javascript
class DependencyGraph {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`)
        this.width = 800
        this.height = 600
    }

    render(data) {
        // nodes: [{id, type, complexity}]
        // edges: [{source, target, type}]

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.edges).id(d => d.id))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(this.width/2, this.height/2))

        // Render SVG nodes and links
        // Add zoom, pan, drag behaviors
        // Color by module, size by complexity
    }
}
```

3. Integrate in app.js:
```javascript
renderDependencyGraph(data) {
    const graph = new DependencyGraph('dependencyGraph');
    graph.render(data);
}
```

**Features to Add**:
- Click node → highlight dependencies
- Hover → show metrics
- Filter by module
- Export as SVG

**Estimated Effort**: 8-12 hours
**Files to Create/Modify**:
- `web/static/visualizations/dependency-graph.js` (new, ~300 lines)
- `web/static/app.js` (modify renderDependencyGraph)
- `web/static/style.css` (add graph styles)

#### 1.2 Three.js 3D Complexity Terrain
**Goal**: Immersive 3D landscape where height = complexity

**Implementation**:
1. Add Three.js:
```html
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
```

2. Create `web/static/visualizations/complexity-terrain.js`:
```javascript
class ComplexityTerrain {
    constructor(containerId) {
        this.container = document.getElementById(containerId)
        this.scene = new THREE.Scene()
        this.camera = new THREE.PerspectiveCamera(75, width/height, 0.1, 1000)
        this.renderer = new THREE.WebGLRenderer()
    }

    buildTerrain(files) {
        // Create grid based on directory structure
        // Height = cyclomatic complexity
        // Color = gradient (green → yellow → red)
        // Add fog for distant files

        const geometry = new THREE.PlaneGeometry(width, height, xSegments, ySegments)
        // Modify vertices based on complexity

        const material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            wireframe: false
        })

        const terrain = new THREE.Mesh(geometry, material)
        this.scene.add(terrain)
    }

    animate() {
        // Rotation, lighting effects
        requestAnimationFrame(() => this.animate())
        this.renderer.render(this.scene, this.camera)
    }
}
```

3. Add controls:
```javascript
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
this.controls = new OrbitControls(this.camera, this.renderer.domElement)
```

**Features**:
- Orbit/zoom navigation
- Click on peak → show file details
- Animate terrain morphing over time
- Peaks pulse for recent changes

**Estimated Effort**: 12-16 hours
**Files to Create/Modify**:
- `web/static/visualizations/complexity-terrain.js` (new, ~400 lines)
- `web/templates/index.html` (add canvas element)
- `web/static/app.js` (integrate terrain rendering)

#### 1.3 Enhanced Timeline Scrubber
**Goal**: Playback through history with smooth transitions

**Implementation**:
```javascript
class TimelineScrubber {
    constructor(commits, onCommitChange) {
        this.commits = commits
        this.currentIndex = 0
        this.playing = false
        this.speed = 1000 // ms per commit
    }

    play() {
        this.playing = true
        this.interval = setInterval(() => {
            this.next()
        }, this.speed)
    }

    pause() {
        this.playing = false
        clearInterval(this.interval)
    }

    next() {
        if (this.currentIndex < this.commits.length - 1) {
            this.currentIndex++
            this.onCommitChange(this.commits[this.currentIndex])
        } else {
            this.pause()
        }
    }

    seekTo(index) {
        this.currentIndex = index
        this.onCommitChange(this.commits[index])
    }
}
```

**UI Elements**:
- Play/pause button
- Speed control (0.5x, 1x, 2x, 5x)
- Drag slider
- Keyboard shortcuts (space = play/pause, ← → = prev/next)

**Estimated Effort**: 4-6 hours

### Phase 2: Advanced Analytics (Medium Priority)

#### 2.1 Technical Debt Tracking
**Goal**: Quantify and track technical debt over time

**New Analyzer** (`core/debt_analyzer.py`):
```python
class DebtAnalyzer:
    def calculate_debt_score(self, file_data: Dict) -> float:
        """
        Debt score formula:
        - High complexity (CC > 10): +10 points
        - Low maintainability (MI < 65): +15 points
        - No docstrings: +5 points
        - Long functions (>50 lines): +8 points
        - Deep nesting (>4 levels): +12 points
        """
        score = 0

        if file_data.get('complexity', {}).get('cyclomatic', 0) > 10:
            score += 10

        if file_data.get('maintainability_index', 100) < 65:
            score += 15

        # ... more debt indicators

        return score

    def track_debt_over_time(self, temporal_data: TemporalData) -> List[DebtMetric]:
        """Calculate debt at each commit"""
        debt_timeline = []
        for commit in temporal_data.commit_snapshots:
            total_debt = sum(
                self.calculate_debt_score(f)
                for f in commit.analysis_data.get('files', [])
            )
            debt_timeline.append(DebtMetric(
                timestamp=commit.timestamp,
                total_debt=total_debt,
                high_debt_files=[...],
                debt_delta=total_debt - previous_debt
            ))
        return debt_timeline
```

**API Endpoint**:
```python
GET /api/v1/projects/{id}/debt/timeline
GET /api/v1/projects/{id}/debt/hotspots  # Files with highest debt
```

**Frontend**:
- Debt trend chart
- Debt heatmap (files colored by debt level)
- Debt reduction recommendations

**Estimated Effort**: 8-10 hours

#### 2.2 Code Quality Trends
**Goal**: Track quality metrics beyond complexity

**Metrics to Track**:
- Test coverage (if pytest-cov data available)
- Type hint coverage (mypy)
- Documentation coverage (docstrings)
- Code duplication percentage
- Average function length
- Import complexity (circular dependencies)

**Implementation**:
```python
# core/quality_analyzer.py
class QualityAnalyzer:
    def analyze_type_hints(self, file_ast) -> float:
        # Percentage of functions with type hints

    def analyze_docstrings(self, file_ast) -> float:
        # Percentage of definitions with docstrings

    def detect_circular_imports(self, dependency_graph) -> List[Cycle]:
        # Use NetworkX cycle detection
```

**Estimated Effort**: 10-12 hours

#### 2.3 Contributor Analysis
**Goal**: Team productivity and ownership insights

**Features**:
- Code ownership map (files by primary author)
- Contributor activity timeline
- Bus factor analysis (files with single contributor)
- Collaboration graph (who modifies whose code)

**API Endpoint**:
```python
GET /api/v1/projects/{id}/contributors/stats
GET /api/v1/projects/{id}/contributors/ownership
GET /api/v1/projects/{id}/contributors/collaboration
```

**Estimated Effort**: 6-8 hours

### Phase 3: Collaboration Features (Medium Priority)

#### 3.1 Export & Share
**Goal**: Share analysis results with team

**Implementation**:
```python
# api/routes/export.py
@router.get("/projects/{project_id}/export/pdf")
async def export_pdf(project_id: str):
    # Use reportlab or weasyprint
    # Generate PDF with charts, metrics
    # Return as downloadable file

@router.get("/projects/{project_id}/export/html")
async def export_html(project_id: str):
    # Static HTML with embedded charts
    # No server needed to view
```

**Share Links**:
```python
@router.post("/projects/{project_id}/share")
async def create_share_link(project_id: str, expiry_days: int = 7):
    # Generate shareable UUID
    # Store in share_links table with expiry
    # Return public URL

@router.get("/share/{share_id}")
async def view_shared_analysis(share_id: str):
    # Read-only view of analysis
```

**Estimated Effort**: 8-10 hours

#### 3.2 Team Dashboard
**Goal**: Aggregate view of multiple projects

**Features**:
- Multi-project overview
- Complexity trends across projects
- Debt comparison
- CI/CD integration (analyze on commit)

**Estimated Effort**: 12-16 hours

### Phase 4: Performance & Scalability (Low Priority - Future)

#### 4.1 Async Analysis
**Goal**: Non-blocking analysis with background workers

**Implementation**:
```python
# Use Celery or similar
from celery import Celery

app = Celery('code_warp_house', broker='redis://localhost:6379')

@app.task
def analyze_project_async(project_path):
    analyzer = ProjectAnalyzer(Path(project_path))
    result = analyzer.execute()
    return result

# API endpoint
@router.post("/projects/analyze-async")
async def analyze_async(request: ProjectAnalysisRequest):
    task = analyze_project_async.delay(request.project_path)
    return {"task_id": task.id, "status": "pending"}

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = AsyncResult(task_id)
    return {"status": task.status, "result": task.result}
```

**Estimated Effort**: 12-16 hours
**Requires**: Redis, Celery

#### 4.2 Database Backend
**Goal**: Replace file caching with proper database

**Options**:
- **PostgreSQL**: Full-featured, great for complex queries
- **SQLite**: Lightweight, no external dependencies
- **MongoDB**: Document store, fits JSON data well

**Schema Design** (PostgreSQL example):
```sql
CREATE TABLE projects (
    id VARCHAR(16) PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_analyzed TIMESTAMP
);

CREATE TABLE analyses (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(16) REFERENCES projects(id),
    commit_hash VARCHAR(40),
    timestamp TIMESTAMP,
    data JSONB,  -- Full analysis result
    INDEX idx_project_commit (project_id, commit_hash)
);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES analyses(id),
    file_path TEXT,
    complexity INTEGER,
    maintainability_index FLOAT,
    lines_of_code INTEGER
);
```

**Migration Path**:
1. Add database dependency (SQLAlchemy)
2. Create models
3. Implement repository pattern
4. Migrate existing file cache
5. Update API to use database

**Estimated Effort**: 16-24 hours

#### 4.3 Incremental Analysis
**Goal**: Only re-analyze changed files

**Implementation**:
```python
class IncrementalAnalyzer:
    def __init__(self, project_path, previous_analysis):
        self.project_path = project_path
        self.previous = previous_analysis

    def analyze_incremental(self, commit_hash):
        # Get changed files from git diff
        changed_files = self.get_changed_files(commit_hash)

        # Copy previous analysis
        new_analysis = copy.deepcopy(self.previous)

        # Re-analyze only changed files
        for file in changed_files:
            file_analysis = self.analyze_single_file(file)
            self.update_analysis(new_analysis, file, file_analysis)

        return new_analysis
```

**Estimated Effort**: 8-12 hours

### Phase 5: Plugin Architecture (Low Priority - Future)

#### 5.1 Plugin System
**Goal**: Allow custom analyzers without modifying core

**Design**:
```python
# core/plugin_system.py
class AnalyzerPlugin(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def analyze(self, file_path: Path) -> Dict:
        pass

class PluginManager:
    def __init__(self):
        self.plugins = []

    def register(self, plugin: AnalyzerPlugin):
        self.plugins.append(plugin)

    def run_all(self, file_path: Path) -> Dict[str, Any]:
        results = {}
        for plugin in self.plugins:
            results[plugin.name()] = plugin.analyze(file_path)
        return results

# Example plugin
class SecurityAnalyzerPlugin(AnalyzerPlugin):
    def name(self) -> str:
        return "security"

    def analyze(self, file_path: Path) -> Dict:
        # Run bandit or similar
        return {"vulnerabilities": [...]}
```

**Plugin Discovery**:
```python
# Discover plugins via entry points
# setup.py / pyproject.toml
[project.entry-points."code_cartographer.plugins"]
security = "my_plugin:SecurityAnalyzerPlugin"
```

**Estimated Effort**: 10-14 hours

#### 5.2 Custom Visualizations
**Goal**: Allow users to create custom D3/Three.js visualizations

**Implementation**:
```javascript
// Plugin API
class VisualizationPlugin {
    constructor(name, renderFunction) {
        this.name = name;
        this.render = renderFunction;
    }
}

// Registry
class PluginRegistry {
    constructor() {
        this.visualizations = new Map();
    }

    register(plugin) {
        this.visualizations.set(plugin.name, plugin);
    }

    get(name) {
        return this.visualizations.get(name);
    }
}

// Usage
const customViz = new VisualizationPlugin('my-viz', (data, container) => {
    // Custom D3/Three.js code
});

registry.register(customViz);
```

**Estimated Effort**: 6-8 hours

---

## Extension Points & Customization

### Adding Custom Metrics

**Step 1**: Extend CodeInspector:
```python
# core/analyzer.py
class CodeInspector(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        # Existing logic...

        # Add custom metric
        custom_metric = self.calculate_custom_metric(node)
        definition_data['custom_metric'] = custom_metric
```

**Step 2**: Update schemas:
```python
# api/models/schemas.py
class ComplexityMetrics(BaseModel):
    cyclomatic: Optional[int] = None
    custom_metric: Optional[float] = None  # New field
```

**Step 3**: Display in UI:
```javascript
// web/static/app.js
<span>Custom: ${file.metrics.custom_metric}</span>
```

### Custom Sampling Strategy

```python
# core/temporal_analyzer.py
def _sample_commits(self, commits, max_commits, strategy):
    if strategy == "custom":
        return self._sample_custom(commits, max_commits)
    # ... existing strategies

def _sample_custom(self, commits, max_commits):
    # Your custom logic
    # Example: Only commits on main branch
    main_commits = [c for c in commits if c.branch == "main"]
    return main_commits[:max_commits]
```

### Custom Refactoring Patterns

```python
# core/git_analyzer.py
def detect_refactoring_events(self, max_commits=50):
    # ... existing logic

    # Add custom pattern
    for commit in commits:
        if self._is_performance_optimization(commit):
            event = RefactoringEvent(
                commit_hash=commit.hash,
                event_type="performance",
                # ...
            )
            events.append(event)

def _is_performance_optimization(self, commit):
    keywords = ['optimize', 'performance', 'faster', 'cache']
    return any(kw in commit.message.lower() for kw in keywords)
```

---

## Production Deployment Considerations

### Security

1. **Add Authentication**:
```python
# api/auth.py
from fastapi.security import HTTPBearer
from jose import jwt

security = HTTPBearer()

async def verify_token(token: str):
    # Verify JWT token
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return payload

# Protect endpoints
@router.get("/projects")
async def list_projects(token: str = Depends(security)):
    await verify_token(token)
    # ... endpoint logic
```

2. **Rate Limiting**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/projects/analyze")
@limiter.limit("5/minute")
async def analyze_project(...):
    # ... logic
```

3. **Input Validation**:
```python
# Prevent path traversal
def validate_project_path(path: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ALLOWED_BASE_DIR):
        raise ValueError("Invalid path")
    return resolved
```

### Scaling

1. **Load Balancer**: Nginx in front of multiple Uvicorn instances
2. **Caching**: Redis for shared cache across instances
3. **Database**: PostgreSQL with connection pooling
4. **Background Jobs**: Celery with Redis/RabbitMQ
5. **Static Files**: CDN for web assets

### Monitoring

1. **Logging**:
```python
import structlog

logger = structlog.get_logger()
logger.info("analysis_started", project_id=project_id, user_id=user_id)
```

2. **Metrics**: Prometheus + Grafana
3. **Error Tracking**: Sentry
4. **APM**: New Relic or DataDog

---

## Contributing Guidelines

### Code Review Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Type hints present
- [ ] Black formatting applied
- [ ] No security vulnerabilities
- [ ] Error handling comprehensive
- [ ] Logging added for key operations
- [ ] API changes backward compatible
- [ ] Performance impact considered

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
```

---

## Resources

### Learning Materials
- **FastAPI**: https://fastapi.tiangolo.com/
- **D3.js**: https://d3js.org/
- **Three.js**: https://threejs.org/
- **GitPython**: https://gitpython.readthedocs.io/

### Similar Projects
- **Code Climate**: https://codeclimate.com/
- **SonarQube**: https://www.sonarqube.org/
- **Sourcegraph**: https://sourcegraph.com/

### Community
- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Contributing.md**: Detailed contribution guidelines
