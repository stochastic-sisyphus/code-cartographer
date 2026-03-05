/**
 * Code Warp House - Frontend Application
 * Interactive web interface for temporal code visualization
 */

class CodeWarpHouse {
    constructor() {
        this.apiBase = '/api/v1';
        this.currentProject = null;
        this.websocket = null;
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadProjects();
    }

    setupEventListeners() {
        document.getElementById('analyzeBtn')?.addEventListener('click', () => this.showAnalyzeDialog());
        document.getElementById('analyzeForm')?.addEventListener('submit', (e) => this.handleAnalyze(e));
        document.getElementById('closeDialog')?.addEventListener('click', () => this.hideAnalyzeDialog());
    }

    showAnalyzeDialog() {
        document.getElementById('analyzeDialog').classList.add('active');
    }

    hideAnalyzeDialog() {
        document.getElementById('analyzeDialog').classList.remove('active');
    }

    async loadProjects() {
        try {
            const response = await fetch(`${this.apiBase}/projects`);
            const data = await response.json();
            this.renderProjects(data.projects);
        } catch (error) {
            console.error('Failed to load projects:', error);
            this.showError('Failed to load projects');
        }
    }

    renderProjects(projects) {
        const container = document.getElementById('projectsList');
        if (!projects || projects.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No projects analyzed yet</p>
                    <button onclick="app.showAnalyzeDialog()" class="btn-primary">
                        Analyze Your First Project
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = projects.map(project => `
            <div class="project-card" onclick="app.loadProject('${project.project_id}')">
                <h3>${this.getProjectName(project.project_path)}</h3>
                <p class="path">${project.project_path}</p>
                <p class="timestamp">Analyzed: ${new Date(project.analysis_timestamp).toLocaleString()}</p>
            </div>
        `).join('');
    }

    getProjectName(path) {
        return path.split('/').filter(Boolean).pop() || 'Unknown Project';
    }

    async handleAnalyze(event) {
        event.preventDefault();
        const formData = new FormData(event.target);
        const projectPath = formData.get('projectPath');

        this.showProgress('Analyzing project...');

        try {
            const response = await fetch(`${this.apiBase}/projects/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_path: projectPath,
                    include_temporal: true
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis failed');
            }

            const result = await response.json();
            this.hideAnalyzeDialog();
            this.hideProgress();
            await this.loadProjects();
            await this.loadProject(result.project_id);
            this.showSuccess('Analysis complete!');

        } catch (error) {
            this.hideProgress();
            this.showError(error.message);
        }
    }

    async loadProject(projectId) {
        this.currentProject = projectId;

        try {
            const [projectData, files] = await Promise.all([
                fetch(`${this.apiBase}/projects/${projectId}`).then(r => r.json()),
                fetch(`${this.apiBase}/projects/${projectId}/files`).then(r => r.json())
            ]);

            this.renderProjectView(projectData, files);

            // Load timeline if it's a git repository
            try {
                const timeline = await fetch(`${this.apiBase}/projects/${projectId}/timeline?max_commits=50`).then(r => r.json());
                this.renderTimeline(timeline);
            } catch (e) {
                console.log('No git history available');
            }

        } catch (error) {
            console.error('Failed to load project:', error);
            this.showError('Failed to load project details');
        }
    }

    renderProjectView(project, filesData) {
        const container = document.getElementById('projectView');
        const analysis = project.analysis_result || {};
        const files = filesData.files || [];

        container.innerHTML = `
            <div class="project-header">
                <h2>${this.getProjectName(project.project_path)}</h2>
                <p>${project.project_path}</p>
                <div class="stats">
                    <div class="stat">
                        <span class="value">${files.length}</span>
                        <span class="label">Files</span>
                    </div>
                    <div class="stat">
                        <span class="value">${this.getTotalLines(files)}</span>
                        <span class="label">Lines of Code</span>
                    </div>
                    <div class="stat">
                        <span class="value">${this.getTotalDefinitions(files)}</span>
                        <span class="label">Definitions</span>
                    </div>
                </div>
            </div>

            <div class="tabs">
                <button class="tab active" onclick="app.showTab('files')">Files</button>
                <button class="tab" onclick="app.showTab('dependencies')">Dependencies</button>
                <button class="tab" onclick="app.showTab('timeline')">Timeline</button>
                <button class="tab" onclick="app.showTab('complexity')">Complexity</button>
            </div>

            <div id="tabContent">
                <div id="filesTab" class="tab-content active">
                    ${this.renderFilesTab(files)}
                </div>
                <div id="dependenciesTab" class="tab-content">
                    <button onclick="app.loadDependencies('${this.currentProject}')" class="btn-primary">
                        Load Dependency Graph
                    </button>
                    <div id="dependencyGraph"></div>
                </div>
                <div id="timelineTab" class="tab-content">
                    <div id="timelineContent">Loading timeline...</div>
                </div>
                <div id="complexityTab" class="tab-content">
                    <button onclick="app.loadComplexityEvolution('${this.currentProject}')" class="btn-primary">
                        Load Complexity Evolution
                    </button>
                    <div id="complexityChart"></div>
                </div>
            </div>
        `;
    }

    renderFilesTab(files) {
        return `
            <div class="files-list">
                ${files.map(file => `
                    <div class="file-item">
                        <h4>${file.path}</h4>
                        <div class="file-stats">
                            <span>Lines: ${file.lines_of_code || 0}</span>
                            <span>Definitions: ${(file.definitions || []).length}</span>
                            ${file.metrics && file.metrics.cyclomatic ?
                                `<span>Complexity: ${file.metrics.cyclomatic}</span>` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    renderTimeline(data) {
        const container = document.getElementById('timelineContent');
        const commits = data.commits || [];

        if (commits.length === 0) {
            container.innerHTML = '<p>No commit history available</p>';
            return;
        }

        container.innerHTML = `
            <div class="timeline">
                ${commits.map(commit => `
                    <div class="commit-item">
                        <div class="commit-hash">${commit.short_hash}</div>
                        <div class="commit-info">
                            <div class="commit-message">${commit.message.split('\\n')[0]}</div>
                            <div class="commit-meta">
                                <span>${commit.author}</span>
                                <span>${new Date(commit.timestamp).toLocaleDateString()}</span>
                                <span class="changes">
                                    +${commit.insertions} -${commit.deletions}
                                </span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    async loadDependencies(projectId) {
        try {
            this.showProgress('Loading dependencies...');
            const response = await fetch(`${this.apiBase}/projects/${projectId}/dependencies`);
            const data = await response.json();
            this.hideProgress();
            this.renderDependencyGraph(data);
        } catch (error) {
            this.hideProgress();
            this.showError('Failed to load dependencies');
        }
    }

    renderDependencyGraph(data) {
        const container = document.getElementById('dependencyGraph');
        const nodes = data.nodes || [];
        const edges = data.edges || [];

        container.innerHTML = `
            <div class="graph-info">
                <p>Nodes: ${nodes.length}, Edges: ${edges.length}</p>
            </div>
            <div class="graph-visualization">
                <p>Dependency graph visualization coming soon...</p>
                <p>For now, here's the data:</p>
                <pre>${JSON.stringify(data, null, 2)}</pre>
            </div>
        `;
    }

    async loadComplexityEvolution(projectId) {
        try {
            this.showProgress('Analyzing complexity evolution...');
            const response = await fetch(`${this.apiBase}/projects/${projectId}/evolution/complexity?max_commits=20`);
            const data = await response.json();
            this.hideProgress();
            this.renderComplexityChart(data);
        } catch (error) {
            this.hideProgress();
            console.error('Complexity evolution:', error);
            document.getElementById('complexityChart').innerHTML =
                '<p>Complexity evolution requires a git repository</p>';
        }
    }

    renderComplexityChart(data) {
        const container = document.getElementById('complexityChart');
        const trends = data.complexity_trends || [];

        if (trends.length === 0) {
            container.innerHTML = '<p>No complexity trends available</p>';
            return;
        }

        container.innerHTML = `
            <div class="complexity-trends">
                ${trends.map(trend => `
                    <div class="trend-item">
                        <h4>${trend.file_path}</h4>
                        <div class="trend-stats">
                            <span>Direction: <strong>${trend.trend_direction}</strong></span>
                            <span>Current: ${trend.current_complexity.toFixed(1)}</span>
                            <span>Max: ${trend.max_complexity.toFixed(1)}</span>
                            <span>Min: ${trend.min_complexity.toFixed(1)}</span>
                        </div>
                        <div class="trend-timeline">
                            ${trend.timeline.map((point, i) => `
                                <div class="timeline-point"
                                     style="height: ${(point.complexity / trend.max_complexity * 100)}%"
                                     title="${new Date(point.timestamp).toLocaleDateString()}: ${point.complexity}">
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    showTab(tabName) {
        document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        event.target.classList.add('active');
        document.getElementById(`${tabName}Tab`).classList.add('active');
    }

    getTotalLines(files) {
        return files.reduce((sum, file) => sum + (file.lines_of_code || 0), 0);
    }

    getTotalDefinitions(files) {
        return files.reduce((sum, file) => sum + (file.definitions || []).length, 0);
    }

    showProgress(message) {
        const overlay = document.getElementById('progressOverlay');
        document.getElementById('progressMessage').textContent = message;
        overlay.classList.add('active');
    }

    hideProgress() {
        document.getElementById('progressOverlay').classList.remove('active');
    }

    showError(message) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = 'toast error active';
        setTimeout(() => toast.classList.remove('active'), 4000);
    }

    showSuccess(message) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = 'toast success active';
        setTimeout(() => toast.classList.remove('active'), 3000);
    }
}

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new CodeWarpHouse();
});
