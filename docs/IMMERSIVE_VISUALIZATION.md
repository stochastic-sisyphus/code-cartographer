# Immersive Visualization Guide

## Overview

Code Cartographer's immersive visualization transforms static code analysis into a living, breathing exploration of your codebase. Instead of flat charts and tables, you get a dynamic, interactive force-directed graph that reveals the relationships, tensions, and evolution of your code.

![Immersive Visualization](https://github.com/user-attachments/assets/5cbaec9c-ec04-40dc-a533-e94599f144d8)

## Features

### Dynamic Force-Directed Graph
- **Physics Simulation**: Code elements attract and repel each other based on dependencies
- **Interactive Exploration**: Drag nodes, zoom, pan, and hover for details
- **Real-time Updates**: Watch your codebase reorganize as you adjust parameters

### Visual Metaphors
- **Harmony (Green)**: Well-balanced dependencies with similar complexity
- **Tension (Red/Orange)**: Mismatched complexity or problematic relationships
- **Neutral (White)**: Standard dependencies

### Temporal Evolution
- **Timeline Slider**: Travel through your codebase's history
- **Git Integration**: Analyze how code evolved commit by commit
- **Velocity Metrics**: See rates of change and development patterns

## Quick Start

Generate a visualization:

```bash
# Basic visualization
code-cartographer visualize -d /path/to/project -o visualization.html

# With temporal evolution
code-cartographer visualize -d /path/to/project -o viz.html --temporal --max-commits 50

# Export data as JSON
code-cartographer visualize -d /path/to/project -o viz.html --export-json
```

See the [full documentation](../README.md) for more details.
