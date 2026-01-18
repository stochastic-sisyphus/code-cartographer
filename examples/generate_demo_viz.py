#!/usr/bin/env python3
"""
Demo script to generate an immersive visualization with sample data.
This showcases the dynamic visualization capabilities of Code Cartographer.
"""

import json
from pathlib import Path

# Sample data representing a small codebase
sample_data = {
    "nodes": [
        {
            "id": "auth/login.py",
            "name": "login",
            "type": "function",
            "complexity": 25,
            "importance": 5,
            "description": "User authentication and login management"
        },
        {
            "id": "auth/session.py",
            "name": "session",
            "type": "class",
            "complexity": 45,
            "importance": 8,
            "description": "Session management and state tracking"
        },
        {
            "id": "auth/permissions.py",
            "name": "permissions",
            "type": "class",
            "complexity": 60,
            "importance": 6,
            "description": "User permissions and access control"
        },
        {
            "id": "api/handlers.py",
            "name": "handlers",
            "type": "function",
            "complexity": 35,
            "importance": 7,
            "description": "API request handlers"
        },
        {
            "id": "api/validators.py",
            "name": "validators",
            "type": "function",
            "complexity": 20,
            "importance": 4,
            "description": "Input validation utilities"
        },
        {
            "id": "db/models.py",
            "name": "models",
            "type": "class",
            "complexity": 55,
            "importance": 9,
            "description": "Database models and schemas"
        },
        {
            "id": "db/migrations.py",
            "name": "migrations",
            "type": "function",
            "complexity": 30,
            "importance": 3,
            "description": "Database migration utilities"
        },
        {
            "id": "utils/crypto.py",
            "name": "crypto",
            "type": "function",
            "complexity": 40,
            "importance": 5,
            "description": "Cryptographic utilities"
        },
        {
            "id": "utils/logging.py",
            "name": "logging",
            "type": "module",
            "complexity": 15,
            "importance": 2,
            "description": "Application logging configuration"
        },
        {
            "id": "config.py",
            "name": "config",
            "type": "module",
            "complexity": 10,
            "importance": 10,
            "description": "Application configuration"
        },
        {
            "id": "main.py",
            "name": "main",
            "type": "function",
            "complexity": 20,
            "importance": 8,
            "description": "Application entry point"
        }
    ],
    "links": [
        # Auth dependencies
        {"source": "auth/login.py", "target": "auth/session.py", "type": "harmony", "strength": 0.9},
        {"source": "auth/login.py", "target": "auth/permissions.py", "type": "harmony", "strength": 0.7},
        {"source": "auth/login.py", "target": "utils/crypto.py", "type": "neutral", "strength": 0.6},
        {"source": "auth/session.py", "target": "db/models.py", "type": "harmony", "strength": 0.8},
        {"source": "auth/permissions.py", "target": "db/models.py", "type": "harmony", "strength": 0.7},
        
        # API dependencies
        {"source": "api/handlers.py", "target": "auth/session.py", "type": "neutral", "strength": 0.8},
        {"source": "api/handlers.py", "target": "api/validators.py", "type": "harmony", "strength": 0.9},
        {"source": "api/handlers.py", "target": "db/models.py", "type": "neutral", "strength": 0.7},
        {"source": "api/validators.py", "target": "utils/logging.py", "type": "harmony", "strength": 0.4},
        
        # Database dependencies
        {"source": "db/models.py", "target": "config.py", "type": "harmony", "strength": 0.6},
        {"source": "db/migrations.py", "target": "db/models.py", "type": "tension", "strength": 0.8},
        
        # Main app dependencies
        {"source": "main.py", "target": "config.py", "type": "harmony", "strength": 0.9},
        {"source": "main.py", "target": "api/handlers.py", "type": "neutral", "strength": 0.8},
        {"source": "main.py", "target": "auth/login.py", "type": "neutral", "strength": 0.7},
        {"source": "main.py", "target": "utils/logging.py", "type": "harmony", "strength": 0.5},
        
        # Utility dependencies
        {"source": "utils/crypto.py", "target": "config.py", "type": "harmony", "strength": 0.4},
        {"source": "utils/logging.py", "target": "config.py", "type": "harmony", "strength": 0.5},
    ],
    "timeline": [],
    "metadata": {
        "generated_at": "2026-01-18T20:00:00",
        "total_nodes": 11,
        "total_links": 17,
        "avg_complexity": 33.18
    }
}

def generate_demo_visualization():
    """Generate a demo visualization HTML file."""
    from jinja2 import Environment, FileSystemLoader
    
    # Get template directory
    template_dir = Path(__file__).parent.parent / 'templates'
    
    # Load template
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template('immersive_dashboard.html.j2')
    
    # Render
    html_content = template.render(data=sample_data)
    
    # Save
    output_path = Path('/tmp/demo_visualization.html')
    output_path.write_text(html_content)
    
    print(f"Demo visualization generated: {output_path}")
    print(f"Open {output_path} in your browser to explore!")
    print()
    print("This demo shows:")
    print("  - 11 code modules with varying complexity")
    print("  - 17 dependencies showing harmony, tension, and neutral relationships")
    print("  - Interactive force-directed graph")
    print("  - Hover to see details, drag to explore")
    
    # Also save the data as JSON
    json_path = Path('/tmp/demo_visualization.json')
    with json_path.open('w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"Visualization data: {json_path}")

if __name__ == '__main__':
    generate_demo_visualization()
