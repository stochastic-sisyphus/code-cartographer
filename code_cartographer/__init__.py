"""Code Cartographer - A tool for visualizing and analyzing code repositories."""

__version__ = "0.1.6"

from code_cartographer.core.analyzer import (
    ProjectAnalyzer,
    CodeInspector,
    ComplexityMetrics,
    DefinitionMetadata,
    FileMetadata,
    generate_markdown,
    generate_dependency_graph
)

from code_cartographer.core.variant_analyzer import (
    VariantAnalyzer,
    CodeNormalizer,
    SemanticAnalyzer,
    VariantGroup,
    VariantMatch
)

__all__ = [
    'ProjectAnalyzer',
    'CodeInspector',
    'ComplexityMetrics',
    'DefinitionMetadata',
    'FileMetadata',
    'generate_markdown',
    'generate_dependency_graph',
    'VariantAnalyzer',
    'CodeNormalizer',
    'SemanticAnalyzer',
    'VariantGroup',
    'VariantMatch'
]
