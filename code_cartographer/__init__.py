"""Code Cartographer - A tool for visualizing and analyzing code repositories."""

__version__ = "0.2.1"

from code_cartographer.core.analyzer import (
    ProjectAnalyzer,
    CodeInspector,
    ComplexityMetrics,
    DefinitionMetadata,
    FileMetadata,
    CodeAnalyzer,
)

from code_cartographer.core.variable_analyzer import (
    VariableAnalyzer,
    VariableDefinition,
    VariableUsage,
    VariableFlow,
)

from code_cartographer.core.dependency_analyzer import (
    DependencyAnalyzer,
    DependencyNode,
)

from code_cartographer.core.visualizer import CodeVisualizer

from code_cartographer.core.reporter import ReportGenerator

from code_cartographer.core.variant_analyzer import (
    VariantAnalyzer,
    CodeNormalizer,
    SemanticAnalyzer,
    VariantGroup,
    VariantMatch,
)

__all__ = [
    "ProjectAnalyzer",
    "CodeInspector",
    "ComplexityMetrics",
    "DefinitionMetadata",
    "FileMetadata",
    "CodeAnalyzer",
    "VariableAnalyzer",
    "VariableDefinition",
    "VariableUsage",
    "VariableFlow",
    "DependencyAnalyzer",
    "DependencyNode",
    "CodeVisualizer",
    "ReportGenerator",
    "VariantAnalyzer",
    "CodeNormalizer",
    "SemanticAnalyzer",
    "VariantGroup",
    "VariantMatch",
]
