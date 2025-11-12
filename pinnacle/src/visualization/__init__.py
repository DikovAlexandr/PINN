"""
Visualization module for comparing PINNacle experiments.

This module provides tools for:
- Loading and aggregating experimental results
- Plotting convergence curves with confidence intervals
- Comparing metrics across different methods
- Generating publication-ready figures in IEEE style
"""

from .compare import (
    load_experiment_results,
    plot_loss_comparison,
    plot_metric_comparison,
    plot_statistical_histories,
    create_metrics_table,
    compare_experiments,
)
from .ieee_style import (
    setup_ieee_style, 
    save_figure,
    get_color_palette,
    get_line_styles,
    get_marker_styles,
)

__all__ = [
    'load_experiment_results',
    'plot_loss_comparison',
    'plot_metric_comparison',
    'plot_statistical_histories',
    'create_metrics_table',
    'compare_experiments',
    'setup_ieee_style',
    'save_figure',
    'get_color_palette',
    'get_line_styles',
    'get_marker_styles',
]

