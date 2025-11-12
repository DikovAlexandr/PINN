"""
IEEE-style configuration for publication-ready figures.

This module provides styling functions to create figures that meet
IEEE publication standards with proper sizing, fonts, and layouts.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


def setup_ieee_style(column_width='single', font_size=10, use_scienceplots=True):
    """
    Setup matplotlib for IEEE-style publication figures.
    
    Args:
        column_width (str): 'single' for single-column (3.5 inches) or 
                           'double' for double-column (7.16 inches) figures.
        font_size (int): Base font size in points. IEEE recommends 8-10pt.
        use_scienceplots (bool): If True and scienceplots is available, use it.
                                If False, use custom IEEE style.
    
    Returns:
        tuple: (fig_width, fig_height) in inches for the selected column width.
    
    Example:
        >>> from src.visualization import setup_ieee_style, save_figure
        >>> fig_width, fig_height = setup_ieee_style('single')
        >>> fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        >>> # ... plot data ...
        >>> save_figure(fig, 'results/figure.pdf')
    
    References:
        IEEE Graphics Requirements:
        https://www.ieee.org/publications/authors/author-guidelines.html
    """
    # IEEE column widths (in inches)
    widths = {
        'single': 3.5,   # Single column width
        'double': 7.16,  # Double column width (full page width)
    }
    
    if column_width not in widths:
        raise ValueError(f"column_width must be 'single' or 'double', got: {column_width}")
    
    fig_width = widths[column_width]
    # Golden ratio for height (aesthetically pleasing)
    golden_ratio = (5 ** 0.5 - 1) / 2
    fig_height = fig_width * golden_ratio
    
    # Try to use SciencePlots if available
    scienceplots_available = False
    if use_scienceplots:
        try:
            import scienceplots
            plt.style.use(['science', 'ieee', 'grid', 'no-latex'])
            scienceplots_available = True
            print("Using SciencePlots style (science + ieee + grid)")
        except ImportError:
            print("SciencePlots not available, using custom IEEE style")
            print("Install with: pip install scienceplots")
    
    # Configure matplotlib for publication quality
    # These settings work with or without SciencePlots
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (fig_width, fig_height),
        'figure.dpi': 300,  # High resolution for publications
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
        
        # Font settings (recommended for publications)
        'text.usetex': False,  # Set to True if LaTeX is installed
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        
        # PGF settings (for LaTeX)
        'pgf.texsystem': 'pdflatex',
        
        # Line and marker settings
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelpad': 4,
        'axes.prop_cycle': mpl.cycler(color=[
            '#0173B2', '#DE8F05', '#029E73', '#CC78BC',
            '#CA9161', '#949494', '#ECE133', '#56B4E9'
        ]),  # Colorblind-friendly palette
        
        # Grid settings
        'grid.color': 'gray',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        
        # Tick settings
        'xtick.major.size': 3,
        'xtick.minor.size': 1.5,
        'xtick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'xtick.direction': 'in',
        'xtick.top': True,
        'xtick.minor.visible': True,
        
        'ytick.major.size': 3,
        'ytick.minor.size': 1.5,
        'ytick.major.width': 0.8,
        'ytick.minor.width': 0.6,
        'ytick.direction': 'in',
        'ytick.right': True,
        'ytick.minor.visible': True,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': '0.5',
        'legend.borderpad': 0.4,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 1.5,
        'legend.handleheight': 0.7,
        'legend.handletextpad': 0.5,
    })
    
    return fig_width, fig_height


def save_figure(fig, filepath, dpi=300, transparent=False, **kwargs):
    """
    Save figure in publication-ready format.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        filepath (str or Path): Output file path. Extension determines format.
                               Recommended: .pdf for publications.
        dpi (int): Dots per inch for raster formats. Default: 300 (publication quality).
        transparent (bool): If True, save with transparent background.
        **kwargs: Additional arguments passed to fig.savefig().
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure(fig, 'output/quadratic.pdf')
        >>> save_figure(fig, 'output/quadratic.png', dpi=600)  # High-res PNG
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Default save parameters
    save_params = {
        'dpi': dpi,
        'bbox_inches': 'tight',
        'pad_inches': 0.01,
        'transparent': transparent,
    }
    save_params.update(kwargs)
    
    fig.savefig(filepath, **save_params)
    print(f"Figure saved to: {filepath}")


def get_color_palette(style='default'):
    """
    Get color-blind friendly color palettes for publications.
    
    Args:
        style (str): Color palette style. Options:
                    - 'default': Standard palette with good contrast
                    - 'colorblind': Optimized for color-blind readers
                    - 'grayscale': Grayscale-friendly (for B&W printing)
    
    Returns:
        list: List of color codes (hex strings).
    
    Example:
        >>> colors = get_color_palette('colorblind')
        >>> for i, color in enumerate(colors):
        ...     plt.plot(x, y[i], color=color, label=f'Method {i+1}')
    """
    palettes = {
        'default': [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Yellow-green
            '#17becf',  # Cyan
        ],
        'colorblind': [
            '#0173B2',  # Blue
            '#DE8F05',  # Orange
            '#029E73',  # Green
            '#CC78BC',  # Purple
            '#CA9161',  # Brown
            '#949494',  # Gray
            '#ECE133',  # Yellow
            '#56B4E9',  # Sky blue
        ],
        'grayscale': [
            '#000000',  # Black
            '#4D4D4D',  # Dark gray
            '#7F7F7F',  # Medium gray
            '#B2B2B2',  # Light gray
            '#CCCCCC',  # Very light gray
        ],
    }
    
    if style not in palettes:
        raise ValueError(f"Unknown palette style: {style}. Choose from {list(palettes.keys())}")
    
    return palettes[style]


def get_line_styles():
    """
    Get a list of distinguishable line styles for multiple curves.
    
    Returns:
        list: List of line style specifiers for matplotlib.
    
    Example:
        >>> colors = get_color_palette('default')
        >>> styles = get_line_styles()
        >>> for i, (color, style) in enumerate(zip(colors, styles)):
        ...     plt.plot(x, y[i], color=color, linestyle=style, label=f'Method {i+1}')
    """
    return ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]


def get_marker_styles():
    """
    Get a list of distinguishable marker styles.
    
    Returns:
        list: List of marker style specifiers for matplotlib.
    
    Example:
        >>> markers = get_marker_styles()
        >>> for i, marker in enumerate(markers):
        ...     plt.plot(x, y[i], marker=marker, markevery=10, label=f'Method {i+1}')
    """
    return ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'P', 'h']

