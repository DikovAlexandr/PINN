"""
Comparison and visualization tools for PINNacle experiments.

This module provides functions to load, aggregate, and visualize results
from multiple experimental runs, enabling statistical comparison of different
methods and architectures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

from .ieee_style import setup_ieee_style, save_figure, get_color_palette, get_line_styles


def load_experiment_results(exp_path: Union[str, Path], 
                            task_ids: Optional[List[int]] = None,
                            repeat_ids: Optional[List[int]] = None) -> Dict:
    """
    Load results from a single experiment with multiple runs.
    
    Args:
        exp_path: Path to experiment directory (e.g., 'runs/adam_experiment').
        task_ids: List of task IDs to load. If None, load all available tasks.
        repeat_ids: List of repeat IDs to load. If None, load all available repeats.
    
    Returns:
        dict: Dictionary with structure:
            {
                'loss': {
                    'task-0': [
                        {'steps': [...], 'train': [...], 'test': [...]},  # repeat 0
                        {'steps': [...], 'train': [...], 'test': [...]},  # repeat 1
                    ],
                    'task-1': [...],
                },
                'metrics': {
                    'task-0': [
                        {'epochs': [...], 'mse': [...], 'l2re': [...], ...},  # repeat 0
                    ],
                },
                'config': {...},  # Experiment configuration
            }
    
    Example:
        >>> results = load_experiment_results('runs/adam_exp')
        >>> print(f"Tasks: {list(results['loss'].keys())}")
        >>> print(f"Repeats for task-0: {len(results['loss']['task-0'])}")
    """
    exp_path = Path(exp_path)
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {exp_path}")
    
    # Load configuration if available
    config = {}
    config_file = exp_path / 'config.json'
    if config_file.exists():
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    # Find all task-repeat combinations
    run_dirs = [d for d in exp_path.iterdir() if d.is_dir() and '-' in d.name]
    
    # Parse task and repeat IDs
    available_runs = []
    for run_dir in run_dirs:
        try:
            task_id, repeat_id = map(int, run_dir.name.split('-'))
            available_runs.append((task_id, repeat_id, run_dir))
        except ValueError:
            continue  # Skip directories that don't match pattern
    
    # Filter by requested IDs
    if task_ids is not None:
        available_runs = [(t, r, d) for t, r, d in available_runs if t in task_ids]
    if repeat_ids is not None:
        available_runs = [(t, r, d) for t, r, d in available_runs if r in repeat_ids]
    
    # Organize results
    loss_data = defaultdict(list)
    metrics_data = defaultdict(list)
    
    for task_id, repeat_id, run_dir in sorted(available_runs):
        task_key = f'task-{task_id}'
        
        # Load loss history
        loss_file = run_dir / 'loss.txt'
        if loss_file.exists():
            try:
                data = np.loadtxt(loss_file)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                # Parse loss.txt format:
                # step, loss_train (components), loss_test (components), loss_weight (components)
                n_cols = data.shape[1]
                steps = data[:, 0]
                
                # Determine number of loss components
                # Format: step + n_train + n_test + n_weights = 1 + 3*n
                n_components = (n_cols - 1) // 3
                
                train_losses = data[:, 1:1+n_components]
                test_losses = data[:, 1+n_components:1+2*n_components]
                
                # Sum components to get total loss
                train_total = train_losses.sum(axis=1) if train_losses.ndim > 1 else train_losses
                test_total = test_losses.sum(axis=1) if test_losses.ndim > 1 else test_losses
                
                loss_data[task_key].append({
                    'steps': steps,
                    'train': train_total,
                    'test': test_total,
                    'train_components': train_losses,
                    'test_components': test_losses,
                    'repeat_id': repeat_id,
                })
            except Exception as e:
                print(f"Warning: Failed to load loss from {loss_file}: {e}")
        
        # Load metrics
        errors_file = run_dir / 'errors.txt'
        if errors_file.exists():
            try:
                data = np.loadtxt(errors_file)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                # Parse errors.txt format:
                # epochs, maes, mses, mxes, l1res, l2res, crmses, frmses(low, mid, high)
                metrics_data[task_key].append({
                    'epochs': data[:, 0],
                    'mae': data[:, 1],
                    'mse': data[:, 2],
                    'mxe': data[:, 3],
                    'l1re': data[:, 4],
                    'l2re': data[:, 5],
                    'crmse': data[:, 6],
                    'frmse_low': data[:, 7] if data.shape[1] > 7 else None,
                    'frmse_mid': data[:, 8] if data.shape[1] > 8 else None,
                    'frmse_high': data[:, 9] if data.shape[1] > 9 else None,
                    'repeat_id': repeat_id,
                })
            except Exception as e:
                print(f"Warning: Failed to load metrics from {errors_file}: {e}")
    
    return {
        'loss': dict(loss_data),
        'metrics': dict(metrics_data),
        'config': config,
        'exp_path': exp_path,
    }


def plot_statistical_histories(histories_per_experiment: Dict[str, List[Dict]],
                               metric: str = 'loss',
                               y_scale: str = 'log',
                               x_label: str = None,
                               y_label: str = None,
                               output_path: Optional[str] = None,
                               column_width: str = 'single',
                               show_legend: bool = True,
                               legend_loc: str = 'best',
                               x_limits: Optional[Tuple[float, float]] = None,
                               y_limits: Optional[Tuple[float, float]] = None,
                               confidence_alpha: float = 0.2) -> plt.Figure:
    """
    Plot mean convergence curves with confidence intervals (mean ± std).
    
    This function aggregates multiple runs of the same experiment, computes
    mean and standard deviation, and plots them with a shaded confidence region.
    
    Args:
        histories_per_experiment: Dictionary mapping experiment names to lists
                                 of history dictionaries. Each history should have
                                 'steps' and the specified metric key (e.g., 'train', 'test').
        metric: Metric to plot. Options: 'train', 'test', 'l2re', 'mse', etc.
        y_scale: Y-axis scale. Options: 'log', 'linear'.
        x_label: X-axis label. If None, uses default based on metric.
        y_label: Y-axis label. If None, uses default based on metric.
        output_path: If provided, save figure to this path (PDF recommended).
        column_width: 'single' or 'double' for IEEE-style sizing.
        show_legend: Whether to show legend.
        legend_loc: Legend location ('best', 'upper right', etc.).
        x_limits: Tuple (xmin, xmax) for x-axis limits. If None, auto-scale.
        y_limits: Tuple (ymin, ymax) for y-axis limits. If None, auto-scale.
        confidence_alpha: Alpha (transparency) for confidence interval shading.
    
    Returns:
        matplotlib.figure.Figure: The created figure.
    
    Example:
        >>> # Load results from multiple experiments
        >>> exp1 = load_experiment_results('runs/adam_exp')
        >>> exp2 = load_experiment_results('runs/lbfgs_exp')
        >>> 
        >>> # Prepare data
        >>> histories = {
        ...     'Adam': exp1['loss']['task-0'],
        ...     'L-BFGS': exp2['loss']['task-0'],
        ... }
        >>> 
        >>> # Plot with confidence intervals
        >>> fig = plot_statistical_histories(
        ...     histories, 
        ...     metric='test',
        ...     output_path='results/convergence_comparison.pdf'
        ... )
    """
    # Setup IEEE style
    fig_width, fig_height = setup_ieee_style(column_width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Get color palette
    colors = get_color_palette('colorblind')
    line_styles = get_line_styles()
    
    # Determine common x-axis for interpolation
    max_steps = 0
    for name, trial_histories in histories_per_experiment.items():
        for h in trial_histories:
            x_key = 'steps' if 'steps' in h else 'epochs'
            if x_key in h and len(h[x_key]) > 0:
                max_steps = max(max_steps, h[x_key][-1])
    
    if max_steps == 0:
        raise ValueError("No valid data found in histories")
    
    # Create common x-axis (200 points for smooth curves)
    common_x_axis = np.linspace(0, max_steps, 200)
    
    # Plot each experiment
    for idx, (name, trial_histories) in enumerate(histories_per_experiment.items()):
        interpolated_values = []
        
        for h in trial_histories:
            # Get x and y data
            x_key = 'steps' if 'steps' in h else 'epochs'
            if x_key not in h or metric not in h:
                continue
            
            x_data = h[x_key]
            y_data = h[metric]
            
            if len(x_data) == 0 or len(y_data) == 0:
                continue
            
            # Handle NaN values
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            
            if len(x_data) < 2:
                continue
            
            # Interpolate to common x-axis
            interp_y = np.interp(common_x_axis, x_data, y_data,
                               left=y_data[0], right=y_data[-1])
            interpolated_values.append(interp_y)
        
        if len(interpolated_values) == 0:
            print(f"Warning: No valid data for {name}")
            continue
        
        # Compute statistics
        interpolated_values = np.array(interpolated_values)
        mean_curve = np.mean(interpolated_values, axis=0)
        std_curve = np.std(interpolated_values, axis=0)
        
        # Plot mean line
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        
        if y_scale == 'log':
            line, = ax.semilogy(common_x_axis, mean_curve, 
                              label=name, color=color, linestyle=linestyle, linewidth=1.5)
        else:
            line, = ax.plot(common_x_axis, mean_curve,
                          label=name, color=color, linestyle=linestyle, linewidth=1.5)
        
        # Plot confidence interval (mean ± std)
        lower_bound = np.maximum(mean_curve - std_curve, 1e-10 if y_scale == 'log' else -np.inf)
        upper_bound = mean_curve + std_curve
        
        ax.fill_between(common_x_axis, lower_bound, upper_bound,
                       color=color, alpha=confidence_alpha)
    
    # Set labels
    if x_label is None:
        x_label = 'Steps' if any('steps' in h for histories in histories_per_experiment.values() 
                                for h in histories) else 'Epochs'
    if y_label is None:
        metric_labels = {
            'train': 'Training Loss',
            'test': 'Test Loss',
            'l2re': 'L2 Relative Error',
            'mse': 'Mean Square Error',
            'mae': 'Mean Absolute Error',
            'mxe': 'Maximum Error',
        }
        y_label = metric_labels.get(metric, metric.upper())
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Set limits
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    elif y_scale == 'log':
        # Set reasonable lower limit for log scale
        ax.set_ylim(bottom=1e-6)
    
    # Legend
    if show_legend:
        ax.legend(loc=legend_loc, framealpha=0.9)
    
    # Grid
    if y_scale == 'log':
        ax.grid(True, which='major', linestyle='-', alpha=0.25, linewidth=0.5, color='gray')
        ax.grid(True, which='minor', linestyle=':', alpha=0.08, linewidth=0.3, color='gray')
        ax.yaxis.set_minor_locator(plt.LogLocator(subs='auto'))
    else:
        ax.grid(True, which='major', linestyle='--', alpha=0.25, linewidth=0.5, color='gray')
        ax.grid(False, which='minor')
    
    ax.minorticks_on()
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path is not None:
        save_figure(fig, output_path)
    
    return fig


def plot_loss_comparison(experiments: Dict[str, Dict],
                        task_id: int = 0,
                        loss_type: str = 'test',
                        output_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
    """
    Compare loss convergence across multiple experiments.
    
    Convenience wrapper around plot_statistical_histories for loss comparison.
    
    Args:
        experiments: Dictionary mapping experiment names to loaded results
                    (from load_experiment_results).
        task_id: Task ID to plot.
        loss_type: 'train' or 'test'.
        output_path: If provided, save figure to this path.
        **kwargs: Additional arguments passed to plot_statistical_histories.
    
    Returns:
        matplotlib.figure.Figure: The created figure.
    
    Example:
        >>> experiments = {
        ...     'Adam': load_experiment_results('runs/adam_exp'),
        ...     'L-BFGS': load_experiment_results('runs/lbfgs_exp'),
        ... }
        >>> fig = plot_loss_comparison(experiments, output_path='results/loss.pdf')
    """
    task_key = f'task-{task_id}'
    histories = {}
    
    for name, exp_data in experiments.items():
        if task_key in exp_data['loss']:
            histories[name] = exp_data['loss'][task_key]
    
    if not histories:
        raise ValueError(f"No data found for task {task_id}")
    
    return plot_statistical_histories(
        histories,
        metric=loss_type,
        y_label=f'{loss_type.capitalize()} Loss',
        output_path=output_path,
        **kwargs
    )


def plot_metric_comparison(experiments: Dict[str, Dict],
                          task_id: int = 0,
                          metric: str = 'l2re',
                          output_path: Optional[str] = None,
                          **kwargs) -> plt.Figure:
    """
    Compare a specific metric across multiple experiments.
    
    Args:
        experiments: Dictionary mapping experiment names to loaded results.
        task_id: Task ID to plot.
        metric: Metric name ('l2re', 'mse', 'mae', 'mxe', etc.).
        output_path: If provided, save figure to this path.
        **kwargs: Additional arguments passed to plot_statistical_histories.
    
    Returns:
        matplotlib.figure.Figure: The created figure.
    
    Example:
        >>> experiments = {
        ...     'Adam': load_experiment_results('runs/adam_exp'),
        ...     'Muon': load_experiment_results('runs/muon_exp'),
        ... }
        >>> fig = plot_metric_comparison(
        ...     experiments, 
        ...     metric='l2re',
        ...     output_path='results/l2re_comparison.pdf'
        ... )
    """
    task_key = f'task-{task_id}'
    histories = {}
    
    for name, exp_data in experiments.items():
        if task_key in exp_data['metrics']:
            histories[name] = exp_data['metrics'][task_key]
    
    if not histories:
        raise ValueError(f"No metric data found for task {task_id}")
    
    return plot_statistical_histories(
        histories,
        metric=metric,
        output_path=output_path,
        **kwargs
    )


def create_metrics_table(experiments: Dict[str, Dict],
                        task_id: int = 0,
                        metrics: Optional[List[str]] = None,
                        output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a table comparing final metrics across experiments.
    
    Args:
        experiments: Dictionary mapping experiment names to loaded results.
        task_id: Task ID to summarize.
        metrics: List of metrics to include. If None, use all available.
        output_path: If provided, save table to CSV/LaTeX file.
    
    Returns:
        pandas.DataFrame: Table with mean ± std for each metric.
    
    Example:
        >>> experiments = {
        ...     'Adam': load_experiment_results('runs/adam_exp'),
        ...     'L-BFGS': load_experiment_results('runs/lbfgs_exp'),
        ... }
        >>> df = create_metrics_table(
        ...     experiments, 
        ...     metrics=['mse', 'l2re', 'mxe'],
        ...     output_path='results/metrics_table.csv'
        ... )
        >>> print(df)
    """
    if metrics is None:
        metrics = ['mse', 'mae', 'mxe', 'l1re', 'l2re', 'crmse']
    
    task_key = f'task-{task_id}'
    results = []
    
    for exp_name, exp_data in experiments.items():
        if task_key not in exp_data['metrics']:
            continue
        
        row = {'Experiment': exp_name}
        
        for metric in metrics:
            values = []
            for repeat_data in exp_data['metrics'][task_key]:
                if metric in repeat_data and repeat_data[metric] is not None:
                    # Use final value (last epoch)
                    metric_vals = repeat_data[metric]
                    if not np.isnan(metric_vals[-1]):
                        values.append(metric_vals[-1])
            
            if values:
                mean = np.mean(values)
                std = np.std(values)
                # Format as "mean ± std" with scientific notation
                row[metric.upper()] = f"{mean:.3e} ± {std:.3e}"
            else:
                row[metric.upper()] = "N/A"
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.tex':
            df.to_latex(output_path, index=False, escape=False)
        else:
            df.to_csv(output_path, index=False)
        
        print(f"Table saved to: {output_path}")
    
    return df


def compare_experiments(exp_paths: Dict[str, str],
                       task_id: int = 0,
                       output_dir: str = 'comparison_results',
                       metrics_to_plot: Optional[List[str]] = None,
                       column_width: str = 'single'):
    """
    Comprehensive comparison of multiple experiments.
    
    This is a high-level function that generates a complete set of comparison
    plots and tables for multiple experiments.
    
    Args:
        exp_paths: Dictionary mapping experiment names to their paths.
        task_id: Task ID to compare.
        output_dir: Directory to save all output files.
        metrics_to_plot: List of metrics to plot. If None, plots loss and L2RE.
        column_width: 'single' or 'double' for figure sizing.
    
    Returns:
        dict: Dictionary with paths to all generated files.
    
    Example:
        >>> exp_paths = {
        ...     'Adam': 'runs/adam_exp',
        ...     'L-BFGS': 'runs/lbfgs_exp',
        ...     'Muon': 'runs/muon_exp',
        ... }
        >>> results = compare_experiments(
        ...     exp_paths,
        ...     output_dir='paper_figures',
        ...     column_width='double'
        ... )
        >>> print(f"Generated files: {list(results.keys())}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    print("Loading experiments...")
    experiments = {}
    for name, path in exp_paths.items():
        try:
            experiments[name] = load_experiment_results(path, task_ids=[task_id])
            print(f" Loaded: {name}")
        except Exception as e:
            print(f" Failed to load {name}: {e}")
    
    if not experiments:
        raise ValueError("No experiments loaded successfully")
    
    generated_files = {}
    
    # Plot training loss
    print("\nGenerating training loss comparison...")
    try:
        fig = plot_loss_comparison(
            experiments,
            task_id=task_id,
            loss_type='train',
            output_path=output_dir / 'train_loss.pdf',
            column_width=column_width
        )
        plt.close(fig)
        generated_files['train_loss'] = str(output_dir / 'train_loss.pdf')
        print(" Saved: train_loss.pdf")
    except Exception as e:
        print(f" Error: {e}")
    
    # Plot test loss
    print("Generating test loss comparison...")
    try:
        fig = plot_loss_comparison(
            experiments,
            task_id=task_id,
            loss_type='test',
            output_path=output_dir / 'test_loss.pdf',
            column_width=column_width
        )
        plt.close(fig)
        generated_files['test_loss'] = str(output_dir / 'test_loss.pdf')
        print(" Saved: test_loss.pdf")
    except Exception as e:
        print(f" Error: {e}")
    
    # Plot metrics
    if metrics_to_plot is None:
        metrics_to_plot = ['l2re', 'mse']
    
    for metric in metrics_to_plot:
        print(f"Generating {metric.upper()} comparison...")
        try:
            fig = plot_metric_comparison(
                experiments,
                task_id=task_id,
                metric=metric,
                output_path=output_dir / f'{metric}_comparison.pdf',
                column_width=column_width
            )
            plt.close(fig)
            generated_files[f'{metric}_comparison'] = str(output_dir / f'{metric}_comparison.pdf')
            print(f" Saved: {metric}_comparison.pdf")
        except Exception as e:
            print(f" Error: {e}")
    
    # Create metrics table
    print("Generating metrics table...")
    try:
        df = create_metrics_table(
            experiments,
            task_id=task_id,
            output_path=output_dir / 'metrics_table.csv'
        )
        generated_files['metrics_table'] = str(output_dir / 'metrics_table.csv')
        print(" Saved: metrics_table.csv")
        print("\nMetrics Summary:")
        print(df.to_string(index=False))
    except Exception as e:
        print(f" Error: {e}")
    
    print(f"\n All results saved to: {output_dir}")
    
    return generated_files

