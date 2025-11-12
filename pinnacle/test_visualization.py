"""
Test script for visualization module.

This script tests the visualization tools on existing experimental results.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add pinnacle directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.visualization import (
    load_experiment_results,
    plot_loss_comparison,
    plot_metric_comparison,
    create_metrics_table,
    compare_experiments,
    setup_ieee_style,
)

print("=" * 70)
print("VISUALIZATION MODULE TEST")
print("=" * 70)
print()

# Setup IEEE style for all plots
print("Setting up IEEE-style plotting...")
fig_width, fig_height = setup_ieee_style('single')
print(f"Figure size: {fig_width:.2f} x {fig_height:.2f} inches")
print(f"DPI: 300 (publication quality)")
print()

# Find available experiments
runs_dir = Path('runs')
if not runs_dir.exists():
    print("No 'runs' directory found. Please run some experiments first.")
    sys.exit(1)

# Get all experiment directories
exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
if not exp_dirs:
    print("No experiments found in 'runs' directory.")
    sys.exit(1)

print(f"Found {len(exp_dirs)} experiment(s) in 'runs/' directory:")
for exp_dir in exp_dirs:
    print(f"  - {exp_dir.name}")
print()

# Test 1: Load a single experiment
print("=" * 70)
print("Test 1: Loading experiment results")
print("=" * 70)

test_exp = exp_dirs[0]
print(f"Loading: {test_exp.name}")

try:
    results = load_experiment_results(test_exp)
    print(f"Successfully loaded experiment")
    print(f"  Tasks found: {list(results['loss'].keys())}")
    
    for task_key, loss_data in results['loss'].items():
        print(f"  {task_key}:")
        print(f"    - Repeats: {len(loss_data)}")
        if loss_data:
            print(f"    - Steps: {len(loss_data[0]['steps'])}")
            print(f"    - Final train loss: {loss_data[0]['train'][-1]:.6e}")
            print(f"    - Final test loss: {loss_data[0]['test'][-1]:.6e}")
    
    if results['metrics']:
        print(f"  Metrics available: {list(results['metrics'].keys())}")
        for task_key, metrics_data in results['metrics'].items():
            if metrics_data:
                print(f"  {task_key}:")
                print(f"    - L2RE (final): {metrics_data[0]['l2re'][-1]:.6e}")
                print(f"    - MSE (final): {metrics_data[0]['mse'][-1]:.6e}")
    
    print()
except Exception as e:
    print(f"Error loading experiment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Plot single experiment (if data available)
print("=" * 70)
print("Test 2: Plotting single experiment convergence")
print("=" * 70)

output_dir = Path('visualization_test')
output_dir.mkdir(exist_ok=True)

try:
    # Prepare data for plotting
    task_key = list(results['loss'].keys())[0]
    histories = {test_exp.name: results['loss'][task_key]}
    
    print(f"Plotting training loss for {test_exp.name}...")
    from src.visualization import plot_statistical_histories
    import matplotlib.pyplot as plt
    
    fig = plot_statistical_histories(
        histories,
        metric='train',
        y_label='Training Loss',
        output_path=output_dir / 'single_exp_train_loss.pdf',
        column_width='single'
    )
    plt.close(fig)
    print(f"Saved: {output_dir / 'single_exp_train_loss.pdf'}")
    
    print(f"Plotting test loss for {test_exp.name}...")
    fig = plot_statistical_histories(
        histories,
        metric='test',
        y_label='Test Loss',
        output_path=output_dir / 'single_exp_test_loss.pdf',
        column_width='single'
    )
    plt.close(fig)
    print(f"Saved: {output_dir / 'single_exp_test_loss.pdf'}")
    
    # Plot metrics if available
    if results['metrics'] and task_key in results['metrics']:
        metric_histories = {test_exp.name: results['metrics'][task_key]}
        
        print(f"Plotting L2RE for {test_exp.name}...")
        fig = plot_statistical_histories(
            metric_histories,
            metric='l2re',
            y_label='L2 Relative Error',
            output_path=output_dir / 'single_exp_l2re.pdf',
            column_width='single'
        )
        plt.close(fig)
        print(f"Saved: {output_dir / 'single_exp_l2re.pdf'}")
    
    print()
except Exception as e:
    print(f"Error plotting: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Compare multiple experiments (if available)
if len(exp_dirs) >= 2:
    print("=" * 70)
    print("Test 3: Comparing multiple experiments")
    print("=" * 70)
    
    # Use first two experiments for comparison
    exp_paths = {exp_dirs[0].name: str(exp_dirs[0]),
                 exp_dirs[1].name: str(exp_dirs[1])}
    
    print(f"Comparing experiments:")
    for name in exp_paths:
        print(f"  - {name}")
    print()
    
    try:
        results = compare_experiments(
            exp_paths,
            task_id=0,
            output_dir=output_dir / 'comparison',
            metrics_to_plot=['l2re', 'mse'],
            column_width='single'
        )
        
        print()
        print("Comparison completed successfully!")
        print(f"  Generated {len(results)} files")
        
    except Exception as e:
        print(f"✗ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
else:
    print("=" * 70)
    print("Test 3: Skipped (need at least 2 experiments for comparison)")
    print("=" * 70)
    print()

# Test 4: Create metrics table
print("=" * 70)
print("Test 4: Creating metrics table")
print("=" * 70)

try:
    # Load all available experiments
    experiments = {}
    for exp_dir in exp_dirs[:3]:  # Limit to first 3 for readability
        try:
            experiments[exp_dir.name] = load_experiment_results(exp_dir)
        except Exception as e:
            print(f"Warning: Skipping {exp_dir.name}: {e}")
    
    if experiments:
        df = create_metrics_table(
            experiments,
            task_id=0,
            metrics=['mse', 'mae', 'l2re', 'mxe'],
            output_path=output_dir / 'metrics_table.csv'
        )
        
        print("\nMetrics Table:")
        print(df.to_string(index=False))
        print(f"\nTable saved to: {output_dir / 'metrics_table.csv'}")
    else:
        print("No valid experiments found for table creation")
    
    print()
except Exception as e:
    print(f"Error creating table: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test different IEEE styles
print("=" * 70)
print("Test 5: Testing IEEE-style variations")
print("=" * 70)

try:
    from src.visualization import get_color_palette, get_line_styles, get_marker_styles
    import matplotlib.pyplot as plt
    
    # Test color palettes
    print("Testing color palettes...")
    for palette_name in ['default', 'colorblind', 'grayscale']:
        colors = get_color_palette(palette_name)
        print(f"  ✓ {palette_name}: {len(colors)} colors")
    
    # Test line styles
    line_styles = get_line_styles()
    print(f"Line styles: {len(line_styles)} variations")
    
    # Test marker styles
    marker_styles = get_marker_styles()
    print(f"Marker styles: {len(marker_styles)} variations")
    
    # Create sample plot with different styles
    print("\nCreating style demonstration plot...")
    setup_ieee_style('double')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))
    
    # Left: Different color palettes
    x = np.linspace(0, 10, 100)
    colors_cb = get_color_palette('colorblind')
    for i, color in enumerate(colors_cb[:4]):
        y = np.exp(-x / (i + 1))
        ax1.semilogy(x, y, color=color, label=f'Method {i+1}', linewidth=1.5)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()
    # Minimal grid for log scale
    ax1.grid(True, which='major', linestyle='-', alpha=0.25, linewidth=0.5, color='gray')
    ax1.grid(True, which='minor', linestyle=':', alpha=0.08, linewidth=0.3, color='gray')
    ax1.yaxis.set_minor_locator(plt.LogLocator(subs='auto'))
    ax1.minorticks_on()
    
    # Right: Different line styles
    line_styles = get_line_styles()
    for i, style in enumerate(line_styles[:4]):
        y = np.exp(-x / (i + 1))
        ax2.semilogy(x, y, linestyle=style, color='black', 
                    label=f'Config {i+1}', linewidth=1.5)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')
    ax2.legend()
    # Minimal grid for log scale
    ax2.grid(True, which='major', linestyle='-', alpha=0.25, linewidth=0.5, color='gray')
    ax2.grid(True, which='minor', linestyle=':', alpha=0.08, linewidth=0.3, color='gray')
    ax2.yaxis.set_minor_locator(plt.LogLocator(subs='auto'))
    ax2.minorticks_on()
    
    plt.tight_layout()
    from src.visualization import save_figure
    save_figure(fig, output_dir / 'style_demonstration.pdf')
    plt.close(fig)
    
    print(f"Saved: {output_dir / 'style_demonstration.pdf'}")
    print()
    
except Exception as e:
    print(f"Error testing styles: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("ALL TESTS COMPLETED!")
print("=" * 70)
print()
print(f"All output files saved to: {output_dir}")
print()
print("You can now use the visualization module in your experiments:")
print()
print("  from src.visualization import compare_experiments")
print()
print("  compare_experiments({")
print("      'Adam': 'runs/adam_exp',")
print("      'L-BFGS': 'runs/lbfgs_exp',")
print("  }, output_dir='paper_figures')")
print()
print("=" * 70)

