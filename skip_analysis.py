#!/usr/bin/env python3
"""
analysis.py

A script to analyze scikit-optimize Bayesian Optimization checkpoint files.
Extracts key metrics and generates informative plots to visualize optimization progress.

Usage:
    python analysis.py --checkpoint_dir ./outputs --output_dir ./plots --skip_iterations 1
"""

import os
import argparse
import joblib
import dill
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_regret, plot_gaussian_process
from skopt.space import Space, Real
from skopt import gp_minimize
from scipy.optimize import OptimizeResult    

from main import cost_fcn_partial, cost_fcn  # Ensure this import works based on your project structure

def find_checkpoint_files(checkpoint_dir, extension='.pkl'):
    """
    Search for checkpoint files with the given extension in the specified directory.

    Parameters:
        checkpoint_dir (str): Path to the directory containing checkpoint files.
        extension (str): File extension to look for.

    Returns:
        list of str: Paths to checkpoint files.
    """
    checkpoint_files = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(extension):
                checkpoint_files.append(os.path.join(root, file))
    return checkpoint_files

def load_checkpoint(checkpoint_file, max_bytes_to_trim=10000, step=100):
    """
    Load the checkpoint file using dill. If the file is corrupted, attempt to load
    as much as possible by trimming bytes from the end.

    Parameters:
        checkpoint_file (str): Path to the checkpoint file.
        max_bytes_to_trim (int): Maximum number of bytes to remove from the end.
        step (int): Number of bytes to remove in each step.

    Returns:
        dict: Loaded checkpoint data, possibly incomplete. Returns None if loading fails.
    """
    try:
        with open(checkpoint_file, 'rb') as f:
            data = f.read()
        obj = dill.loads(data)
        print(f"Successfully loaded checkpoint file: {checkpoint_file}")
        return obj
    except (EOFError, dill.UnpicklingError, AttributeError, ValueError) as e:
        print(f"Error loading checkpoint file {checkpoint_file}: {e}")
        print("Attempting to load partial checkpoint by adding fake bytes...")
        
        for trim in range(step, max_bytes_to_trim + step, step):
            try:
                partial_data = data[:-trim]
                obj = dill.loads(partial_data)
                print(f"Successfully loaded partial checkpoint by trimming {trim} bytes.")
                return obj
            except (EOFError, dill.UnpicklingError, AttributeError, ValueError) as e_partial:
                continue  # Continue trimming
        print(f"Failed to load checkpoint file {checkpoint_file} even after trimming {max_bytes_to_trim} bytes.")
        return None

def extract_metrics(checkpoint_data, penalty_threshold=1e5, skip_iterations=1):
    """
    Extract key metrics from the checkpoint data, excluding the first few iterations.

    Parameters:
        checkpoint_data (dict): Data loaded from the checkpoint file.
        penalty_threshold (float): Threshold to consider a function evaluation as valid.
        skip_iterations (int): Number of initial iterations to skip.

    Returns:
        dict: Extracted metrics including best cost, best parameters, cumulative best, etc.
    """
    metrics = {}
    try:
        x_iters = checkpoint_data.get('x_iters', [])
        func_vals = checkpoint_data.get('func_vals', [])
        space = checkpoint_data.get('space', None)
        models = checkpoint_data.get('models', [])

        if x_iters is None or func_vals is None:
            print("No iterations or function values found in checkpoint.")
            return metrics

        # Ensure there are enough iterations to skip
        if len(x_iters) <= skip_iterations or len(func_vals) <= skip_iterations:
            print(f"Not enough iterations to skip the first {skip_iterations} iterations.")
            return metrics

        # Skip the first 'skip_iterations' iterations
        x_iters = x_iters[skip_iterations:]
        func_vals = func_vals[skip_iterations:]

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(x_iters, columns=[dim.name for dim in space.dimensions])
        df['func_val'] = func_vals

        # Identify invalid points (assuming high penalty values)
        df['is_valid'] = df['func_val'] < penalty_threshold

        # Best cost and parameters
        valid_df = df[df['is_valid']]
        if not valid_df.empty:
            best_idx = valid_df['func_val'].idxmin()
            metrics['best_cost'] = valid_df.loc[best_idx, 'func_val']
            metrics['best_params'] = valid_df.loc[best_idx].drop(['func_val', 'is_valid']).to_dict()
        else:
            metrics['best_cost'] = np.nan
            metrics['best_params'] = None

        # Cumulative best
        df['cumulative_best'] = df['func_val'].cummin()

        # Function evaluation statistics
        metrics['func_vals'] = df['func_val'].values
        metrics['x_iters'] = x_iters
        metrics['cumulative_best'] = df['cumulative_best'].values
        metrics['is_valid'] = df['is_valid'].values

        # Function evaluation statistics
        metrics['mean_cost'] = df['func_val'].mean()
        metrics['median_cost'] = df['func_val'].median()
        metrics['std_cost'] = df['func_val'].std()

        # Number of valid and invalid points
        metrics['num_valid'] = valid_df.shape[0]
        metrics['num_invalid'] = df.shape[0] - valid_df.shape[0]

        # Handle uncertainty if models are available
        if models:
            # Extract the latest model
            latest_model = models[-1]
            metrics['latest_model'] = latest_model
        else:
            metrics['latest_model'] = None

        return metrics

    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return metrics

def plot_convergence_custom(metrics, output_path=None):
    """
    Plot the convergence of the optimization.

    Parameters:
        metrics (dict): Extracted metrics containing 'cumulative_best'.
        output_path (str): Path to save the plot. If None, display the plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['cumulative_best'], marker='o', linestyle='-')
        plt.title('Convergence Plot')
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Best Cost')
        plt.grid(True)
        if output_path:
            plt.savefig(os.path.join(output_path, 'convergence_plot.png'))
            plt.close()
            print(f"Convergence plot saved to {output_path}/convergence_plot.png")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting convergence: {e}")

def plot_function_values(metrics, output_path=None):
    """
    Plot all function evaluations over iterations.

    Parameters:
        metrics (dict): Extracted metrics containing 'func_vals'.
        output_path (str): Path to save the plot. If None, display the plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['func_vals'], marker='x', linestyle='-', label='Function Value')
        plt.title('Function Evaluations Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.legend()
        plt.grid(True)
        if output_path:
            plt.savefig(os.path.join(output_path, 'function_values_plot.png'))
            plt.close()
            print(f"Function values plot saved to {output_path}/function_values_plot.png")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting function values: {e}")

def plot_kdeplot(metrics, output_path=None):
    """
    Plot a kdeplot of function evaluations.

    Parameters:
        metrics (dict): Extracted metrics containing 'func_vals'.
        output_path (str): Path to save the plot. If None, display the plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(metrics['func_vals'], shade=True)
        plt.title('KDE of Function Evaluations')
        plt.xlabel('Function Value')
        plt.ylabel('Frequency')
        if output_path:
            plt.savefig(os.path.join(output_path, 'function_values_kdeplot.png'))
            plt.close()
            print(f"Function values kdeplot saved to {output_path}/kdeplot.png")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting kdeplot: {e}")

def plot_improvement(metrics, output_path=None):
    """
    Plot the improvement of the best cost over iterations.

    Parameters:
        metrics (dict): Extracted metrics containing 'cumulative_best'.
        output_path (str): Path to save the plot. If None, display the plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        improvement = metrics['cumulative_best'][0] - metrics['cumulative_best']
        plt.plot(improvement, marker='o', linestyle='-')
        plt.title('Improvement Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Improvement from Initial Best')
        plt.grid(True)
        if output_path:
            plt.savefig(os.path.join(output_path, 'improvement_plot.png'))
            plt.close()
            print(f"Improvement plot saved to {output_path}/improvement_plot.png")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting improvement: {e}")

def plot_regret(metrics, output_path=None):
    """
    Plot the simple regret over iterations.

    Parameters:
        metrics (dict): Extracted metrics containing 'cumulative_best'.
        output_path (str): Path to save the plot. If None, display the plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        regret = metrics['cumulative_best'] - metrics['cumulative_best'][-1]
        plt.plot(regret, marker='x', linestyle='-', color='r')
        plt.title('Simple Regret Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Simple Regret')
        plt.grid(True)
        if output_path:
            plt.savefig(os.path.join(output_path, 'regret_plot.png'))
            plt.close()
            print(f"Regret plot saved to {output_path}/regret_plot.png")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting regret: {e}")

def plot_evaluations_custom(metrics, space, output_path=None):
    """
    Plot the function evaluations across the parameter space.

    Parameters:
        metrics (dict): Extracted metrics containing 'x_iters' and 'func_vals'.
        space (skopt.space.Space): The search space.
        output_path (str): Path to save the plot. If None, display the plot.
    """
    try:
        # For multi-dimensional parameter spaces, select pairs to plot
        # Here, we plot the first two parameters as an example
        if len(space.dimensions) < 2:
            print("Not enough dimensions to plot evaluations.")
            return

        df = pd.DataFrame(metrics['x_iters'], columns=[dim.name for dim in space.dimensions])
        df['func_val'] = metrics['func_vals']

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df['func_val'], cmap='viridis', marker='o')
        plt.colorbar(scatter, label='Function Value')
        plt.title('Function Evaluations in Parameter Space')
        plt.xlabel(space.dimensions[0].name)
        plt.ylabel(space.dimensions[1].name)
        plt.grid(True)
        if output_path:
            plt.savefig(os.path.join(output_path, 'evaluations_scatter_plot.png'))
            plt.close()
            print(f"Evaluations scatter plot saved to {output_path}/evaluations_scatter_plot.png")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting evaluations: {e}")

def plot_uncertainty(metrics, res, output_path=None):
    """
    Plot the uncertainty from the Gaussian Process surrogate model.

    Parameters:
        metrics (dict): Extracted metrics containing 'latest_model'.
        res (dict): The final result dictionary containing the surrogate model.
        output_path (str): Path to save the plot. If None, display the plot.
    """
    try:
        optimizer = metrics.get('latest_model', None)
        if optimizer is None:
            print("No GP model available for uncertainty plot.")
            return

        # Ensure that the latest_model is an skopt.Optimizer
        if not hasattr(res, "space"):
            print("The provided model is not an skopt.Optimizer.")
            return

        # Define a grid over the search space for plotting
        # Here, we handle 2D parameter space for visualization purposes
        if len(res.space.dimensions) < 2:
            print("Not enough dimensions to plot uncertainty.")
            return

        x0_min, x0_max = res.space.dimensions[0].bounds
        x1_min, x1_max = res.space.dimensions[1].bounds

        x0 = np.linspace(x0_min, x0_max, 100)
        x1 = np.linspace(x1_min, x1_max, 100)
        X0, X1 = np.meshgrid(x0, x1)
        X = np.column_stack([X0.ravel(), X1.ravel()])

        # Predict mean and standard deviation using the surrogate model
        y_mean, y_std = res.models[-1].predict(X, return_std=True)

        Y_mean = y_mean.reshape(X0.shape)
        Y_std = y_std.reshape(X0.shape)

        plt.figure(figsize=(12, 6))

        # Plot mean
        plt.subplot(1, 2, 1)
        contour_mean = plt.contourf(X0, X1, Y_mean, levels=20, cmap='viridis')
        plt.colorbar(contour_mean, label='Predicted Mean')
        plt.title('GP Predicted Mean')
        plt.xlabel(res.space.dimensions[0].name or "X0")
        plt.ylabel(res.space.dimensions[1].name or "X1")
        plt.grid(True)

        # Plot standard deviation
        plt.subplot(1, 2, 2)
        contour_std = plt.contourf(X0, X1, Y_std, levels=20, cmap='viridis')
        plt.colorbar(contour_std, label='Predicted Std Dev')
        plt.title('GP Predicted Uncertainty')
        plt.xlabel(res.space.dimensions[0].name or "X0")
        plt.ylabel(res.space.dimensions[1].name or "X1")
        plt.grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(os.path.join(output_path, 'gp_uncertainty_plot.png'))
            plt.close()
            print(f"GP uncertainty plot saved to {output_path}/gp_uncertainty_plot.png")
        else:
            plt.show()

    except Exception as e:
        print(f"Error plotting uncertainty: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze scikit-optimize Bayesian Optimization checkpoint files.')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to the directory containing checkpoint .pkl files.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to the directory to save plots. If not specified, plots will be displayed.')
    parser.add_argument('--penalty_threshold', type=float, default=1e5,
                        help='Threshold to consider a function evaluation as valid. Adjust based on your penalty value.')
    parser.add_argument('--skip_iterations', type=int, default=1,
                        help='Number of initial iterations to skip from the analysis.')
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    penalty_threshold = args.penalty_threshold
    skip_iterations = args.skip_iterations

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find checkpoint files
    checkpoint_files = find_checkpoint_files(checkpoint_dir, extension='.pkl')
    if not checkpoint_files:
        print(f"No checkpoint files with extension .pkl found in {checkpoint_dir}.")
        return

    for checkpoint_file in checkpoint_files:
        print(f"\nAnalyzing checkpoint file: {checkpoint_file}")
        data = load_checkpoint(checkpoint_file)
        if data is None:
            print("Skipping this checkpoint due to loading failure.")
            continue

        metrics = extract_metrics(data, penalty_threshold=penalty_threshold, skip_iterations=skip_iterations)
        if not metrics:
            print("No metrics extracted. Skipping this checkpoint.")
            continue

        print("\n--- Optimization Metrics ---")
        print(f"Number of Valid Evaluations: {metrics.get('num_valid', 0)}")
        print(f"Number of Invalid Evaluations: {metrics.get('num_invalid', 0)}")
        print(f"Best Cost: {metrics.get('best_cost', 'N/A')}")
        print(f"Best Parameters: {metrics.get('best_params', 'N/A')}")
        print(f"Mean Cost: {metrics.get('mean_cost', 'N/A')}")
        print(f"Median Cost: {metrics.get('median_cost', 'N/A')}")
        print(f"Standard Deviation of Cost: {metrics.get('std_cost', 'N/A')}")

        # Extract search space
        space = data.get('space', None)
        if space is None:
            print("No search space information found in checkpoint.")
            continue

        # Locate the final_result.pkl file relative to the checkpoint directory
        final_result_pth = os.path.join(checkpoint_dir, 'final_result.pkl')
        if not os.path.exists(final_result_pth):
            print(f"final_result.pkl not found in {checkpoint_dir}. Skipping uncertainty plot.")
            final_result = None
        else:
            try:
                with open(final_result_pth, 'rb') as f:
                    final_result = dill.load(f)
            except Exception as e:
                print(f"Error loading final_result.pkl: {e}")
                final_result = None

        # Generate and save plots
        if output_dir:
            plot_convergence_custom(metrics, output_path=output_dir)
            plot_function_values(metrics, output_path=output_dir)
            plot_kdeplot(metrics, output_path=output_dir)
            plot_improvement(metrics, output_path=output_dir)
            plot_regret(metrics, output_path=output_dir)
            plot_evaluations_custom(metrics, space, output_path=output_dir)
            if final_result:
                plot_uncertainty(metrics, final_result, output_path=output_dir)
        else:
            plot_convergence_custom(metrics)
            plot_function_values(metrics)
            plot_kdeplot(metrics)
            plot_improvement(metrics)
            plot_regret(metrics)
            plot_evaluations_custom(metrics, space)
            if final_result:
                plot_uncertainty(metrics, final_result)

        # Optionally, save the metrics to a CSV file
        if output_dir:
            metrics_df = pd.DataFrame({
                'Iteration': range(1, len(metrics['func_vals']) + 1),
                'Function Value': metrics['func_vals'],
                'Cumulative Best': metrics['cumulative_best'],
                'Is Valid': metrics['is_valid']
            })
            metrics_df.to_csv(os.path.join(output_dir, 'bo_metrics.csv'), index=False)
            print(f"Metrics saved to {output_dir}/bo_metrics.csv")


if __name__ == "__main__":
    main()
    # Example usage:
    # python skip_analysis.py --checkpoint_dir ./output/1733412039 --output_dir ./plots --skip_iterations 1
