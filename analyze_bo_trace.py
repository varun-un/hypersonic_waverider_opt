#!/usr/bin/env python3
"""
analyze_bo_trace.py

A script to parse Bayesian Optimization trace files and generate metrics and plots.

Usage:
    python analyze_bo_trace.py --trace_file trace.txt --output_dir ./plots

Parameters:
    --trace_file: Path to the Bayesian Optimization trace file.
    --output_dir: Directory where the plots and metrics will be saved. If not specified, plots will be displayed.
"""

import os
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_trace_file(trace_file_path):
    """
    Parses the Bayesian Optimization trace file.

    Parameters:
        trace_file_path (str): Path to the trace file.

    Returns:
        pd.DataFrame: DataFrame containing parsed data.
    """
    # Initialize lists to store data
    iterations = []
    parameters = []
    func_vals = []
    current_min = []
    times = []
    valid = []

    # Regular expressions to extract data
    iter_start_re = re.compile(r"Iteration No: (\d+) started\.")
    params_re = re.compile(r"Running cost function with parameters: \[(.*?)\]")
    func_val_re = re.compile(r"Function value obtained: ([\d\.\-eE]+)")
    current_min_re = re.compile(r"Current minimum: ([\d\.\-eE]+)")
    time_re = re.compile(r"Time taken: ([\d\.]+)")

    with open(trace_file_path, 'r') as file:
        lines = file.readlines()

    current_iter = None
    for line in lines:
        # Check for iteration start
        iter_start_match = iter_start_re.search(line)
        if iter_start_match:
            current_iter = int(iter_start_match.group(1))
            continue

        if current_iter is not None:
            # Extract parameters
            params_match = params_re.search(line)
            if params_match:
                params_str = params_match.group(1)
                # Split parameters by comma and convert to float
                params = [float(p.strip()) for p in params_str.split(',')]
                parameters.append(params)
                iterations.append(current_iter)
                continue

            # Extract function value
            func_val_match = func_val_re.search(line)
            if func_val_match:
                func_val = float(func_val_match.group(1))
                func_vals.append(func_val)
                # Determine validity
                is_valid = func_val < 0
                valid.append(is_valid)
                continue

            # Extract current minimum
            current_min_match = current_min_re.search(line)
            if current_min_match:
                c_min = float(current_min_match.group(1))
                current_min.append(c_min)
                continue

            # Extract time taken
            time_match = time_re.search(line)
            if time_match:
                time_taken = float(time_match.group(1))
                times.append(time_taken)
                # Reset current_iter after collecting all data
                current_iter = None
                continue

    # Create DataFrame
    data = {
        'Iteration': iterations,
        'Parameters': parameters,
        'Function_Value': func_vals,
        'Current_Minimum': current_min,
        'Time_Taken': times,
        'Valid': valid
    }

    df = pd.DataFrame(data)
    return df

def compute_metrics(df):
    """
    Computes additional metrics for the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with parsed data.

    Returns:
        pd.DataFrame: DataFrame with additional metrics.
    """
    # Sort by Iteration just in case
    df = df.sort_values('Iteration').reset_index(drop=True)

    # Compute Cumulative Best (most negative function value so far)
    df['Cumulative_Best'] = df['Function_Value'].cummin()

    # Compute Improvement (difference from the initial best)
    initial_best = df['Cumulative_Best'].iloc[0]
    df['Improvement'] = initial_best - df['Cumulative_Best']

    return df

def plot_convergence(df, output_dir=None):
    """
    Plots the convergence of the optimization.

    Parameters:
        df (pd.DataFrame): DataFrame with metrics.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['Cumulative_Best'], marker='o', linestyle='-')
    plt.title('Convergence Plot')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Best Function Value')
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'convergence_plot.png'))
        plt.close()
        print("Convergence plot saved.")
    else:
        plt.show()

def plot_improvement(df, output_dir=None):
    """
    Plots the improvement over iterations.

    Parameters:
        df (pd.DataFrame): DataFrame with metrics.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['Improvement'], marker='o', linestyle='-')
    plt.title('Improvement Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Improvement from Initial Best')
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'improvement_plot.png'))
        plt.close()
        print("Improvement plot saved.")
    else:
        plt.show()

def plot_kde(df, output_dir=None):
    """
    Plots the KDE of function evaluations.

    Parameters:
        df (pd.DataFrame): DataFrame with metrics.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['Function_Value'], shade=True)
    plt.title('KDE of Function Evaluations')
    plt.xlabel('Function Value')
    plt.ylabel('Density')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'kde_plot.png'))
        plt.close()
        print("KDE plot saved.")
    else:
        plt.show()

def plot_parameter_space(df, output_dir=None):
    """
    Plots the parameter search space (first two parameters).

    Parameters:
        df (pd.DataFrame): DataFrame with metrics.
        output_dir (str): Directory to save the plot.
    """
    if df['Parameters'].iloc[0].__len__() < 2:
        print("Not enough parameters to plot parameter space.")
        return

    param1 = [params[0] for params in df['Parameters']]
    param2 = [params[1] for params in df['Parameters']]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(param1, param2, c=df['Function_Value'], cmap='viridis', marker='o')
    plt.colorbar(scatter, label='Function Value')
    plt.title('Parameter Search Space (Param 1 vs Param 2)')
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'parameter_space_plot.png'))
        plt.close()
        print("Parameter space plot saved.")
    else:
        plt.show()

def plot_function_evaluations(df, output_dir=None):
    """
    Plots all function evaluations over iterations.

    Parameters:
        df (pd.DataFrame): DataFrame with metrics.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['Function_Value'], marker='x', linestyle='-', label='Function Value')
    plt.title('Function Evaluations Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'function_evaluations_plot.png'))
        plt.close()
        print("Function evaluations plot saved.")
    else:
        plt.show()

def save_metrics_csv(df, output_dir):
    """
    Saves the metrics to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame with metrics.
        output_dir (str): Directory to save the CSV.
    """
    csv_path = os.path.join(output_dir, 'optimization_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze Bayesian Optimization trace files.')
    parser.add_argument('--trace_file', type=str, required=True,
                        help='Path to the Bayesian Optimization trace file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the plots and metrics. If not specified, plots will be displayed.')
    args = parser.parse_args()

    trace_file = args.trace_file
    output_dir = args.output_dir

    # Check if trace file exists
    if not os.path.isfile(trace_file):
        print(f"Trace file {trace_file} does not exist.")
        return

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots and metrics will be saved to {output_dir}")

    # Parse the trace file
    print("Parsing the trace file...")
    df = parse_trace_file(trace_file)

    if df.empty:
        print("No data found in the trace file.")
        return

    # Compute additional metrics
    print("Computing additional metrics...")
    df = compute_metrics(df)

    # Generate and save plots
    print("Generating plots...")
    plot_convergence(df, output_dir)
    plot_improvement(df, output_dir)
    plot_kde(df, output_dir)
    plot_parameter_space(df, output_dir)
    plot_function_evaluations(df, output_dir)

    # Save metrics to CSV if output directory is specified
    if output_dir:
        save_metrics_csv(df, output_dir)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
    # python analyze_bo_trace.py --trace_file ./trace3.log --output_dir ./plots