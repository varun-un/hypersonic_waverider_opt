import os
import time
import dill
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import CheckpointSaver

from main import cost_fcn_partial, get_i
import trajectory as tj
import tj_opt
from volume import find_x_bounds, analytical_volume, back_area
from mesher import generate_mesh

import sys

# Define the same parameter space as before
space = [
    Real(-5.0, 5.0, name='a'),
    Real(-4.0, 4.0, name='c'),
    Real(-2.0, 2.0, name='g'),
    Real(0.0, 5.0, name='h'),
    Real(0.0, 1.0, name='j'),
    Real(0.0, 15.0, name='q'),  # non-zero to avoid q=0, s=0 singularity
    Real(0.0, 6.0, name='s')
]

# get command line argument
output_num = sys.argv[1]

# Path to the existing checkpoint
existing_checkpoint_path = f"./output/{output_num}/bo_checkpoint.pkl"

# Load the existing checkpoint
with open(existing_checkpoint_path, "rb") as f:
    loaded_result = dill.load(f)

# Extract previously evaluated points and their function values
x0 = loaded_result.x_iters
y0 = loaded_result.func_vals

# Ensure the output folder for the continued run
cur_path = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(cur_path, "output")
output_folder = os.path.join(output_folder, f"{output_num}_continued")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the new checkpoint saver for the continued run
new_pkl_name = f"{output_folder}/bo_checkpoint_continued.pkl"
checkpoint_saver = CheckpointSaver(new_pkl_name, compress=3)

# Define the total desired iterations
total_iterations = 1500
additional_iterations = total_iterations - len(y0)

if additional_iterations <= 0:
    print(f"Optimization already has {len(y0)} iterations, which is >= {total_iterations}.")
else:
    # Run Bayesian Optimization with the additional iterations
    result = gp_minimize(
        func=cost_fcn_partial,      # Your objective function
        dimensions=space,           # Search space
        acq_func="EI",              # Acquisition function
        n_calls=additional_iterations,  # Additional evaluations needed
        x0=x0,                      # Previously evaluated points
        y0=y0,                      # Previously evaluated function values
        random_state=2,             # Seed for reproducibility
        callback=[checkpoint_saver],# Save progress
        noise="gaussian",           # Assume somewhat noisy observations
        verbose=True                # Print progress
    )

    # Optionally, save the final result
    final_result_path = os.path.join(output_folder, "final_result.pkl")
    with open(final_result_path, "wb") as f:
        dill.dump(result, f)

    print(f"Continued optimization completed. Results saved to {final_result_path}")

    # do trajectory opt + sweep + stuff


    best_params = result.x

    print(f"Best parameters: {best_params}")

    print("RESULTS OBJ:")
    print(result)

    # Parameters for the final best shape evaluation
    dy = 0.002
    initial_N = 502
    filename = "../generated_waverider.vtk"

    # generate the best mesh
    a, c, g, h, j, q, s = best_params
    i = get_i(best_params)

    generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, filename)

    # also save it to the output folder
    best_vtk = f"{output_folder}/best_waverider.vtk"
    generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, best_vtk)

    # do CFD sweep
    mach_range = (2, 10)
    mach_step = 0.5
    angle_range = (-10, 10)
    angle_step = 0.5

    try:
        print("Running CFD sweep...")
        df = tj_opt.sweep_cfd(mach_range, mach_step, angle_range, angle_step)
    except Exception as e:
        print(f"Error running CFD sweep: {e}")
        df = None

    # save the results
    try:
        df.to_csv(f"{output_folder}/cfd_sweep_results.csv", index=False)
    except Exception as e:
        print(f"Error saving CFD sweep results: {e}")

    A_b = back_area(h, i, j, q, s)

    # https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    try:
        result = tj_opt.optimize_aoa(df, A_b)
    except Exception as e:
        print(f"Error optimizing AOA: {e}")
        result = None

    # save the results to a pickle file
    try:
        with open(f"{output_folder}/aoa_opt_results.pkl", "wb") as file:
            dill.dump(result, file)
    except Exception as e:
        print(f"Error saving AOA optimization results: {e}")

    try:
        if result.success:
            print("Optimization successful!")
        else:
            print("Optimization failed.")
    except Exception as e:
        print(f"Error checking optimization success: {e}")

    # print the results
    print(result)

    # python continue_main.py 1733188247