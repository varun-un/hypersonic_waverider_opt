import sys
import re
import os
from main import get_i
from mesher import generate_mesh
import pyvista as pv

def parse_log_file(log_file_path):
    """
    Parses the log file and extracts iteration details.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        list of dict: Each dict contains 'iteration', 'parameters', and 'function_value'.
    """
    iterations = []
    current_iteration = None
    params_pattern = re.compile(r"Running cost function with parameters:\s*\[(.*?)\]")
    func_val_pattern = re.compile(r"Function value obtained:\s*([-+]?\d*\.\d+|\d+)")
    iteration_start_pattern = re.compile(r"Iteration No:\s*(\d+)\s*started\.")
    iteration_end_pattern = re.compile(r"Iteration No:\s*(\d+)\s*ended\.")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:

                # Check for the start of an iteration
                start_match = iteration_start_pattern.search(line)
                if start_match:
                    iteration_num = int(start_match.group(1))
                    current_iteration = {
                        'iteration': iteration_num,
                        'parameters': None,
                        'function_value': None
                    }
                    continue  # Proceed to next line

                # If inside an iteration, look for parameters and function value
                if current_iteration is not None:
                    # Extract parameters
                    params_match = params_pattern.search(line)
                    if params_match:
                        params_str = params_match.group(1)
                        try:
                            parameters = [float(param.strip()) for param in params_str.split(',')]
                            current_iteration['parameters'] = parameters
                        except ValueError:
                            print(f"Warning: Could not parse parameters for iteration {current_iteration['iteration']}.")
                            current_iteration['parameters'] = None
                        continue  # Proceed to next line

                # Extract function value
                if current_iteration is not None and func_val_pattern.search(line):
                    print(line)
                    func_val_match = func_val_pattern.search(line)
                    try:
                        function_value = float(func_val_match.group(1))
                        current_iteration['function_value'] = function_value
                    except ValueError:
                        print(f"Warning: Could not parse function value for iteration {current_iteration['iteration']}.")
                        current_iteration['function_value'] = None

                    # Add the iteration to the list
                    iterations.append(current_iteration)
                    current_iteration = None

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        sys.exit(1)

    return iterations

def generate_vtks_for_valid_iterations(log_file_path, output_dir):
    """
    Generates VTK files for all valid iterations (function value negative).

    Args:
        log_file_path (str): Path to the log file.
        output_dir (str): Directory to save the VTK files.
    """
    # Parse the log file
    iterations = parse_log_file(log_file_path)
    print(f"Total iterations found: {len(iterations)}")

    # Filter valid iterations
    valid_iterations = [it for it in iterations if it['function_value'] is not None and it['function_value'] < 0]
    print(f"Valid iterations (function value negative): {len(valid_iterations)}")

    if not valid_iterations:
        print("No valid iterations found with negative function values.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error: Could not create output directory '{output_dir}'. {e}")
            sys.exit(1)
    else:
        print(f"Output directory already exists: {output_dir}")

    # Process each valid iteration
    for idx, it in enumerate(valid_iterations):
        iteration_num = it['iteration']
        parameters = it['parameters']

        if parameters is None:
            print(f"Skipping iteration {iteration_num}: Parameters not found or could not be parsed.")
            continue

        if len(parameters) != 7:
            print(f"Skipping iteration {iteration_num}: Expected 7 parameters, got {len(parameters)}.")
            continue

        # Unpack parameters
        a, c, g, h, j, q, s = parameters
        f = 0  # As per the original script

        # Compute 'i' using get_i
        try:
            i = get_i([a, c, f, g, h, j, q, s])
        except Exception as e:
            print(f"Error computing 'i' for iteration {iteration_num}: {e}")
            continue

        if i is None:
            print(f"Skipping iteration {iteration_num}: 'i' could not be calculated.")
            continue

        dy = 0.01
        initial_N = 102
        filename = f"{idx}.vtk"
        filepath = os.path.join(output_dir, filename)

        # Generate mesh
        try:
            generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, filepath)
            print(f"Generated mesh for iteration {iteration_num}: {filepath}")
        except Exception as e:
            print(f"Error generating mesh for iteration {iteration_num}: {e}")
            continue

        # Optionally, read and plot the mesh (commented out to speed up processing)
        # Uncomment the following lines if you want to visualize each mesh
        """
        try:
            mesh = pv.read(filepath)
            mesh.plot(show_edges=True, show_grid=True)
        except Exception as e:
            print(f"Error reading or plotting mesh for iteration {iteration_num}: {e}")
        """

    print("VTK generation completed.")

def main():
    # Ensure the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python generate_valid_vtks.py <trace.log> <output_directory>")
        sys.exit(1)

    log_file = sys.argv[1]
    output_dir = sys.argv[2]

    generate_vtks_for_valid_iterations(log_file, output_dir)

if __name__ == "__main__":
    main()
    # python generate_valid_vtks.py trace.log valid_vtks
