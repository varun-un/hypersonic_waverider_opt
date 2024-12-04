import sys
import re
from main import get_i
from mesher import generate_mesh
import pyvista as pv

def extract_parameters(log_file_path, iteration_number):
    """
    Extracts the parameters array for a specified iteration from the log file.

    Args:
        log_file_path (str): Path to the log file.
        iteration_number (int): The iteration number to extract parameters for.

    Returns:
        list: A list of parameters if found, else None.
    """
    parameters = None
    # Regular expression to match the parameters line
    params_pattern = re.compile(r"Running cost function with parameters:\s*\[(.*?)\]")
    # Regular expression to identify the start of the desired iteration
    iteration_start_pattern = re.compile(rf"Iteration No:\s*{iteration_number}\s*started\.")

    try:
        with open(log_file_path, 'r', encoding='utf-16-be') as file:
            for line in file:
                # Check if the current line marks the start of the desired iteration
                if iteration_start_pattern.search(line):
                    # Iterate through the subsequent lines to find the parameters
                    for subsequent_line in file:
                        params_match = params_pattern.search(subsequent_line)
                        if params_match:
                            # Extract the parameters string
                            params_str = params_match.group(1)
                            # Convert the string to a list of floats
                            parameters = [float(param.strip()) for param in params_str.split(',')]
                            return parameters
                    break  # Exit if the iteration start is found but parameters are missing
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        sys.exit(1)

    return parameters

def main():
    # Ensure the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python get_mesh.py <trace.log> <iteration_number>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    try:
        iteration = int(sys.argv[2])
    except ValueError:
        print("Error: The iteration number must be an integer.")
        sys.exit(1)
    
    # Extract parameters
    parameters = extract_parameters(log_file, iteration)
    
    if parameters is not None:
        print(f"parameters = {parameters}")
    else:
        print(f"Parameters for iteration {iteration} not found.")

    a, c, g, h, j, q, s = parameters
    f = 0

    i = get_i([a, c, f, g, h, j, q, s])

    if i is None:
        print("Error: Could not calculate i.")
        sys.exit(1)

    dy = 0.01
    initial_N = 102
    filename = f"iteration_{iteration}_mesh.vtk"

    generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, filename)

    mesh = pv.read(filename)
    mesh.plot(show_edges=True, show_grid=True)

if __name__ == "__main__":
    main()

    # python get_mesh.py trace.log 6
