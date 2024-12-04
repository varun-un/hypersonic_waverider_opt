"""
Use Bayesian Optimization here to refine # of evals and get probabilistic model
alternative is scipy.optimize.minimize with COBYLA
    if model changes, define volume as constraint and use SLSQP or quadratic programming
"""

import trajectory as tj
from volume import find_x_bounds, analytical_volume, back_area
from mesher import generate_mesh
import numpy as np
from skopt import gp_minimize       # BO w/ Gaussian Process & RBF
from skopt.space import Real
from skopt.callbacks import CheckpointSaver

import os
import subprocess
import time
import dill
import math

MASS = 25
INITIAL_ALT = 40000.0  # meters (40 km)
INITIAL_M = 10.0      # Mach number
GEOMETRY_LENGTH = 1.0  # meters

PENALTY = 1e10


def get_i(params):
    """
    Given the 8-D parameter vector, find the parameter i that satisfies the constraint.

    Constraint: Volume = 0.01 m**3
                Length = 1 m

    Parameters:
        params (np.array): 8-D vector of parameters. In order: [a, c, f, g, h, j, q, s]

    Returns:
        i: The i coeeficient for the difference term to the bottom surface. None if no solution exists.
    """

    a, c, f, g, h, j, q, s = params

    # break based on leading edge coefficient for defined behavior
    # integrated difference between top and bottom surfaces over x-y domain & solved constraint eqn for i
    if q == 0 and s != 0:
        try:
            i = -1 * (20*h*q - 10*h*s**2 - 100*j*s**4 + 3*s**5 + 50*j*q*s**2)/(40*q*s**2 - 75*s**4)
        except ZeroDivisionError:
            # finite bounds DNE
            return None
        
    elif q != 0:
        
        # check matlab conditions for solution
        check1 = ((j*(s - np.sqrt(s**2 + 4*q)))/(2*q) - (((h*(2*q - s**2))/10 - (j*q**2)/10)*(s - np.sqrt(s**2 + 4*q))**5)/(16*q**5) + (((h*s)/4 - (j*q*s)/4)*(s - np.sqrt(s**2 + 4*q))**4)/(8*q**4) + ((h/6 - (j*(2*q - s**2))/6)*(s - np.sqrt(s**2 + 4*q))**3)/(4*q**3) + (h*(s - np.sqrt(s**2 + 4*q))**7)/(896*q**5) - (h*s*(s - np.sqrt(s**2 + 4*q))**6)/(192*q**5) + (j*s*(s - np.sqrt(s**2 + 4*q))**2)/(4*q**2) + 1/100)/((s - np.sqrt(s**2 + 4*q))**7/(2688*q**4) - (s - np.sqrt(s**2 + 4*q))**3/(24*q**2) + (s - np.sqrt(s**2 + 4*q))/(3*q) + ((s - np.sqrt(s**2 + 4*q))**5*((q*s**2)/15 - (q*(2*q - s**2))/30 + q**2/15))/(16*q**5) + (s*(s - np.sqrt(s**2 + 4*q))**2)/(8*q**2) - (s*(s - np.sqrt(s**2 + 4*q))**6)/(384*q**4) + ((s - np.sqrt(s**2 + 4*q))**4*((s*(2*q - s**2))/24 - (q*s)/12))/(8*q**4))

        check2 = np.sqrt(s**2 + 4*q) != s

        check3 = 168*(s - np.sqrt(s**2 + 4*q))**4*((q*s**2)/15 - (q*(2*q - s**2))/30 + q**2/15) + q*(s - np.sqrt(s**2 + 4*q))**6 + 896*q**4 + 336*q**3*s*(s - np.sqrt(s**2 + 4*q)) + 336*q*(s - np.sqrt(s**2 + 4*q))**3*((s*(2*q - s**2))/24 - (q*s)/12) != 112*q**3*(s - np.sqrt(s**2 + 4*q))**2 + 7*q*s*(s - np.sqrt(s**2 + 4*q))**5

        if check1 or check2 or check3:
            return None
        
        # solve for i
        i = -1 * ((j*(s - np.sqrt(s**2 + 4*q)))/(2*q) - (((h*(2*q - s**2))/10 - (j*q**2)/10)*(s - np.sqrt(s**2 + 4*q))**5)/(16*q**5) + (((h*s)/4 - (j*q*s)/4)*(s - np.sqrt(s**2 + 4*q))**4)/(8*q**4) + ((h/6 - (j*(2*q - s**2))/6)*(s - np.sqrt(s**2 + 4*q))**3)/(4*q**3) + (h*(s - np.sqrt(s**2 + 4*q))**7)/(896*q**5) - (h*s*(s - np.sqrt(s**2 + 4*q))**6)/(192*q**5) + (j*s*(s - np.sqrt(s**2 + 4*q))**2)/(4*q**2) + 1/100)/((s - np.sqrt(s**2 + 4*q))**7/(2688*q**4) - (s - np.sqrt(s**2 + 4*q))**3/(24*q**2) + (s - np.sqrt(s**2 + 4*q))/(3*q) + ((s - np.sqrt(s**2 + 4*q))**5*((q*s**2)/15 - (q*(2*q - s**2))/30 + q**2/15))/(16*q**5) + (s*(s - np.sqrt(s**2 + 4*q))**2)/(8*q**2) - (s*(s - np.sqrt(s**2 + 4*q))**6)/(384*q**4) + ((s - np.sqrt(s**2 + 4*q))**4*((s*(2*q - s**2))/24 - (q*s)/12))/(8*q**4))
        
    else:  # q == 0 and s == 0
        return None
    
    return i

def run_cfd(vtk_filename, drag_loc = -5, lift_loc = -4):
    """
    Runs the Champs solver on the given VTK file.

    Parameters:
        vtk_filename (str): Name of the VTK file to run the solver on.
        drag_loc (int): Column index of the drag coefficient in the integrated_loads.dat file. By default, -5.
        lift_loc (int): Column index of the lift coefficient in the integrated_loads.dat file. By default, -4.

    Returns:
        tuple: (lift, drag) coefficients from the simulation.
               Returns (np.inf, np.inf) if an error occurs.
    """
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        
        # Path to the integrated_loads.dat file
        loads_file = os.path.join(parent_dir, "sim", "surf", "integrated_loads.dat")
        
        # Ensure the loads_file exists; if not, wait for it to be created
        max_wait_time = 300  # Maximum wait time in seconds (5 minutes)
        wait_interval = 5    # Interval between checks in seconds
        elapsed_time = 0

        # check if the file exists
        exists = os.path.exists(loads_file)

        print("Integrated loads file exists? ", exists)

        # Record the initial modification time of the loads_file
        if exists:
            initial_mtime = os.path.getmtime(loads_file)
        else:
            initial_mtime = -1

        
        # Submit the CFD job using sbatch
        submit_command = ["./champs+", "input.sdf"]
        print("Gonna try to run: ", str(parent_dir), str(submit_command))
        subprocess.run(submit_command, cwd=parent_dir, check=True)
        
        print("CFD job submitted. Waiting for completion...")

        
        # Wait for the loads_file to be updated by the CFD job
        while True:
            time.sleep(wait_interval)
            elapsed_time += wait_interval
            if elapsed_time >= max_wait_time:
                print("Timeout waiting for CFD job to complete.")
                return np.inf, np.inf
            if exists:
                # Check if the file has been modified
                current_mtime = os.path.getmtime(loads_file)
                if current_mtime > initial_mtime:
                    print("CFD job completed.")
                    break
            else:
                # Check if the file exists now
                exists = os.path.exists(loads_file)

        # Read the last line of the integrated_loads.dat file
        with open(loads_file, "r") as file:
            lines = file.readlines()
            if not lines:
                print("Error: integrated_loads.dat is empty.")
                return np.inf, np.inf
            index = -1
            # check for empty lines at the end of the file
            if not lines[index].strip():
                index -= 1
            last_line = lines[index].strip()
            tokens = last_line.split()
            
            if len(tokens) < max(-1 * drag_loc, -1 * lift_loc):
                print("Error: Not enough data in the last line of integrated_loads.dat.")
                return np.inf, np.inf
            
            try:
                # Extract drag and lift from the specified positions
                drag = float(tokens[drag_loc]) * 2  # Multiply by 2 to get the total drag (runs half waverider)
                lift = float(tokens[lift_loc]) * 2
                print(f"Lift: {lift}, Drag: {drag}")
                return lift, drag
            except ValueError as ve:
                print(f"Error parsing lift and drag values: {ve}")
                return np.inf, np.inf

    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        return np.inf, np.inf
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
        return np.inf, np.inf
    
def run_cfdJ(vtk_filename):
    """
    Version of run_cfd for John's system
    """
    return run_cfd(vtk_filename, -2, -1)
    

def run_cfdA(vtk_filename):
    """
    SITL test version
    """

    lift = np.random.uniform(10, 500)

    drag = np.random.uniform(250, 1000)

    print(f"Lift: {lift}, Drag: {drag}")

    return lift, drag


def cost_fcn(params, dy, initial_N, timestep = 1, filename="generated_waverider.vtk", calc_tj=False):
    """
    Takes the 8-D parameter vector and computes the cost function. 
    Uses params to find the i coefficient, creates a VTK, calls the CFD solver, and computes the cost.
    Cost is negative of trajectory travelled.

    Parameters:
        params (np.array): 8-D vector of parameters. In order: [a, c, f, g, h, j, q, s]
        dy: Step size in y for each row in the meshing.
        initial_N: Initial number of points in the meshing, in the y=-1 row.
        timestep: Timestep of the trajectory simulation, in seconds.
        filename: Name of the VTK file to save.
        calc_tj: If True, calculate the trajectory using the given parameters. Otherwise, cost function is lift/drag ratio.

    Returns:
        cost: The computed cost of the trajectory. This is the negative of the distance travelled.
              If geometry is invalid, returns a penalized cost, dependent on deviation from constrained volume.
    """

    i = get_i(params)

    a, c, f, g, h, j, q, s = params

    if i is None:

        try:
            # Let i be 0.5 if it cannot be determined
            i = 0.5
            volume = analytical_volume(h, i, j, q, s)

            # Penalize the cost based on the deviation from the constrained volume
            penalty = np.abs(volume - 0.01) * PENALTY

        except ValueError as ve:
            print(f"Error: {ve}")
            penalty = PENALTY
        
        return penalty

    valid = generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, filename)
    if not valid:
        return PENALTY

    # ------ Run CFD ------
    lift, drag = run_cfdJ(filename)
    if lift == np.inf or drag == np.inf:
        return PENALTY
    
    if not calc_tj:

        back_areaD = back_area(h, i, j, q, s)

        atm_dict = tj.get_atm(30000)

        back_pressure = back_areaD * atm_dict['pressure'] / 7

        drag -= back_pressure

        ratio = lift / drag

        # inflate ratio to make it more significant
        ratio *= 1e5

        return -1 * ratio

    # calculate reference area (area between y = -qx^2 -s|x| and y = -1)
    _, x_max = find_x_bounds(q, s)

    S = 2 * (x_max - q * x_max**3 / 3 - s * x_max**2 / 2)  # Reference area in m²


    # ------ Convert to Coefficients ------
    GAMMA = 1.4               # Ratio of specific heats for air
    R = 287.05                # Specific gas constant for air, J/(kg·K)
    MACH_NUMBER = 7.0         # CFD Mach number
    CFD_ALT = 30000.0         # meters (30 km)

    # Retrieve atmospheric properties
    atm = tj.get_atm(CFD_ALT)
    temperature = atm['temperature']        # in Kelvin
    density = atm['density']                # in kg/m³

    # speed of sound
    speed_of_sound = np.sqrt(GAMMA * R * temperature)  # in m/s

    velocity = MACH_NUMBER * speed_of_sound           # in m/s

    # dynamic pressure
    dynamic_pressure = 0.5 * density * velocity**2    # in Pascals (N/m²)

    c_l = lift / (dynamic_pressure * S)   # Dimensionless
    c_d = drag / (dynamic_pressure * S)   # Dimensionless

    b_area = back_area(h, i, j, q, s)

    cost = tj.simulate_trajectory(MASS, INITIAL_ALT, INITIAL_M, GEOMETRY_LENGTH, S, b_area, timestep, cl=c_l, cd=c_d)

    return -1 * cost

def cost_fcn_partial(x):

    f = 0

    # spacing in the y direction for meshing
    dy = 0.01
    # max number of mesh vertices in the y=-1 row
    initial_N = 102
    # timestep for trajectory simulation
    timestep = 1
    # filename to save the VTK file to
    filename = "../generated_waverider.vtk"
    # do we want to calculate the trajectory as part of the cost
    calc_tj = False

    if len(x) != 7:
        print(f"Expected 7 parameters, got {len(x)}")
        return PENALTY

    print(f"Running cost function with parameters: {x}")
    mmm = cost_fcn([x[0], x[1], f, x[2], x[3], x[4], x[5], x[6]], dy, initial_N, timestep, filename, calc_tj)

    return mmm


if __name__ == "__main__":

    a = -0.3
    c = 0.0
    f = 0
    g = 0.0
    h = 0.0
    i = 0.12
    j = 0.0
    q = 0
    s = 3

    # Define the parameter space    (exclude f and i)
    space = [
        Real(-10.0, 10.0, name='a'),
        Real(-10.0, 10.0, name='c'),
        Real(-5.0, 5.0, name='g'),
        Real(0.0, 15.0, name='h'),
        Real(0.0, 2.0, name='j'),
        Real(0.0, 10.0, name='q'),
        Real(0.0, 10.0, name='s')
    ]

    # check above for meshing parameters

    cur_path = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(cur_path, "output")
    output_folder = os.path.join(output_folder, f"{int(time.time())}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pkl_name = f"{output_folder}/bo_checkpoint.pkl"
    checkpoint_saver = CheckpointSaver(pkl_name, compress=3)

    # Run Bayesian Optimization
    result = gp_minimize(
        x0=[a, c, g, h, j, q, s],        # Initial guess
        func=cost_fcn_partial,              # Objective function to minimize
        dimensions=space,                   # Search space
        acq_func="EI",                      # Acquisition function
        n_calls=2,                         # Total number of evaluations
        n_initial_points=1,                 # Initial random evaluations
        random_state=1,                     # Seed for reproducibility
        callback=[checkpoint_saver],        # Save progress
        noise="gaussian",                   # Assume somewhat noisy observations
        verbose=True                        # Print progress
    )

    # Save the final result
    with open(f"{output_folder}/final_result.pkl", "wb") as file:
        dill.dump(result, file)
