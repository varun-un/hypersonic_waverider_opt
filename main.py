"""
Use Bayesian Optimization here to refine # of evals and get probabilistic model
alternative is scipy.optimize.minimize with COBYLA
    if model changes, define volume as constraint and use SLSQP or quadratic programming
"""

import trajectory as tj
from volume import find_x_bounds
from mesher import generate_mesh
import numpy as np
import pyvista as pv
import bpy
from skopt import gp_minimize       # BO w/ Gaussian Process & RBF
from skopt.space import Real
from skopt.callbacks import CheckpointSaver


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

def run_cfd(vtk_filename):
    """
    Runs the Champs solver on the given VTK file.

    Parameters:
        vtk_filename: Name of the VTK file to run the solver on.

    Returns:
        lift: Lift coefficient from the simulation.
        drag: Drag coefficient from the simulation.
    """
    pass


def cost_fcn(params, dy, initial_N, timestep = 1, filename="generated_waverider.vtk"):
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

    Returns:
        cost: The computed cost of the trajectory. This is the negative of the distance travelled.
    """

    i = get_i(params)

    if i is None:
        return np.inf - 1

    valid = generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, filename)
    if not valid:
        return np.inf - 1

    # ------ Run CFD ------
    lift, drag = run_cfd(filename)

    cost = tj.simulate_trajectory(timestep, lift, drag)

    return -1 * cost

if __name__ == "__main__":

    a = 0
    c = 0.0
    f = 0.2
    g = -0.8
    h = 1.7
    i = 0.45
    j = 0.17
    q = 0
    s = 1.84

    # Define the parameter space
    space = [
        Real(-10.0, 10.0, name='a'),
        Real(-10.0, 10.0, name='c'),
        Real(0.0, 0.0, name='f'),           # restrict f to 0 to get 0 angle of attack
        Real(-5.0, 5.0, name='g'),
        Real(0.0, 15.0, name='h'),
        Real(0.0, 2.0, name='j'),
        Real(0.0, 10.0, name='q'),
        Real(0.0, 10.0, name='s')
    ]

    # spacing in the y direction for meshing
    dy = 0.05
    # max number of mesh vertices in the y=-1 row
    initial_N = 22
    # timestep for trajectory simulation
    timestep = 1
    # filename to save the VTK file to
    filename = "../generated_waverider.vtk"

    cost_fcn_partial = lambda x: cost_fcn(x, dy, initial_N, timestep, filename)       # partial function to pass to gp_minimize

    checkpoint_saver = CheckpointSaver("./outputs/bo_checkpoint.pkl", compress=3)

    # Run Bayesian Optimization
    result = gp_minimize(
        x0=[a, c, f, g, h, j, q, s],        # Initial guess
        func=cost_fcn_partial,              # Objective function to minimize
        dimensions=space,                   # Search space
        acq_func="EI",                      # Acquisition function
        n_calls=50,                         # Total number of evaluations
        n_initial_points=5,                 # Initial random evaluations
        random_state=1,                     # Seed for reproducibility
        callback=[checkpoint_saver],        # Save progress
        noise="gaussian",                   # Assume somewhat noisy observations
        verbose=True                        # Print progress
    )
