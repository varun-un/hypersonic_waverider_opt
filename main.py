"""
Use Bayesian Optimization here to refine # of evals and get probabilistic model
alternative is scipy.optimize.minimize with COBYLA
    if model changes, define volume as constraint and use SLSQP or quadratic programming
"""

import trajectory as tj
from volume import find_x_bounds
import numpy as np
import pyvista as pv
import bpy
from skopt import gp_minimize       # BO w/ Gaussian Process & RBF (allegedly better than Random Forest)

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

def generate_vtk(geometry_params):
    """
    Given the 9-D parameter vector, generate a VTK file for the solid defined by the parameters.

    Parameters:
        geometry_params (np.array): 9D vector of parameters. In order: [a, c, f, g, h, i, j, q, s]
    """
    a, c, f, g, h, i, j, q, s = geometry_params

    X_POINTS = 10
    Y_POINTS = 10

    # Top surface function
    def top_surface(x, y):
        return a * x**4 + c * x**2 - f * y + g * np.abs(x)

    # Bottom surface function
    def bottom_surface(x, y):
        term1 = a * x**4 + c * x**2 - f * y + g * np.abs(x)
        term2 = (y + q * x**2 + s * np.abs(x)) * (h * x**2 - i * y + j)
        return term1 + term2

    # Domain functions
    def domain_boundary(x):
        return -q * x**2 - s * np.abs(x)
    
    # Define the range of x values for the domain
    x_min, x_max = find_x_bounds(q, s)
    num_points = 10  # Number of points along the edge (set N as needed)
    x_edge = np.linspace(x_min, x_max, num_points)

    # Compute y values along the intersection edge y = -q*x^2 - s*abs(x)
    y_edge = domain_boundary(x_edge)

    # Compute z values on the top and bottom surfaces along the edge
    z_top_edge = top_surface(x_edge, y_edge)
    z_bottom_edge = bottom_surface(x_edge, y_edge)

    y_back = -1
    x_back = np.linspace(x_min, x_max, num_points)

    # Compute z values on the top and bottom surfaces at y = -1
    z_top_back = top_surface(x_back, y_back)
    z_bottom_back = bottom_surface(x_back, y_back)

    # Create mesh grids for x and y within the domain
    x_vals = np.linspace(x_min, x_max, X_POINTS)  # Number of x points
    y_vals = np.linspace(y_back, y_edge.min(), Y_POINTS)  # Number of y points

    print(x_vals)
    print(y_vals)

    X, Y = np.meshgrid(x_vals, y_vals)

    # Flatten the arrays for processing
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Compute Z values for the top and bottom surfaces
    Z_top = top_surface(X_flat, Y_flat)
    Z_bottom = bottom_surface(X_flat, Y_flat)

    # Stack the coordinates
    points_top = np.vstack((X_flat, Y_flat, Z_top)).T
    points_bottom = np.vstack((X_flat, Y_flat, Z_bottom)).T

    # Determine the number of cells in x and y directions
    n_cells_x = X_POINTS - 1
    n_cells_y = Y_POINTS - 1
    n_cells = n_cells_x * n_cells_y

    # Initialize lists to hold cell definitions
    faces = []

    # Build faces for the top surface
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            idx = i * X_POINTS + j
            face = [4,  # Number of points in the face (quad)
                    idx,
                    idx + 1,
                    idx + X_POINTS + 1,
                    idx + X_POINTS]
            faces.extend(face)

    # Convert to numpy array
    faces = np.array(faces)


    # Create PyVista mesh for the top surface
    mesh_top = pv.PolyData()
    mesh_top.points = points_top
    mesh_top.faces = faces

    # Similarly for the bottom surface
    mesh_bottom = pv.PolyData()
    mesh_bottom.points = points_bottom
    mesh_bottom.faces = faces

    # Combine top and bottom meshes
    combined_mesh = mesh_top.merge(mesh_bottom)

    # Add the back surface (plane at y = -1)
    back_plane_points = np.vstack((x_back, y_back * np.ones_like(x_back), z_top_back)).T
    back_plane = pv.PolyData(back_plane_points)

    # Assuming the back plane is connected appropriately
    combined_mesh = combined_mesh.merge(back_plane)

    # Perform mesh cleaning operations if necessary
    combined_mesh.clean(inplace=True)

    combined_mesh.plot(show_edges=True)

    # Save the mesh
    combined_mesh.save('waverider.vtk')








def cost_fcn(params):
    """
    Takes the 8-D parameter vector and computes the cost function. 
    Uses params to find the i coefficient, creates a VTK, calls the CFD solver, and computes the cost.
    Cost is negative of trajectory travelled.

    Parameters:
        params (np.array): 8-D vector of parameters. In order: [a, c, f, g, h, j, q, s]
    """

    i = get_i(params)

    if i is None:
        return np.inf - 1

    #-------------------------------------

    return cost

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

    generate_vtk([a, c, f, g, h, i, j, q, s])