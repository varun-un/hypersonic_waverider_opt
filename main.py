"""
Use Bayesian Optimization here to refine # of evals and get probabilistic model
alternative is scipy.optimize.minimize with COBYLA
    if model changes, define volume as constraint and use SLSQP or quadratic programming
"""

import trajectory as tj
import numpy as np
import pyvista as pv
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

    # Define grid resolution
    num_x = 10  # Number of points along x
    num_y = 10  # Number of points along y

    filename = "waverider.vtk"

    # ==============================
    # 1. Parameter Definition and Domain Setup
    # ==============================

    # Compute x_max by solving q*x^2 + s*|x| = 1
    if q == 0 and s != 0:
        x_pos = 1 / s
        x_neg = -1 / s
        x_max = max(abs(x_pos), abs(x_neg))
        x_min = -x_max
    elif q != 0:
        discriminant = s**2 + 4 * q
        if discriminant < 0:
            raise ValueError("Discriminant is negative. No real solutions for x_max.")
        x_pos = (-s + np.sqrt(discriminant)) / (2 * q)
        x_neg = (s + np.sqrt(discriminant)) / (2 * q)
        x_max = max(abs(x_pos), abs(x_neg))
        x_min = -x_max
    else:
        raise ValueError("Both q and s cannot be zero, leading to non-finite bounds.")

    # Define x array
    x = np.linspace(x_min, x_max, num_x)

    # Compute Y_max for all x to determine y_min
    Y_max = 0
    y_min = -1
    dy = (0 - y_min) / (num_y - 1)  # Step size
    y_max = Y_max + dy  # Slightly above -1 to include points where Y > -1

    # Define y array from y_min to y_max
    y = np.linspace(y_min, y_max, num_y)

    # Create meshgrid for x and y
    X, Y = np.meshgrid(x, y)

    # Compute mask: y <= Y_max(x) and y > -1
    mask = (Y <= -q * X**2 - s * np.abs(X)) & (Y > -1)

    # Debug statements
    print(f"x_min: {x_min}, x_max: {x_max}")
    print(f"y_min: {y_min}, y_max: {y_max}")
    print(f"Number of masked points: {np.sum(mask)}")

    # ==============================
    # 2. Point Generation for Top and Bottom Surfaces
    # ==============================

    # Compute Z for top and bottom surfaces
    Z_top = a * X**4 + c * X**2 - f * Y + g * np.abs(X)
    Z_bottom = Z_top + (Y + q * X**2 + s * np.abs(X)) * (h * X**2 - i * Y + j)

    # Initialize lists to collect points and their indices
    points = []
    point_indices = -np.ones((num_y, num_x, 2), dtype=int)  # [i,j,0] top, [i,j,1] bottom

    current_index = 0
    for i in range(num_y):
        for j in range(num_x):
            if mask[i, j]:

                print(f"({X[i, j]:.4f}, {Y[i, j]:.4f}, {Z_top[i, j]:.4f})")
                # Top point
                points.append([X[i, j], Y[i, j], Z_top[i, j]])
                point_indices[i, j, 0] = current_index
                current_index += 1
                # Bottom point
                points.append([X[i, j], Y[i, j], Z_bottom[i, j]])
                point_indices[i, j, 1] = current_index
                current_index += 1

    if not points:
        raise ValueError("No points generated. Check parameter ranges and mask conditions.")

    # Convert points to NumPy array
    points = np.array(points)

    # ==============================
    # 3. Point Generation for the Back Surface
    # ==============================

    # At y = -1, define back surface points
    z_back_top = a * x**4 + c * x**2 - f * (-1) + g * np.abs(x)
    z_back_bottom = z_back_top + ((-1) + q * x**2 + s * np.abs(x)) * (h * x**2 - i * (-1) + j)

    # Create back surface points (two points per x: top and bottom)
    back_points = np.column_stack((x, np.full_like(x, -1), z_back_top)).tolist()
    back_points += np.column_stack((x, np.full_like(x, -1), z_back_bottom)).tolist()

    back_points = np.array(back_points)

    # Append back_points to points
    all_points = np.vstack([points, back_points])

    # Assign indices to back surface points
    back_start_idx = len(points)
    back_indices_top = np.arange(back_start_idx, back_start_idx + num_x)
    back_indices_bottom = np.arange(back_start_idx + num_x, back_start_idx + 2 * num_x)

    # ==============================
    # 4. Connectivity and Face Generation
    # ==============================

    faces = []

    # Helper function to add a triangle to faces
    def add_triangle(p1, p2, p3):
        return [3, p1, p2, p3]

    # Top and Bottom Surface Faces
    for i in range(num_y - 1):
        for j in range(num_x - 1):
            if mask[i, j] and mask[i, j + 1] and mask[i + 1, j] and mask[i + 1, j + 1]:
                # Indices for top and bottom points
                idx_top_current = point_indices[i, j, 0]
                idx_bottom_current = point_indices[i, j, 1]
                idx_top_right = point_indices[i, j + 1, 0]
                idx_bottom_right = point_indices[i, j + 1, 1]
                idx_top_next = point_indices[i + 1, j, 0]
                idx_bottom_next = point_indices[i + 1, j, 1]

                # Ensure indices are valid
                if -1 not in [idx_top_current, idx_bottom_current, idx_top_right, idx_bottom_right, idx_top_next, idx_bottom_next]:
                    # Create two triangles for each quad cell
                    faces.append(add_triangle(idx_top_current, idx_top_right, idx_bottom_right))
                    faces.append(add_triangle(idx_top_current, idx_bottom_right, idx_bottom_current))
                    faces.append(add_triangle(idx_top_current, idx_bottom_current, idx_bottom_next))
                    faces.append(add_triangle(idx_top_current, idx_bottom_next, idx_top_next))

    # Back Surface Faces
    for j in range(num_x - 1):
        # Indices for back surface triangles
        idx_back_top_current = back_indices_top[j]
        idx_back_top_next = back_indices_top[j + 1]
        idx_back_bottom_current = back_indices_bottom[j]
        idx_back_bottom_next = back_indices_bottom[j + 1]

        # Create two triangles for each quad cell at the back
        faces.append(add_triangle(idx_back_top_current, idx_back_top_next, idx_back_bottom_next))
        faces.append(add_triangle(idx_back_top_current, idx_back_bottom_next, idx_back_bottom_current))

    # Side Surfaces (Left and Right)
    for i in range(num_y - 1):
        # Left Side (x_min)
        j = 0
        if mask[i, j] and mask[i + 1, j]:
            idx_top_current = point_indices[i, j, 0]
            idx_bottom_current = point_indices[i, j, 1]
            idx_top_next = point_indices[i + 1, j, 0]
            idx_bottom_next = point_indices[i + 1, j, 1]

            if -1 not in [idx_top_current, idx_bottom_current, idx_top_next, idx_bottom_next]:
                faces.append(add_triangle(idx_top_current, idx_top_next, idx_bottom_next))
                faces.append(add_triangle(idx_top_current, idx_bottom_next, idx_bottom_current))

        # Right Side (x_max)
        j = num_x - 1
        if mask[i, j] and mask[i + 1, j]:
            idx_top_current = point_indices[i, j, 0]
            idx_bottom_current = point_indices[i, j, 1]
            idx_top_next = point_indices[i + 1, j, 0]
            idx_bottom_next = point_indices[i + 1, j, 1]

            if -1 not in [idx_top_current, idx_bottom_current, idx_top_next, idx_bottom_next]:
                faces.append(add_triangle(idx_top_current, idx_bottom_current, idx_bottom_next))
                faces.append(add_triangle(idx_top_current, idx_bottom_next, idx_top_next))

    # ==============================
    # 5. Mesh Assembly and Validation
    # ==============================

    if not faces:
        raise ValueError("No faces generated. Check connectivity conditions.")

    # Convert faces list to a flat array
    faces_flat = np.hstack(faces)

    # Create the PolyData mesh
    mesh = pv.PolyData(all_points, faces_flat)

    # Check mesh integrity
    if not mesh.is_manifold:
        print("Warning: The mesh is not manifold. Some edges may not be properly connected.")
    else:
        print("Mesh is manifold.")

    # Clean the mesh to remove any duplicate points or unused points
    mesh = mesh.clean()

    # ==============================
    # 6. Visualization and Export
    # ==============================

    # Optional: Visualize the mesh to verify
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color='lightblue')
    plotter.add_axes()
    plotter.show()

    # Export the mesh to a VTK file
    mesh.save(filename)
    print(f"VTK mesh '{filename}' has been successfully created.")



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