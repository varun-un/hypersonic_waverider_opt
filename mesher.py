"""
For the top surface:
z=a*x^4+c*x^2-f*y + g*abs(x)
with the domain:
y < -q*x^2-s*abs(x)
y > -1

For the bottom surface:
z=a*x^4+c*x^2-f*y + g*abs(x)+(y+q*x**2+s*abs(x))*(h*x**2-i*y+j)
with the domain:
y < -q*x^2-s*abs(x)
y > -1

The back surface is just a plane, and so it is:
y=-1
with the domain:
z < a*x^4+c*x^2-f*y + g*abs(x)
z > a*x^4+c*x^2-f*y + g*abs(x)+(y+q*x**2+s*abs(x))*(h*x**2-i*y+j)
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

points = []
faces = []
distances_to_edges = []

# top surface
def top_z(x, y, a=0, c=0.0, f=0.2, g=-0.8, h=1.7, i=0.45, j=0.17, q=2.5, s=0.5):
    return a * x**4 + c * x**2 - f * y + g * abs(x)

# bottom surface, characterized by the `term` variable which models the z difference
def bottom_z(x, y, a=0, c=0.0, f=0.2, g=-0.8, h=1.7, i=0.45, j=0.17, q=2.5, s=0.5):
    term = (y + q * x**2 + s * abs(x)) * (h * x**2 - i * y + j)
    return top_z(x, y) + term

# Make the approximation that the gradients of the top surface approximate the bottom faces
# You can use gradients in the meshing process to assign more cells to regions of high curvature, but
# in this model, regions of high gradients tend to be near the edges of the surface, so as long as we 
# mesh the edges well, we should be fine.
def gradient_top(x, y):
    return np.array([4 * a * x**3 + 2 * c * x - g * np.sign(x), -f])

def gradient_bottom(x, y):
    return np.array([4 * a * x**3 + 2 * c * x - g * np.sign(x) + 2*y*h*x + 4*q*h*x**3 - 2*q*i*y*x + 2*q*j*x + 3*s*h*x**2 + s*j,
                     -f + h*x**2 - 2*i*y + j - q*i*x**2 + s*i*x])


def solve_for_x(y, q, s):
    """
    Solve y = -q*x^2 - s*|x| for x >= 0.
    Returns the positive solution.
    """
    # y = -q*x^2 - s*x => q*x^2 + s*x + y = 0
    # Using quadratic formula
    discriminant = s**2 - 4*q*y
    if discriminant < 0:
        return None  # No real solution
    
    # handle abs value case and parabola case
    if q == 0:
        x = -y / s
    else:
        # quadratic formula
        x = (-s + np.sqrt(discriminant)) / (2*q)
    return x

def generate_x_points(x_max, N):
    """
    Generate N points between -x_max and x_max with higher density near x=0 and the boundaries.
    """
    if N == 1:
        return [0.0]
    
    # Generate parameter t
    t = np.linspace(0, 1, N//2 + 1)  # Half for positive x
    
    # Clustering near t=0 and t=1 (edges)
    # using cosine spacing to cluster near edges
    t_clipped = t[1:-1]  # exclude the first and last points
    theta = t_clipped * np.pi
    clustered = (1 - np.cos(theta)) / 2  # maps t in [0,1] to [0,1] with clustering near 0 and 1
    
    # combine clustered points
    positive_x = np.concatenate(([0], x_max * clustered, [x_max]))
    
    # Mirror to negative x
    negative_x = -positive_x[1:-1][::-1]
    
    # negative and positive x points
    x_points = np.concatenate((negative_x, positive_x))
    
    # Adjust in case of odd N
    if len(x_points) > N:
        x_points = x_points[:N]
    elif len(x_points) < N:
        x_points = np.append(x_points, x_max)

    # usually omits the left border point, so correct for that
    if x_points[0] != -x_max:
        x_points = np.append(-x_max, x_points)
    
    return x_points


def generate_points(q, s, dy, initial_N, y_min=-1, y_max=0, append_origin=True):
    """
    Generate a list of lists containing (x, y, z) points within the specified region.
    
    Parameters:
    - q, s: Parameters defining the boundary.
    - dy: Step size in y for each row.
    - initial_N: Number of points in the first row (y = y_min).
    - y_min: Starting y-value (default -1).
    - y_max: Ending y-value (default 0).
    - append_origin: Whether to append the origin (0, 0) to the points list.
    
    Returns:
    - points_list: List of lists, each containing (x, y) tuples for a row.
    """
    points_list = []
    current_y = y_min
    N = initial_N
    
    while current_y <= y_max and N > 0:
        x_pos = solve_for_x(current_y, q, s)
        if x_pos is None:
            break  # No more solutions
        
        # x points for this row
        x_points = generate_x_points(x_pos, N)
        
        # create into tuples
        row_points = [(x, current_y) for x in x_points]
        
        points_list.append(row_points)
        
        # next row
        current_y += dy
        N -= 1
    
    
    if append_origin:
        points_list.append([(0, 0)])

    return points_list

def main():

    a = 0
    c = 0.0
    f = 0.2
    g = -0.8
    h = 1.7
    i = 0.45
    j = 0.17
    q = 2.5
    s = 0.5

    dy = 0.05
    initial_N = 21
    y_min = -1.0
    y_max = 0.0
    
    points = generate_points(q, s, dy, initial_N, y_min, y_max)
    
    plt.figure(figsize=(10, 6))
    
    for row in points:
        xs, ys = zip(*row)
        plt.plot(xs, ys, 'o')
    
    # Plot the boundary
    x_boundary = np.linspace(-5, 5, 400)
    y_boundary = -q * x_boundary**2 - s * np.abs(x_boundary)
    plt.plot(x_boundary, y_boundary, 'r-', label='Boundary y = -q*x² - s|x|')
    plt.axhline(y=-1, color='g', linestyle='--', label='y = -1')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-1.1, 0.1)
    plt.xlim(-1, 1)
    plt.title('Generated Points Within the Specified Region')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
