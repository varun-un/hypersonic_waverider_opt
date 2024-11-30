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

# top surface
def top_z(x, y, a=0, c=0.0, f=0, g=-0.8, h=1.7, i=0.45, j=0.17, q=2.5, s=0.5):
    try:
        return a * x**4 + c * x**2 - f * y + g * abs(x)
    except Exception as e:
        print(x, y)
        raise e

# bottom surface, characterized by the `term` variable which models the z difference
def bottom_z(x, y, a=0, c=0.0, f=0, g=-0.8, h=1.7, i=0.45, j=0.17, q=2.5, s=0.5):
    term = (y + q * x**2 + s * abs(x)) * (h * x**2 - i * y + j)
    return top_z(x, y, a, c, f, g, h, i, j, q, s) + term

# Make the approximation that the gradients of the top surface approximate the bottom faces
# You can use gradients in the meshing process to assign more cells to regions of high curvature, but
# in this model, regions of high gradients tend to be near the edges of the surface, so as long as we 
# mesh the edges well, we should be fine.
def gradient_top(x, y, a=0, c=0.0, f=0.2, g=-0.8, h=1.7, i=0.45, j=0.17, q=2.5, s=0.5):
    return np.array([4 * a * x**3 + 2 * c * x - g * np.sign(x), -f])

def gradient_bottom(x, y, a=0, c=0.0, f=0.2, g=-0.8, h=1.7, i=0.45, j=0.17, q=2.5, s=0.5):
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

    # check to right duplicates
    if x_points[-1] == x_points[-2]:
        x_points = x_points[:-1]
    
    return x_points


def generate_points(q, s, dy, initial_N, y_min=-1, y_max=0):
    """
    Generate a list of lists containing (x, y) points within the specified region.
    
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

    if initial_N > (1 / dy) + 1:        # add origin to close the last row
        append_origin = True
    else:
        append_origin = False
    
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


    # first row has two last points, so fix that
    if initial_N % 2 == 1:
        points_list[0] = points_list[0][:-1]


    return points_list

def get_distances_to_edges(points, q, s):
    """
    Get the distances to the edges for each point in the points list.
    
    Parameters:
    - points: List of lists containing (x, y) tuples.
    - q, s: Parameters defining the boundary.
    
    Returns:
    - distances: List of lists containing the distance to the edge for each point.
    """
    distances = []
    for row in points:
        row_distances = []
        for point in row:
            x, y = point
            distance = y + q * x**2 + s * abs(x)        # use just straight y-distance as an approximation
            row_distances.append(distance)
        distances.append(row_distances)
    return distances

def evaluate_points(points, z_func, a, c, f, g, h, i, j, q, s):
    """
    Evaluate the z-values of the points using surface functions.
    
    Parameters:
    - points: List of lists containing (x, y) tuples.
    - z_func: Function for the surface. Takes (x, y, a, c, f, g, h, i, j, q, s) as arguments.
    - a, c, f, g, h, i, j, q, s: Parameters for the surface.

    Returns:
    - z_values: List of lists containing (x, y, z) tuples.
    """
    return [[(*point, z_func(*point, a, c, f, g, h, i, j, q, s)) for point in row] for row in points]

def generate_faces(points, a, c, f, g, h, i, j, q, s):
    """
    Generate faces from the points.
    
    Parameters:
    - points: List of lists containing (x, y) tuples.
    - a, c, f, g, h, i, j, q, s: Parameters for the surface.

    Returns:
    - faces: List of lists containing indices of points for each face. The fourth entry is the distance to the edge.
    - total_points: Total number of points in the mesh on one side
    - row_lengths: List of indices for each row in the points list.
    """

    row_lengths = [0]
    current_num_points = len(points[0])
    for row in points[1:]:
        row_lengths.append(current_num_points)
        current_num_points += len(row)

    total_points = row_lengths[-1] + len(points[-1])

    faces = []

    distances = get_distances_to_edges(points, q, s)

    # mesh the top surface first
    for row_idx, row in enumerate(points[:-1]):         # loop over all but the last row (origin)

        i = 0       # this row
        j = 0       # next row
        while (i < len(row) - 1 and len(points[row_idx + 1]) < len(row)) or (i <= len(row) - 1 and j < len(points[row_idx + 1]) - 1 and len(points[row_idx + 1]) == len(row)):

            # if we are referring vertical points, or there are no more points in the next row, we create right-moving triangles
            if j >= i or j >= len(points[row_idx + 1]) - 1:
                # create a triangualr face with this point, the next row's point, and the next point in this row
                cur = row_lengths[row_idx] + i
                above = row_lengths[row_idx + 1] + j
                right = cur + 1

                faces.append((above, cur, right, distances[row_idx][i]))

                i += 1

            else:       # left-moving triangles
                # create a triangualr face with this point, the next row's point, and the next point in the next row
                cur = row_lengths[row_idx] + i
                above_left = row_lengths[row_idx + 1] + j
                above = above_left + 1

                faces.append((above_left, cur, above, distances[row_idx][i]))

                j += 1      # okay to increment, since right-moving triangle condition is >=

    # mesh the bottom surface
    # we will use the same logic as the top surface, but with the bottom surface function
    # point indicies will be offset by the number of points in the top surface, while also accounting for boundary points
    for row_idx, row in enumerate(points[:-1]):         # loop over all but the last row (origin)

        i = 0       # this row
        j = 0       # next row
        while (i < len(row) - 1 and len(points[row_idx + 1]) < len(row)) or (i <= len(row) - 1 and j < len(points[row_idx + 1]) - 1 and len(points[row_idx + 1]) == len(row)):
            # if we are referring vertical points, or there are no more points in the next row
            # right facing triangles
            if j >= i or j >= len(points[row_idx + 1]) - 1:
                # create a triangular face with this point, the next row's point, and the next point in this row

                if i == 0 or i == len(row) - 1:  # boundary point
                    cur = row_lengths[row_idx] + i
                else:
                    prev_boundaries = 2 * (row_idx + 1) - 1
                    cur = row_lengths[row_idx] + i + total_points - prev_boundaries

                if j == 0 or j == len(points[row_idx + 1]) - 1:  # boundary point
                    above = row_lengths[row_idx + 1] + j
                else:
                    prev_boundaries = 2 * (row_idx + 2) - 1
                    above = row_lengths[row_idx + 1] + j + total_points - prev_boundaries

                if i >= len(row) - 2:
                    right = row_lengths[row_idx] + i + 1         # right point is boundary point
                else:
                    if i == 0:      # cur is a boundary point
                        prev_boundaries = 2 * (row_idx + 1) - 1
                        right= cur + 1 + total_points - prev_boundaries
                    else:
                        right = cur + 1

                faces.append((above, cur, right, distances[row_idx][i]))

                i += 1

            else:       # left-moving triangles
                # create a triangular face with this point, the next row's point, and the next point in the next row

                if i == 0 or i == len(row) - 1:  # boundary point
                    cur = row_lengths[row_idx] + i
                else:
                    prev_boundaries = 2 * (row_idx + 1) - 1
                    cur = row_lengths[row_idx] + i + total_points - prev_boundaries

                if j == 0 or j == len(points[row_idx + 1]) - 1:  # boundary point
                    above_left = row_lengths[row_idx + 1] + j
                else:
                    prev_boundaries = 2 * (row_idx + 2) - 1
                    above_left = row_lengths[row_idx + 1] + j + total_points - prev_boundaries

                if j == len(points[row_idx + 1]) - 2:
                    above = row_lengths[row_idx + 1] + j + 1      # above point is boundary point
                else:
                    if j == 0 or j == len(points[row_idx + 1]) - 1:     # above left is a boundary point
                        prev_boundaries = 2 * (row_idx + 2) - 1
                        above = above_left + 1 + total_points - prev_boundaries
                    else:
                        above = above_left + 1

                faces.append((above_left, cur, above, distances[row_idx][i]))

                j += 1      # okay to increment, since right-moving triangle condition is >=

    # mesh the back surface
    # back surface is just a plane, connecting the first row for top and bottom surfaces
    jdx = 0
    idx = 0
    x_max = solve_for_x(-1, q, s)
    while idx < len(points[0]) - 1:

        # get distance
        dist = x_max - np.abs(points[0][idx][0])

        if idx == 0 and jdx == 0:       # left corner
            faces.append((idx + 1, idx, jdx + total_points, dist))
            idx += 1
            jdx += 1
        elif idx == len(points[0]) - 2 and jdx == len(points[0]) - 2:        # right corner
            faces.append((jdx + total_points - 1, idx, idx + 1, dist))
            idx += 1
            jdx += 1
            break
        elif idx == jdx:        # right up facing triangle
            faces.append((jdx + total_points, idx, jdx + total_points - 1, dist))
            jdx += 1
        elif jdx > idx:         # left down facing triangle
            faces.append((jdx + total_points - 1, idx, idx + 1, dist))
            idx += 1

    return faces, total_points, row_lengths

def write_file(points, faces, total_points, row_lengths, filename, a, c, f, g, h, i, j, q, s, plot_points=False):
    """
    Write the points and faces to a VTK file.

    Parameters:
    - points: List of lists containing (x, y) tuples. Surface agnostic.
    - faces: List of lists containing indices of points for each face. Contains top, bottom and back surfaces.
    - total_points: Total number of points in the mesh on the top side
    - row_lengths: List of indices for each row in the points list.
    - filename: Name of the VTK file to write to.
    - a, c, f, g, h, i, j, q, s: Parameters for the surface.
    - priority_scale: List of two integers, the first being the lowest priority and the second being the highest priority assigned to cells.
    """

    f_param = f

    # setup file
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # for top and bottom surfaces, but remove boundary points for back surface
        abs_tot_points = total_points * 2 - 2 * (len(row_lengths) - 1) - 1      # minus 1 at end bc origin row only has 1 boundary

        # write points
        f.write(f"POINTS {abs_tot_points} float\n")
        ps = []
        pst = []
        psb = []
        for row in points:
            for point in row:
                z = top_z(point[0], point[1], a, c, f_param, g, h, i, j, q, s)
                f.write(f"{point[0]} {-1 * point[1]} {z}\n")

                if plot_points:
                    pst.append([point[0], point[1], z])

        # write back surface
        for row in points:
            # print("ROW: ", row)
            for point in row[1:-1]:     # exclude boundary points
                z = bottom_z(point[0], point[1], a, c, f_param, g, h, i, j, q, s)
                f.write(f"{float(point[0])} {float(-1 * point[1])} {float(z)}\n")
            
                if plot_points:
                    psb.append([point[0], point[1], z])

        if plot_points:
            # plot the points in 3D with matplotlib
            pst = np.array(pst)
            psb = np.array(psb)
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pst[:, 0], pst[:, 1], pst[:, 2], c='r', marker='o')
            ax.scatter(psb[:, 0], psb[:, 1], psb[:, 2], c='g', marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

        # write faces
        f.write(f"POLYGONS {len(faces)} {len(faces) * 4}\n")
        for face in faces:
            f.write(f"3 {face[1]} {face[0]} {face[2]}\n")

        # write face priorities
        f.write(f"CELL_DATA {len(faces)}\n")
        f.write("SCALARS Components int\n")
        f.write("LOOKUP_TABLE default\n")

        for face in faces:
            # just use the first point in the face to determine priority
            distance = face[3]

            # find the priority
            if distance < 0.01:
                priority = 3
            else:
                priority = 1

            f.write(f"           {priority}\n")

def generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, filename):
    """
    Generate a full hypersonic waverider mesh given the surface parameters and meshing parameters.

    Parameters:
    - a, c, f, g, h, i, j, q, s: Parameters for the surface.
    - dy: Step size in y for each row.
    - initial_N: Number of points in the first row (y = -1).
    - filename: Name of the VTK file to write to.

    Returns:
    - success (boolean): Whether the mesh was generated successfully.
    """

    try:
        points = generate_points(q, s, dy, initial_N)
        faces, total_points, row_lengths = generate_faces(points, a, c, f, g, h, i, j, q, s)
        write_file(points, faces, total_points, row_lengths, filename, a, c, f, g, h, i, j, q, s)
        return True
    except Exception as e:
        print(e)
        return False
        

def main():

    a = 0
    c = 0.0
    f = 0
    g = -0.8
    h = 1.7
    i = 0.45
    j = 0.17
    q = 2.5
    s = 0.5

    dy = 0.02
    initial_N = 52      # usually >= 1/dy + 2
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

    faces, total_points, row_lengths = generate_faces(points, a, c, f, g, h, i, j, q, s)

    write_file(points, faces, total_points, row_lengths, "output.vtk", a, c, f, g, h, i, j, q, s, plot_points=True)

    # delayed import to avoid bloating opt script
    import pyvista as pv

    mesh = pv.read("output.vtk")
    mesh.plot_normals(mag=0.1, color='black')

if __name__ == "__main__":
    main()
