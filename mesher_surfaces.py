import numpy as np
import pyvista as pv

# Define parameters (you can adjust these as needed)
a = 0
c = 0.0
f = 0.2
g = -0.8
h = 1.7
i = 0.45
j = 0.17
q = 0
s = 1.84

# Define the range for x
x_min = -1.8
x_max = 1.8
x_steps = 100
x = np.linspace(x_min, x_max, x_steps)

# Define a function to compute y upper bound based on x
def y_upper(x):
    return -q * x**2 - s * np.abs(x)

# Compute the upper y bound for each x
y_upper_vals = y_upper(x)

# Since y must be greater than -1 and less than y_upper(x),
# determine valid x where y_upper(x) > -1
valid_indices = y_upper_vals > -1
x_valid = x[valid_indices]
y_upper_valid = y_upper_vals[valid_indices]

# Create a finer grid for x and y within the valid domain
y_min = -1.0
y_max = y_upper_valid.max()

# To handle varying y_max for each x, we'll generate points for each valid x
surface1_points = []
surface2_points = []
surface3_points = []

for xi, y_upper_i in zip(x_valid, y_upper_valid):
    # Define y range for this x
    y_vals = np.linspace(y_min, y_upper_i, 50)
    
    for yi in y_vals:
        # Surface 1
        z1 = a * xi**4 + c * xi**2 - f * yi + g * np.abs(xi)
        surface1_points.append([xi, yi, z1])
        
        # Surface 2
        additional_term = (yi + q * xi**2 + s * np.abs(xi)) * (h * xi**2 - i * yi + j)
        z2 = z1 + additional_term
        surface2_points.append([xi, yi, z2])
        
        # Surface 3 (y = -1)
        if yi == y_min:
            # At y = -1, z ranges between z2 and z1
            # We'll create vertical lines from z2 to z1
            z_lower = z2
            z_upper = z1
            surface3_points.append([xi, yi, z_lower])
            surface3_points.append([xi, yi, z_upper])

# Convert lists to NumPy arrays
surface1_points = np.array(surface1_points)
surface2_points = np.array(surface2_points)
surface3_points = np.array(surface3_points)

# Create PyVista meshes
# Surface 1
mesh1 = pv.PolyData(surface1_points)
mesh1 = mesh1.delaunay_2d()

# Surface 2
mesh2 = pv.PolyData(surface2_points)
mesh2 = mesh2.delaunay_2d()

# Surface 3
# Since Surface 3 consists of vertical lines at y = -1, we'll create lines
lines = []
n_points = len(surface3_points)
for i in range(0, n_points, 2):
    if i+1 < n_points:
        lines.append([2, i, i+1])  # Each line has 2 points

# Convert lines to a flat array
lines_flat = np.hstack(lines)

# Create the PolyData for Surface 3
mesh3 = pv.PolyData(surface3_points, lines_flat)

# Visualization
plotter = pv.Plotter()

# Add Surface 1
# plotter.add_mesh(mesh1, color='red', opacity=0.6, label='Surface 1', show_edges=True)

# Add Surface 2
# plotter.add_mesh(mesh2, color='green', opacity=0.6, label='Surface 2', show_edges=True)

# Add Surface 3
plotter.add_mesh(mesh3, color='blue', line_width=2, label='Surface 3', show_edges=True)

# Add a legend
plotter.add_legend()

# Add axes and grid
plotter.show_grid()
plotter.add_axes()

# Display the plot
plotter.show(title="Finite Surfaces Visualization")
