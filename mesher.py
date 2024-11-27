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

points = []
faces = []
distances_to_edges = []

# top surface
def top_z(x, y):
    return a * x**4 + c * x**2 - f * y + g * abs(x)

# bottom surface, characterized by the `term` variable which models the z difference
def bottom_z(x, y):
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