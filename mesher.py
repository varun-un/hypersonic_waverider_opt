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