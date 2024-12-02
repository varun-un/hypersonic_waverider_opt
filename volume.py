import numpy as np
from scipy.integrate import dblquad
import math

# Define the parameters
# Replace these values with the actual coefficients
a = 0
c = 0.0
f = 0.2
g = -0.8
h = 1.7
i = 0.45
j = 0.17
q = 2.5
s = 1.84

# top surface
def top_z(x, y):
    return a * x**4 + c * x**2 - f * y + g * abs(x)

# bottom surface, characterized by the `term` variable which models the z difference
def bottom_z(x, y):
    term = (y + q * x**2 + s * abs(x)) * (h * x**2 - i * y + j)
    return top_z(x, y) + term

# same as `term` variable, but idk could just leave it as is for now in case it changes
def integrand(y, x):
    return top_z(x, y) - bottom_z(x, y)

# find x bounds by sovling qx^2 + s|x| = 1
def find_x_bounds(q, s):
    """
    Finds the bounds for x by solving qx^2 + s |x| = 1.

    Returns:
        (x_neg, x_pos): Tuple containing the lower and upper bounds for x.

    Raises:
        ValueError: If no real solutions exist or if parameters lead to undefined bounds.
    """
    # Case 1: q != 0
    if q != 0:
        discriminant_pos = s**2 + 4 * q  # Always >=0 since q and s are real numbers
        sqrt_discriminant = math.sqrt(discriminant_pos)
        
        # x >= 0: qx^2 + sx -1 =0
        x_pos = (-s + sqrt_discriminant) / (2 * q)
        
        # x < 0: qx^2 - sx -1 =0
        x_neg = -1 * x_pos
        
        if x_pos < 0 or x_neg > 0:
            raise ValueError("Invalid bounds computed. Check parameter values for q and s.")
        
        return x_neg, x_pos

    # Case 2: q == 0 and s != 0
    elif s != 0:
        # reduces to s|x| = 1
        x_pos = 1 / s
        x_neg = -1 / s
        return x_neg, x_pos

    else:
        raise ValueError("Both q and s cannot be zero, non-finite bounds.")
    
def back_area(h, i, j, q, s):
    """
    Analytically computes the area of the back surface of the solid defined by the given parameters.
    """

    _, x_max = find_x_bounds(q, s)

    area = -(((30 * 0**2 - 30 * x_max**2) * s + (20 * 0**3 - 20 * x_max**3) * q - 60 * 0 + 60 * x_max) * i + ((30 * 0**2 - 30 * x_max**2) * j + (15 * 0**4 - 15 * x_max**4) * h) * s + ((20 * 0**3 - 20 * x_max**3) * j + (12 * 0**5 - 12 * x_max**5) * h) * q + (60 * x_max - 60 * 0) * j + (20 * x_max**3 - 20 * 0**3) * h) / 30

    return area


# Get y as a function of x
def y_lower(x):
    return -1

def y_upper(x):
    return -q * x**2 - s * abs(x)

# note - ensure y_upper > y_lower for all x in [x_min, x_max]



def analytical_volume(h, i, j, q, s):
    """
    Analytically computes the volume of the solid defined by the given parameters.

    Args:
        h, i, j, q, s: Coefficients defining the solid.

    Returns:
        volume: The computed volume of the solid.
    """

    if q == 0 and s == 0:
        raise ValueError("Both q and s cannot be zero, non-finite volume.")

    if q == 0:
        V = -1 * (2*((- i/8 - j/6)*s**4 + ((i*q)/15 - h/60 + (j*q)/12)*s**2 + (h*q)/30))/s**5
    else:
        V = (((i*q**2*s)/36 + (q*s*(3*h + i*q))/18)*(s - np.sqrt(s**2 + 4*q))**6)/(32*q**6) + (((i*s)/12 - (s*(2*i + 3*j))/6)*(s - np.sqrt(s**2 + 4*q))**2)/(2*q**2) - ((3*h + i*q)*(s - np.sqrt(s**2 + 4*q))**7)/(2688*q**5) - ((s - np.sqrt(s**2 + 4*q))**4*((s*(3*h + i*q))/12 - (q*s*(2*i + 3*j))/12 + (i*s*(2*q - s**2))/24))/(8*q**4) - ((s - np.sqrt(s**2 + 4*q))*(i/3 + j/2))/q - ((s - np.sqrt(s**2 + 4*q))**3*(h/6 + (i*q)/18 - (i*s**2)/9 - ((2*q - s**2)*(2*i + 3*j))/18))/(4*q**3) - ((s - np.sqrt(s**2 + 4*q))**5*((q**2*(2*i + 3*j))/30 - ((3*h + i*q)*(2*q - s**2))/30 + (i*q*s**2)/15))/(16*q**5)

    return V

if "__name__" == "__main__":

        
    x_min, x_max = find_x_bounds(q, s)
    print(f"Integration bounds for x: from {x_min:.6f} to {x_max:.6f}")
    print(f"Integration bounds for x: from {x_min:.6f} to {x_max:.6f}")

    try:
        volume, error = dblquad(
            integrand,
            x_min,      # Lower limit for x
            x_max,      # Upper limit for x
            y_lower,    # Lower limit for y as a function of x
            y_upper     # Upper limit for y as a function of x
        )
        print(f"Calculated Volume: {volume:.6f} cubic units")
        print(f"Estimated Integration Error: {error:.6e}")
    except Exception as e:
        print(f"Error during integration: {e}")

    analytical_volume = analytical_volume(h, i, j, q, s)

    print(f"Analytical Volume: {analytical_volume:.6f} cubic units")