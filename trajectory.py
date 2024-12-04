import math
import pandas as pd
import numpy as np
import volume as vol
from scipy.interpolate import griddata
from scipy.optimize import minimize
from ambiance import Atmosphere

# Constants
GAMMA = 1.4  
R = 287.05  
G = 9.81    
MAX_ALT = 81020  # meters

# ambiance library (ref PyPi) (uses 1993 Std Atm tables)
def get_atm(altitude):
    """
    Retrieve atmospheric properties at a given altitude using ambiance lib.
    
    Parameters:
        altitude (float): Altitude in meters.
    
    Returns:
        dict: Dictionary containing temperature (K), pressure (Pa), density, dynamic_viscosity.
    """
    atm = Atmosphere(altitude)
    return {
        'temperature': atm.temperature.item(),          # K
        'pressure': atm.pressure.item(),                # Pa
        'density': atm.density.item(),                  # kg/m^3
        'dynamic_viscosity': atm.kinematic_viscosity.item() * atm.density.item()  # mu = ν * ρ
    }

# Method 1: Using lift_drag.csv (Cl, Cd, Mach, Re)
def get_cl_cd_MRe(speed, altitude, geometry_length, get_atm):
    """
    Retrieve C_L and C_D from lift_drag.csv based on Mach number and Reynolds number.
    Grid interpolation is used to estimate the values between data points. This uses simplex, 
    so at least 3 points are needed to interpolate.
    
    Parameters:
        speed (float): Speed in m/s.
        altitude (float): Altitude in meters.
        geometry_length (float): Characteristic length of the geometry in meters.
        get_atm (function): Function to get atmospheric properties.
    
    Returns:
        tuple: (C_L, C_D)
    """
    # Read CSV if not already done
    # caches dataframe lookups for speed (may not be necessary but idk)
    if not hasattr(get_cl_cd_MRe, "df"):
        get_cl_cd_MRe.df = pd.read_csv('lift_drag.csv')
    
    # Get atmospheric properties
    atm = get_atm(altitude)
    temperature = atm['temperature']
    pressure = atm['pressure']
    rho = atm['density']
    mu = atm['dynamic_viscosity']
    
    a = math.sqrt(GAMMA * R * temperature)
    Mach = speed / a
    Re = (rho * speed * geometry_length) / mu
    
    points = get_cl_cd_MRe.df[['M', 'Re']].values
    Cl = get_cl_cd_MRe.df['Cl'].values
    Cd = get_cl_cd_MRe.df['Cd'].values
    
    # Interpolate Cl and Cd
    C_L = griddata(points, Cl, (Mach, Re), method='linear')
    C_D = griddata(points, Cd, (Mach, Re), method='linear')
    
    # Extrapolate the data using nearest neighbor tree
    if np.isnan(C_L):
        C_L = griddata(points, Cl, (Mach, Re), method='nearest')
    if np.isnan(C_D):
        C_D = griddata(points, Cd, (Mach, Re), method='nearest')
    
    return C_L, C_D

# Method 2: Using lift_drag_atm.csv (Cl, Cd, Speed, Altitude)
def get_cl_cd_SAlt(speed, altitude):
    """
    Retrieve C_L and C_D from lift_drag_atm.csv based on speed and altitude.
    Similarly, grid interpolation is used to estimate the values between data points,
    so it needs at least 3 data entries.
    
    Parameters:
        speed (float): Speed in m/s.
        altitude (float): Altitude in meters.
    
    Returns:
        tuple: C_L, C_D
    """
    # cahce csv read
    if not hasattr(get_cl_cd_SAlt, "df"):
        get_cl_cd_SAlt.df = pd.read_csv('lift_drag_atm.csv')
    
    df = get_cl_cd_SAlt.df
    
    points = df[['Speed(m/s)', 'Altitude']].values
    Cl = df['Cl'].values
    Cd = df['Cd'].values
    
    # Interpolate Cl and Cd
    C_L = griddata(points, Cl, (speed, altitude), method='linear')
    C_D = griddata(points, Cd, (speed, altitude), method='linear')
    
    # nearest neighbor tree
    if np.isnan(C_L):
        C_L = griddata(points, Cl, (speed, altitude), method='nearest')
    if np.isnan(C_D):
        C_D = griddata(points, Cd, (speed, altitude), method='nearest')
    
    return C_L, C_D


# Lift and Drag Calculation
def get_lift_drag(speed, altitude, geometry_length, S, back_area, **kwargs):
    """
    Calculate lift and drag forces based on speed and altitude.
    
    Parameters:
        speed (float): Speed in m/s.
        altitude (float): Altitude in meters.
        geometry_length (float): Characteristic length of the geometry in meters.
        S (float): Reference area in m^2.
        back_area (float): Area of the back surface in m^2.
        **kwargs: Additional keyword arguments to specify Cl and Cd directly.
                    Use `cl` and `cd` to pass the values.
    
    Returns:
        tuple: (F_L, F_D) in N
    """

    if 'cl' in kwargs and 'cd' in kwargs:
        C_L = kwargs['cl']
        C_D = kwargs['cd']
    else:
        # Get Cl and Cd
        C_L, C_D = get_cl_cd_MRe(speed, altitude, geometry_length, get_atm)

    
    # std atm density
    atm = get_atm(altitude)
    rho = atm['density']
    
    # Calculate dynamic pressure
    q = 0.5 * rho * speed ** 2
    
    # lift & drag formula
    F_L = C_L * q * S
    F_D = C_D * q * S

    # add back pressure
    a = math.sqrt(GAMMA * R * atm['temperature'])
    F_D += back_area * atm['pressure'] / (speed / a)
    
    return F_L, F_D

# Trajectory Simulation
def simulate_trajectory(mass, initial_altitude, initial_mach, geometry_length, S, back_area, timestep=.01, verbose = False, **kwargs):
    """
    Simulate the trajectory of the waverider until it reaches the ground.
    
    Parameters:
        mass (float): Mass of the waverider in kg.
        initial_altitude (float): Initial altitude in meters.
        initial_mach (float): Initial Mach number.
        geometry_length (float): Characteristic length of the geometry in meters.
        S (float): Reference area in m^2.
        back_area (float): Area of the back surface in m^2.
        timestep (float): Time step resolution in seconds.
        verbose (bool): Print simulation details.
        **kwargs: Additional keyword arguments to pass to get_lift_drag.
                    Can specify Cl and Cd directly. Use `cl` and `cd` to pass the values.
    
    Returns:
        float: Total horizontal distance traveled in meters.
                Accurate to resolution of the timestep.
    """
    
    altitude = initial_altitude  # z position in meters
    x_position = 0.0             # x position in meters
    
    # initial properties
    atm = get_atm(altitude)
    temperature = atm['temperature']
    
    # speed of sound
    a = math.sqrt(GAMMA * R * temperature)
    speed = initial_mach * a  # m/s

    Vx = speed  
    Vz = 0.0  
    
    time_elapsed = 0.0  
    
    while altitude > 0:

        # max out at circumference of earth
        if x_position > 20075000:
            print("Reached the half the circumference of the Earth. Exiting simulation.")
            break

        current_speed = math.sqrt(Vx**2 + Vz**2)        # normalizing factor
        
        # Get lift and drag forces
        F_L, F_D = get_lift_drag(current_speed, min(altitude, MAX_ALT), geometry_length, S, back_area, **kwargs)
        
        # use vector dynamics to decompose and calculate forces
        # in order to maintain 0 AoA as it falls, it will pitch to maintain 0 AoA, where it 
        # faces the fresstream velocity vector, and lift is perpendicular to this
        # in the global frame, use the velocity vector to decompose the forces
        if current_speed != 0:
            F_Dx = -F_D * (Vx / current_speed)
            F_Dz = -F_D * (Vz / current_speed)
        else:
            F_Dx = 0.0
            F_Dz = 0.0

        if current_speed != 0:
            F_Lx = F_L * (-Vz / current_speed)
            F_Lz = F_L * (Vx / current_speed)
        else:
            F_Lx = 0.0
            F_Lz = F_L  # All lift acts vertically if there's no horizontal speed
        
        # grav
        F_g = -mass * G  # Negative since it acts downward
        
        # Net forces
        F_net_x = F_Dx + F_Lx
        F_net_z = F_Dz + F_Lz + F_g
        
        # Accelerations
        a_x = F_net_x / mass
        a_z = F_net_z / mass
        
        # Update velocities - by doing numerical integration we doing a simple euler method for dt
        Vx += a_x * timestep
        Vz += a_z * timestep
        
        # Update positions
        x_position += Vx * timestep
        altitude += Vz * timestep

        time_elapsed += timestep
        
        if altitude < 0:
            altitude = 0
        
        if verbose:
            atm = get_atm(min(altitude, MAX_ALT))
            temperature = atm['temperature']
            a = math.sqrt(GAMMA * R * temperature)
            mach = current_speed / a
            back_pressure = back_area * atm['pressure'] / mach

            print(f"Time: {time_elapsed:.3f}s, X: {x_position:.3f}m, Altitude: {altitude:.3f}m, Vx: {Vx:.3f}m/s, Vz: {Vz:.3f}m/s, Mach: {mach:.3f}, Lift: {F_L:.3f}N, Net Drag: {F_D:.3f}N, Back Pressure: {back_pressure:.3f}N")
    
    return x_position

def single_shot():

    mass = 25           # kg
    initial_altitude = 40000.0  # meters (40 km)
    initial_mach = 10.0      # Mach number
    geometry_length = 1.0  # meters
    back_area = vol.back_area(0, 0.12, 0, 0, 3)  # m^2
    S = back_area  # Reference Area (m^2)
    timestep = 1          # seconds

    # using first case Duncan ran
    lift = 794.6*2
    drag = 90.3*2
    print(f"Using Lift: {lift:.3f}, Drag: {drag:.3f}")

    atm = get_atm(30000)
    density = atm['density']
    a = math.sqrt(GAMMA * R * atm['temperature'])
    speed = 7 * a

    # convert to cl and cd
    cl = lift / (0.5 * density * speed**2 * S)
    cd = drag / (0.5 * density * speed**2 * S)

    print(f"Using Cl: {cl:.3f}, Cd: {cd:.3f}")
    print(f"Simulating trajectory starting from {initial_altitude} meters with Mach {initial_mach} and reference area {back_area} meters.")
    
    distance_traveled = simulate_trajectory(
        mass=mass,
        initial_altitude=initial_altitude,
        initial_mach=initial_mach,
        geometry_length=geometry_length,
        S=S,
        timestep=timestep,
        verbose=True,
        back_area=back_area,
        cl=cl,
        cd=cd
    )
    
    print(f"Total horizontal distance traveled: {distance_traveled:.3f} meters")

    

if __name__ == "__main__":
    single_shot()
