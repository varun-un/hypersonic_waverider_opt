import os
import time
import subprocess
import numpy as np
import pandas as pd
import shutil
import re
from itertools import product
from scipy.interpolate import griddata
import trajectory as tj


def run_cfd(drag_loc=-5, lift_loc=-4):
    """
    Runs the Champs solver on the given VTK file.

    Parameters:
        drag_loc (int): Column index of the drag coefficient in the integrated_loads.dat file. Default is -5.
        lift_loc (int): Column index of the lift coefficient in the integrated_loads.dat file. Default is -4.

    Returns:
        tuple: (lift, drag) coefficients from the simulation.
               Returns (np.inf, np.inf) if an error occurs.
    """
    try:

        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        
        # Path to the integrated_loads.dat file
        loads_file = os.path.join(parent_dir, "sim", "surf", "integrated_loads.dat")
        
        # Ensure the loads_file exists; if not, wait for it to be created
        max_wait_time = 900  # Maximum wait time in seconds (5 minutes)
        wait_interval = 5    # Interval between checks in seconds
        elapsed_time = 0

        # check if the file exists
        exists = os.path.exists(loads_file)

        print("Integrated loads file exists? ", exists)

        # Record the initial modification time of the loads_file
        if exists:
            initial_mtime = os.path.getmtime(loads_file)
        else:
            initial_mtime = -1

        
        # Submit the CFD job using sbatch
        submit_command = ["./champs+", "input.sdf", " > /dev/null"]
        print("Gonna try to run: ", str(parent_dir), str(submit_command))
        subprocess.run(submit_command, cwd=parent_dir, check=True)
        
        print("CFD job submitted. Waiting for completion...")

        
        # Wait for the loads_file to be updated by the CFD job
        while True:
            time.sleep(wait_interval)
            elapsed_time += wait_interval
            if elapsed_time >= max_wait_time:
                print("Timeout waiting for CFD job to complete.")
                return np.inf, np.inf
            if exists:
                # Check if the file has been modified
                current_mtime = os.path.getmtime(loads_file)
                if current_mtime > initial_mtime:
                    print("CFD job completed.")
                    break
            else:
                # Check if the file exists now
                exists = os.path.exists(loads_file)

        # Read the last line of the integrated_loads.dat file
        with open(loads_file, "r") as file:
            lines = file.readlines()
            if not lines:
                print("Error: integrated_loads.dat is empty.")
                return np.inf, np.inf
            index = -1
            # Check for empty lines at the end of the file
            while not lines[index].strip() and abs(index) <= len(lines):
                index -= 1
            if abs(index) > len(lines):
                print("Error: integrated_loads.dat has no valid data.")
                return np.inf, np.inf
            last_line = lines[index].strip()
            tokens = last_line.split()

            if len(tokens) < abs(drag_loc):
                print("Error: Not enough data in the last line of integrated_loads.dat.")
                return np.inf, np.inf

            try:
                # Extract drag and lift from the specified positions
                drag = float(tokens[drag_loc]) * 2  # Multiply by 2 to get the total drag (runs half waverider)
                lift = float(tokens[lift_loc]) * 2
                print(f"Lift: {lift}, Drag: {drag}")
                return lift, drag
            except ValueError as ve:
                print(f"Error parsing lift and drag values: {ve}")
                return np.inf, np.inf

    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        return np.inf, np.inf
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
        return np.inf, np.inf


def sweep_cfd(mach_range, mach_interval, angle_range, angle_interval):
    """
    Sweeps across ranges of Mach numbers and angles of attack, running CFD simulations
    for each combination and collecting lift and drag.

    Parameters:
        mach_range (tuple): (min_mach, max_mach) inclusive.
        mach_interval (float): Step size for Mach numbers.
        angle_range (tuple): (min_aoa, max_aoa) inclusive, in degrees.
        angle_interval (float): Step size for angles of attack.

    Returns:
        pd.DataFrame: Columns ['mach', 'aoa', 'lift', 'drag'].
    """
    # Unpack ranges
    min_mach, max_mach = mach_range
    min_aoa, max_aoa = angle_range

    # Create lists of Mach and AoA values
    mach_values = np.arange(min_mach, max_mach + mach_interval, mach_interval)
    aoa_values = np.arange(min_aoa, max_aoa + angle_interval, angle_interval)

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # Path to input.sdf
    input_sdf_path = os.path.join(parent_dir, "input.sdf")
    backup_sdf_path = os.path.join(parent_dir, "input_backup.sdf")

    # Backup the original input.sdf if not already backed up
    if not os.path.exists(backup_sdf_path):
        try:
            shutil.copy(input_sdf_path, backup_sdf_path)
            print(f"Backup of input.sdf created at {backup_sdf_path}")
        except Exception as e:
            print(f"Failed to create backup of input.sdf: {e}")
            return pd.DataFrame(columns=['mach', 'aoa', 'lift', 'drag'])

    results = []

    # Iterate over all combinations of AoA and Mach
    for mach, aoa in product(mach_values, aoa_values):
        print(f"\nRunning CFD for Mach {mach}, AoA {aoa} degrees")

        try:
            # Read the backup input.sdf
            with open(backup_sdf_path, 'r') as f:
                sdf_content = f.read()

            # Modify Mach and AoA using regular expressions
            sdf_content_modified = re.sub(
                r'(?m)^\s*mach\s*=\s*\d+(\.\d+)?',
                f'mach     = {mach}',
                sdf_content
            )
            if aoa >= 0:
                aoa_str = str(aoa)
            else:
                aoa_str = f"0.{float(aoa):02d}"
            sdf_content_modified = re.sub(
                r'(?m)^\s*aoa\s*=\s*\d+(\.\d+)?',
                f'aoa      = {aoa_str}',
                sdf_content_modified
            )

            # Write the modified content to input.sdf
            with open(input_sdf_path, 'w') as f:
                f.write(sdf_content_modified)

            print(f"Modified input.sdf with Mach={mach}, AoA={aoa}")

        except Exception as e:
            print(f"Error modifying input.sdf for Mach={mach}, AoA={aoa}: {e}")
            results.append({'mach': mach, 'aoa': aoa, 'lift': np.inf, 'drag': np.inf})
            continue

        # Run the CFD simulation
        lift, drag = run_cfd()

        # Append the results
        results.append({'mach': mach, 'aoa': aoa, 'lift': lift, 'drag': drag})

    # Restore the original input.sdf
    try:
        shutil.copy(backup_sdf_path, input_sdf_path)
        print(f"\nRestored the original input.sdf from {backup_sdf_path}")
    except Exception as e:
        print(f"Error restoring input.sdf: {e}")

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    return df

def interpolate_lift_drag(df, mach, aoa):
    """
    Interpolates lift and drag from the DataFrame for the given Mach and AoA.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['mach', 'aoa', 'lift', 'drag'].
        mach (float): Mach number.
        aoa (float): Angle of attack in degrees.

    Returns:
        tuple: (lift, drag) interpolated values.
    """
    # Ensure the DataFrame is not empty
    if df.empty:
        print("Error: The DataFrame is empty.")
        return np.inf, np.inf

    # Determine the range of Mach and AoA in the DataFrame
    min_mach, max_mach = df['mach'].min(), df['mach'].max()
    min_aoa, max_aoa = df['aoa'].min(), df['aoa'].max()

    # Clip the inputs to the DataFrame's range
    mach_clipped = np.clip(mach, min_mach, max_mach)
    aoa_clipped = np.clip(aoa, min_aoa, max_aoa)

    # Prepare points and corresponding lift and drag values
    points = df[['mach', 'aoa']].values
    lift_values = df['lift'].values
    drag_values = df['drag'].values

    # Perform linear interpolation
    lift_interp = griddata(points, lift_values, (mach_clipped, aoa_clipped), method='linear')
    drag_interp = griddata(points, drag_values, (mach_clipped, aoa_clipped), method='linear')

    # Handle cases where interpolation returns NaN
    if np.isnan(lift_interp):
        # Fallback to nearest interpolation
        lift_interp = griddata(points, lift_values, (mach_clipped, aoa_clipped), method='nearest')
    if np.isnan(drag_interp):
        drag_interp = griddata(points, drag_values, (mach_clipped, aoa_clipped), method='nearest')

    return lift_interp, drag_interp

def AoA_param(a, b, c, d, n, k):
    """
    Creates a function that returns the angle of attack (in degrees) as a function of x.

    a, b, c, d, n, k: Coefficients defining the angle of attack function. It's a damping sine wave.
    """
    def aoa(x):
        return a / (x - c)**n * np.cos(b * (x - c)) + k * (x - c) + d
    
    return aoa

# Trajectory Simulation
def simulate_trajectory(mass, initial_altitude, initial_mach, geometry_length, S, back_area, timestep=0.01, verbose=False, angle_of_attack_func=None, **kwargs):
    """
    Simulate the trajectory of the waverider with variable Angle of Attack (AoA) until it reaches the ground.
    
    Parameters:
        mass (float): Mass of the waverider in kg.
        initial_altitude (float): Initial altitude in meters.
        initial_mach (float): Initial Mach number.
        geometry_length (float): Characteristic length of the geometry in meters.
        S (float): Reference area in m^2.
        back_area (float): Area of the back surface in m^2.
        timestep (float): Time step resolution in seconds.
        verbose (bool): Print simulation details.
        angle_of_attack_func (function): Function that takes x_distance (m) and returns AoA in degrees.
        **kwargs: Additional keyword arguments to pass to get_lift_drag.
                  Can specify Cl and Cd directly. Use `cl` and `cd` to pass the values.
    
    Returns:
        float: Total horizontal distance traveled in meters.
                Accurate to resolution of the timestep.
    """
    alt = []
    x_dist = []
    lift_arr = []
    drag_arr = []
    bp_arr = []
    t_arr = []
    
    altitude = initial_altitude  # z position in meters
    x_position = 0.0             # x position in meters
    
    # Initial atmospheric properties
    atm = get_atm(initial_altitude)
    temperature = atm['temperature']
    
    # Speed of sound
    a = math.sqrt(GAMMA * R * temperature)
    speed = initial_mach * a  # m/s

    # Initial velocity components
    Vx = speed * math.cos(0.0)  # Assuming initial AoA is 0
    Vz = speed * math.sin(0.0)
    
    time_elapsed = 0.0  
    
    while altitude > 0:
        # Prevent glider from orbiting indefinitely
        if x_position > 20075000:  # Half the Earth's circumference
            print("Reached half the circumference of the Earth. Exiting simulation.")
            break

        current_speed = math.sqrt(Vx**2 + Vz**2)  # Relative speed
        
        # Get lift and drag forces
        F_L, F_D = get_lift_drag(current_speed, min(altitude, MAX_ALT), geometry_length, S, back_area, **kwargs)
        
        # Determine Angle of Attack
        if angle_of_attack_func is not None:
            AoA_deg = angle_of_attack_func(x_dist)
            AoA_rad = math.radians(AoA_deg)
        else:
            AoA_deg = 0
            AoA_rad = 0.0  # Default to 0 if no function is provided

        theta = math.atan2(-Vz, Vx)

        phi = theta - AoA_rad
        
        # Decompose Drag Force
        F_Dx = -F_D * math.cos(phi)
        F_Dz = F_D * math.sin(phi)
        
        # Decompose Lift Force based on AoA
        F_Lx = F_L * math.sin(phi)
        F_Lz = F_L * math.cos(phi)
        
        # Gravity Force
        F_g = -mass * G  # Downward
        
        # Net Forces
        F_net_x = F_Dx + F_Lx
        F_net_z = F_Dz + F_Lz + F_g

        # print(F_net_x, F_net_z)
        
        # Accelerations
        a_x = F_net_x / mass
        a_z = F_net_z / mass
        
        # Update velocities
        Vx += a_x * timestep
        Vz += a_z * timestep
        
        # Update positions
        x_position += Vx * timestep
        altitude += Vz * timestep
        
        # Update time
        time_elapsed += timestep
        
        # Prevent negative altitude
        if altitude < 0:
            altitude = 0

        # Append data for plotting
        if verbose:
            alt.append(altitude)
            x_dist.append(x_position)
            lift_arr.append(F_L)
            drag_arr.append(F_D)
            bp_arr.append(back_area * atm['pressure'] / (current_speed / a))
            t_arr.append(time_elapsed)

            atm = get_atm(min(altitude, MAX_ALT))
            temperature = atm['temperature']
            a_speed = math.sqrt(GAMMA * R * temperature)
            mach = current_speed / a_speed
            back_pressure = back_area * atm['pressure'] / mach
            
            print(f"Time: {time_elapsed:.3f}s, X: {x_position:.3f}m, Altitude: {altitude:.3f}m, Vx: {Vx:.3f}m/s, Vz: {Vz:.3f}m/s, Mach: {mach:.3f}, AoA: {AoA_deg:.2f}, Lift: {F_L:.3f}N, Drag: {F_D:.3f}N, Back Pressure: {back_pressure:.3f}N")

def tj_cost_fcn(params):
    pass