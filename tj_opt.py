import os
import time
import subprocess
import numpy as np
import pandas as pd
import shutil
import re
from itertools import product
from scipy.interpolate import griddata


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
        import os
        import time
        import subprocess
        import numpy as np

        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

        # Path to the integrated_loads.dat file
        loads_file = os.path.join(parent_dir, "sim", "surf", "integrated_loads.dat")

        # Ensure the loads_file exists; if not, wait for it to be created
        max_wait_time = 900  # Maximum wait time in seconds (15 minutes)
        wait_interval = 5    # Interval between checks in seconds
        elapsed_time = 0

        while not os.path.exists(loads_file):
            if elapsed_time >= max_wait_time:
                print(f"Timeout: {loads_file} does not exist.")
                return np.inf, np.inf
            time.sleep(wait_interval)
            elapsed_time += wait_interval

        # Record the initial modification time of the loads_file
        initial_mtime = os.path.getmtime(loads_file)

        # Submit the CFD job using sbatch
        submit_command = ["./champs+", "input.sdf"]
        subprocess.run(submit_command, cwd=parent_dir, check=True)

        print("CFD job submitted. Waiting for completion...")

        # Wait for the loads_file to be updated by the CFD job
        while True:
            time.sleep(wait_interval)
            elapsed_time += wait_interval
            if elapsed_time >= max_wait_time:
                print("Timeout waiting for CFD job to complete.")
                return np.inf, np.inf
            current_mtime = os.path.getmtime(loads_file)
            if current_mtime > initial_mtime:
                print("CFD job completed.")
                break

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
            sdf_content_modified = re.sub(
                r'(?m)^\s*aoa\s*=\s*\d+(\.\d+)?',
                f'aoa      = {aoa}',
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
