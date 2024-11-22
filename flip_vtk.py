#!/usr/bin/env python3
import sys

def negate_second_column(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        in_points_section = False
        points_remaining = 0
        for line in fin:
            if not in_points_section:
                if line.startswith("POINTS"):
                    in_points_section = True
                    parts = line.strip().split()
                    if len(parts) < 3:
                        print("Error: POINTS line does not have enough parts.")
                        sys.exit(1)
                    try:
                        num_points = int(parts[1])
                        data_type = parts[2]
                    except ValueError:
                        print("Error: Cannot parse number of points or data type.")
                        sys.exit(1)
                    points_remaining = num_points * 3  # 3 coordinates per point
                    fout.write(line)  # Write the POINTS header
                else:
                    fout.write(line)  # Write other lines unchanged
            else:
                if points_remaining > 0:
                    # Split the line into tokens
                    tokens = line.strip().split()
                    # Process each coordinate
                    for i in range(0, len(tokens), 3):
                        if points_remaining <= 0:
                            break
                        # Ensure there are enough tokens
                        if i+2 >= len(tokens):
                            print("Warning: Incomplete point data.")
                            break
                        x = tokens[i]
                        y = tokens[i+1]
                        z = tokens[i+2]
                        try:
                            z_neg = str(-float(z))
                        except ValueError:
                            print(f"Warning: Cannot convert '{y}' to float. Leaving unchanged.")
                            z_neg = z
                        # Write the modified point
                        fout.write(f"  {x}   {y}   {z_neg}  \n")
                        points_remaining -= 3
                else:
                    in_points_section = False
                    fout.write(line)  # Write the current line as it's not part of POINTS
        if points_remaining > 0:
            print("Warning: Not enough point data found.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python flip_vtk.py input.vtk output.vtk")
        sys.exit(1)
    input_vtk = sys.argv[1]
    output_vtk = sys.argv[2]
    negate_second_column(input_vtk, output_vtk)
    print(f"Processed '{input_vtk}' and saved the modified VTK to '{output_vtk}'.")
