
from main import generate_mesh, get_i, back_area
import time
import tj_opt
import dill

a = -0.3
c = 0.0
f = 0
g = 0.0
h = 0.0
i = 0.12
j = 0.0
q = 0
s = 3


dy = 0.005
initial_N = 202
filename = "../generated_waverider.vtk"

best_params = [a, c, f, g, h, j, q, s]

generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, filename)

# also save it to the output folder
output_folder = f"./output/{int(time.time())}_post"
best_vtk = f"{output_folder}/best_waverider.vtk"
generate_mesh(a, c, f, g, h, i, j, q, s, dy, initial_N, best_vtk)

# do CFD sweep
mach_range = (2, 8)
mach_step = 1
angle_range = (-8, 8)
angle_step = 2

try:
    print("Running CFD sweep...")
    df = tj_opt.sweep_cfd(mach_range, mach_step, angle_range, angle_step)
except Exception as e:
    print(f"Error running CFD sweep: {e}")
    df = None

# save the results
try:
    df.to_csv(f"{output_folder}/cfd_sweep_results.csv", index=False)
except Exception as e:
    print(f"Error saving CFD sweep results: {e}")

A_b = back_area(h, i, j, q, s)

# https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
try:
    result = tj_opt.optimize_aoa(df, A_b)
except Exception as e:
    print(f"Error optimizing AOA: {e}")
    result = None

# save the results to a pickle file
try:
    with open(f"{output_folder}/aoa_opt_results.pkl", "wb") as file:
        dill.dump(result, file)
except Exception as e:
    print(f"Error saving AOA optimization results: {e}")

try:
    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed.")
except Exception as e:
    print(f"Error checking optimization success: {e}")

# print the results
print(result)
