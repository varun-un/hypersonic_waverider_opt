# hypersonic_waverider_opt
 
### Core Scripts
- `volume.py`: Functions for core volume, area, and bounds calculations. If ran as main, does a single shot comparison of numerical volume calculations vs analytical calculation to validate the analytical volume solver.
- `trajectory.py`: Core functionality for calculating the trajectory and flight path of the waverider given Cl and Cd values. Mostly a numerical dynamics simulation, but also contains standard atmosphere table lookups. If set to verbose, it'll generate plots and print outputs. If ran as main, it performs a trajectory simulation for a fixed given Cl and Cd.
- `mesher.py`: All core functionality related to generating a VTK from equations. Uses a custom meshing algorithm to generate a manifold mesh for a geometry to be used. Running it as main generates and displays visualizations for a given set of parameters
- `tj_opt`: Contains all the code for the trajectory optimization by optimizing the angle of attack function. Defines a parameteric function for angle of attack (akin to damping sinusoidal with bias), and has the Nelder-Mead calls to optimize it. It also includes functionality to read, call, and interpolate the CFD lift and drag data during the mach value and angle of attack sweep.
- `main.py`: Runs everything. Also has the get_i function. Contains logic for shape optimization, as well as calling and reading the CFD outputs. In order, it will perform shape optimization with Bayesian Optimization, take best geometry and perform sweep of Mach numbers and Angle of Attacks, and feed that into the AoA function optimizer. Continuously pickles outputs and prints progress.

### Post-processing
- `flip_vtk.py`: Simply flips a vtk to negate the y coordinates to fix incorrect flow direction for CFD

  To run: `python flip_vtk.py <input_vtk_file> <output_vtk_file>`
- `get_mesh.py`: Reads a log output file (the piped output from *main.py*'s shape optimization) and when given an iteration number, will automatically produce the VTK file that iteration tested

  To run: `python get_mesh.py <log_file_output> <iteration_number>`
- `generate_valid_vtks.py`: Given the log output file, it will process every valid (correct-volumed and converged) iteration that the Bayesian optimization had, and produce a VTK for each. This will be saved to a folder and numbered chronologically, with the best performing geometry occurring a second time as the last VTK. Useful for producing animations.
  
  To run: `python generate_valid_vtks.py <log_file_output> <folder_name>`
- `analysis.py`: Has a whole bunch of post processing tools and metrics to decode the output of the shape Bayesian Optimization and produce some cool plots of convergence and tested values, and etc. Requires the pickle files in the output folder. The plot folder location flag is optional.
   
  To run: `python analysis.py --checkpoint_dir <route_to_timestamped_output_folder> --output_dir <plot_folder_location>`

### Output
Other than just printing out results, which can be parsed, `main.py` saves pickel file outputs for a run to `outputs/UNIX_TIMESTAMP/`, with a new timestamped folder created for each run.
