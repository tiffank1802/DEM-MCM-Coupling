from src import run_sweep as r_s


configs=[
    # r_s.ExperimentConfig("cartesian",{"nx":5,"ny":3,"nz":5},nlt=10,dt=2,step=50),
    # r_s.ExperimentConfig("cartesian",{"nx":4,"ny":3,"nz":4},nlt=10,dt=2,step=50),
    # r_s.ExperimentConfig("cartesian",{"nx":3,"ny":2,"nz":3},nlt=10,dt=2,step=50),
    # r_s.ExperimentConfig("cylindrical",{"nr":1,"ntheta":10,"nz":1},),
    # r_s.ExperimentConfig(method="octree", method_kwargs={"max_particles": 5, "max_depth": 2}, particle_diameter=0.004),

    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 20, "velocity_weight": 0.5}, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 40, "velocity_weight": 0.5}, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 50, "velocity_weight": 0.5}, particle_diameter=0.004),

    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 20, "velocity_weight": 0.5}, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 40, "velocity_weight": 0.5}, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 50, "velocity_weight": 0.5}, particle_diameter=0.008),

    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 20, "velocity_weight": 0.0}, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 20, "velocity_weight": 1.0}, particle_diameter=0.004),

]


r_s.run_markov_sweep("physics", configs=configs, particle_diameter=None)
