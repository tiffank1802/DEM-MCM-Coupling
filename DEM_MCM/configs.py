from src import run_sweep as r_s


configs=[
    r_s.ExperimentConfig("cartesian",{"nx":5,"ny":3,"nz":5},nlt=10,dt=2,step=50),
    r_s.ExperimentConfig("cartesian",{"nx":4,"ny":3,"nz":4},nlt=10,dt=2,step=50),
    r_s.ExperimentConfig("cartesian",{"nx":3,"ny":3,"nz":3},nlt=10,dt=2,step=50),


]


r_s.run_markov_sweep("cartesian",configs=configs)