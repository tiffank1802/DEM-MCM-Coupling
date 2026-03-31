
from src.bucket_io import load_experiment_from_bucket

Matrix=load_experiment_from_bucket("quantile_nx15_ny15_nz15_NLT100_step1_start250_dt0.1")
print(Matrix["matrix"].sum(axis=0))