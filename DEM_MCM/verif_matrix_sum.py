
from src.bucket_io import load_experiment_from_bucket

Matrix=load_experiment_from_bucket("adaptive_z_cylindrical_top1_bot320_split0.075355_n_bot320_n_top1_NLT100_step10_start250_dt0.1")
print(Matrix["matrix"].sum(axis=0))