
from src.bucket_io import load_experiment_from_bucket

Matrix=load_experiment_from_bucket("physics_125cells_pos_NLT5_step10_dt2_tau50_start250")
print(Matrix["matrix"].sum(axis=0))