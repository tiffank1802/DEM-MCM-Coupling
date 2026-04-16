
from src.bucket_io import load_experiment_from_bucket

Matrix=load_experiment_from_bucket("quantile_nx5_ny5_nz1_NLT10_step10_dt2_tau50_start250")
print(Matrix["matrix"].sum(axis=0))