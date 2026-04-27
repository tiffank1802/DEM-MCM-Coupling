
from src.bucket_io import load_experiment_from_bucket

Matrix=load_experiment_from_bucket("cylindrical_nr1_nth10_nz1_equal_area_NLT10_step10_dt2_tau50_start250")
print(Matrix["matrix"].sum(axis=0))