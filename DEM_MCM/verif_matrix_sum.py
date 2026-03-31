
from olds.bucket_io import load_experiment_from_bucket

Matrix=load_experiment_from_bucket("cylindrical_nr5_nth8_nz5_equal_area_NLT100_step10_start250")
print(Matrix["matrix"].sum(axis=0))