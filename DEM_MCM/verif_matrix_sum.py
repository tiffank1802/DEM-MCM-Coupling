
from olds.bucket_io import load_experiment_from_bucket

Matrix=load_experiment_from_bucket("voronoi_27cells_NLT100_step1_start250")
print(Matrix["matrix"].sum(axis=0))