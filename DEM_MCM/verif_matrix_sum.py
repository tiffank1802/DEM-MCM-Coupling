
from src.bucket_io import load_experiment_from_bucket
import numpy as np
import matplotlib.pyplot as plt

# a=np.array([[1,0,0],[0,0,1],[0,2,0]])
# print(a.sum(axis=1))
# Matrix=load_experiment_from_bucket("cylindrical_nr2_nth5_nz5_equal_area_NLT10_step50_dt2_tau1000_start250_d0004")
Matrix=load_experiment_from_bucket("cylindrical_nr2_nth4_nz4_equal_area_NLT20_step50_dt2_tau50_start250_d0004")
print(Matrix["matrix"].sum(axis=0))
# M=Matrix["matrix"]
# S=np.zeros(10)
# S[0:2]=1
# S_history=np.zeros((10,10))
# for i in range(10):
#     S_history[i,:]=S
#     print(S[1:-2].std()/S[1:-2].mean())
#     S=S@M

# plt.plot(S_history[:,1:-2].std(axis=1)/S_history[:,1:-2].mean(axis=1),'.')
# plt.savefig('s.png')