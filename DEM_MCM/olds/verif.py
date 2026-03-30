import run_sweep as rs
from partitioners import CylindricalPartitioner as Cyl
from analyze_results import MarkovAnalyzer as MK
import numpy as np
import torch
import json

def test_sum_colonne():
    # coordinates_prev=torch.rand(5,3)
    # coordinates_curr=torch.rand(5,3)
    coordinates_prev=np.random.rand(5,3)
    coordinates_curr=np.random.rand(5,3)
    partitioner=Cyl(nr=2,ntheta=2,nz=1)
    partitioner=partitioner.fit(coordinates=coordinates_prev)
    x,y,z=coordinates_prev[:,0],coordinates_prev[:,1],coordinates_prev[:,2]
    s_prev=partitioner.compute_states(x=x,y=y,z=z)
    x,y,z=coordinates_curr[:,0],coordinates_curr[:,1],coordinates_curr[:,2]
    s_curr=partitioner.compute_states(x=x,y=y,z=z)
    P=rs.compute_P_matrix_numpy(states_prev=s_prev,states_curr=s_curr,n_states=4)
    P=np.asarray(P)
    sum_P=np.sum(P,0)
    print(sum_P)
    expected=1

analyzer=MK()
# analyzer.load_method("cylindrical")
M=analyzer._load_experiment(folder_name='cylindrical_nr3_nth1_nz5_equal_area_NLT100_step1_start250')
# print(M["matrix"].sum(axis=0))
folders=analyzer._list_folders()
verif=dict()
for folder in folders:
    P=analyzer._load_experiment(folder_name=folder)
    verif[folder]=(P["matrix"].sum(axis=0).all()==1)or (P["matrix"].sum(axis=1).all()==1)
print(verif)

file = open('verifications.txt' , 'w')
 
# ajouter les lignes au fichiers students.txt
file.writelines(list(verif))
file.close()