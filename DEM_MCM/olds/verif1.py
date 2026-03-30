import numpy as np
import run_sweep as rs
from partitioners import CylindricalPartitioner as Cyl
from analyze_results import MarkovAnalyzer as MK
import numpy as np
import torch
import json
from bucket_io import load_matrix_from_bucket

def phi_particule(state:int,partition:int)-> bool:
    """Vérifie si une particule est bien dans une partition"""
    return 1 if state==partition else 0

def phi_sum_partition(states: list[int],partition:int)-> int:
    """Somme les particules qui sont dans une partition"""
    phi_s=0
    for i in range(len(states)):
        phi_s+=phi_particule(states[i],partition=partition)
    return phi_s
coordinates_prev=np.random.rand(500,3)
coordinates_curr=np.random.rand(500,3)
partitioner=Cyl(nr=400,ntheta=1,nz=1)
partitioner=partitioner.fit(coordinates=coordinates_prev)
x,y,z=coordinates_prev[:,0],coordinates_prev[:,1],coordinates_prev[:,2]
s_prev=partitioner.compute_states(x=x,y=y,z=z)
x,y,z=coordinates_curr[:,0],coordinates_curr[:,1],coordinates_curr[:,2]
s_curr=partitioner.compute_states(x=x,y=y,z=z)
# n=partitioner.n_cells
# M=np.zeros((n,n))

# for i in range(n):
#     print(i,phi_sum_partition(s_curr,i))
#     for j in range(n):
#         inter=0
#         for p in range(len(s_curr)):
#             inter+=phi_particule(state=s_prev[p],partition=i)*phi_particule(state=s_curr[p],partition=j)
#         M[i,j]=inter/phi_sum_partition(states=s_prev,partition=i) if phi_sum_partition(states=s_prev,partition=i)>0 else 0
# M=M.T

def compute_P_matrix_torch(states_prev, states_curr, n_states, device="cpu"):
    """
    Calcule P_n pour un timestep en utilisant phi_particule et phi_sum_partition.
    Normalisation par colonnes (somme des colonnes = 1).
    """
    # Conversion en tensor si nécessaire
    if isinstance(states_curr, np.ndarray):
        states_curr = torch.from_numpy(states_curr)
    if isinstance(states_prev, np.ndarray):
        states_prev = torch.from_numpy(states_prev)
    
    s_prev = states_prev.to(device).long()
    s_curr = states_curr.to(device).long()
    
    # Initialisation de la matrice de transition
    P = torch.zeros((n_states, n_states), device=device, dtype=torch.float64)
    
    # Calcul des transitions P[i,j] = probabilité d'aller de i à j
    for i in range(n_states):
        for j in range(n_states):
            # Compte les transitions de i vers j
            inter = 0
            n = min(len(s_prev), len(s_curr))
            for p in range(n):
                inter += phi_particule(state=s_prev[p].item(), partition=i) * phi_particule(state=s_curr[p].item(), partition=j)
            
            # Normalisation par le nombre de particules dans l'état i au temps précédent
            denominator = phi_sum_partition(s_prev.cpu().numpy(), i)
            P[i, j] = inter / denominator if denominator > 0 else 0.0
    
    # Transposition pour avoir les états courants en lignes, précédents en colonnes
    P = P.T
    
    # # Normalisation par colonnes (somme des colonnes = 1) avec torch.sum(dim=0)
    # col_sums = torch.sum(P, dim=0)
    
    # P = torch.where(col_sums > 0, P / col_sums, torch.zeros_like(P))
    
    return P

P=compute_P_matrix_torch(states_prev=s_prev,states_curr=s_curr,n_states=partitioner.n_cells,device="cpu")
P=np.asarray(P)
# print(P.sum(axis=0))

