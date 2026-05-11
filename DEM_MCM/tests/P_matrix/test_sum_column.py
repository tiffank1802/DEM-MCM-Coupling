import pytest
from ... import run_sweep as rs
from ...partitioners import CylindricalPartitioner as Cyl
import numpy as np
import torch

def test_sum_colonne():
    coordinates_prev=torch.rand(5,3)
    coordinates_curr=torch.rand(5,3)
    partitioner=Cyl()
    partitioner=partitioner.fit(coordinates=coordinates_prev)
    x,y,z=coordinates_prev[:,0],coordinates_prev[:,1],coordinates_prev[:,2]
    s_prev=partitioner.compute_states(x=x,y=y,z=z)
    x,y,z=coordinates_curr[:,0],coordinates_curr[:,1],coordinates_curr[:,2]
    s_curr=partitioner.compute_states(x=x,y=y,z=z)
    P=rs.compute_P_matrix_torch(states_prev=s_prev,states_curr=s_curr,n_states=np.max(s_prev+1))
    P=np.asarray(P)
    print(P)
    sum_P=P.sum(axis=1)
    expected=1
    np.testing.assert_(sum_P.all(),expected)