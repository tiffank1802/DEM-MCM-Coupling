import pytest
import numpy as np
from ...partitioners import QuantileGridPartitioner as Qua


"""
Hypothèses sur le partitionnement:la numérotation
la numérotation se fait partant de l'axe des abcisses, puis vers l'axe des ordonnées et enfin vers l'axe des z 
"""

#=======================================================================
#vérification des numérotations suivant l'axe des abcisses
#=======================================================================

def test_compute_state_coord_x_0():
    """vérifie que la particule de coordonnées en x=0 est bien dans la partition d'indice 0"""
    coordinates=[[0,0,0],[0,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,0)
    expected=0
    np.testing.assert_(state,expected)


def test_compute_state_coord_x_1():
    """Vérifie que la particule de coordonnées en x=1 est dans la partion d'indice 0"""
    coordinates=[[0,0,0],[1,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(1,0,0)
    expected=0
    np.testing.assert_(state,expected)

def test_compute_state_coord_x_2():
    """Vérifie que la particule de coordonnées en x=2 est dans la partion d'indice 1"""
    coordinates=[[0,0,0],[2,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(2,0,0)
    expected=1
    np.testing.assert_(state,expected)

def test_compute_state_coord_x_3():
    """Vérifie que la particule de coordonnées en x=3 est dans la partion d'indice 2"""
    coordinates=[[0,0,0],[3,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(3,0,0)
    expected=2
    np.testing.assert_(state,expected)

def test_compute_state_coord_x_4():
    """Vérifie que la particule de coordonnées en x=4 est dans la partion d'indice 3"""
    coordinates=[[0,0,0],[4,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(4,0,0)
    expected=3
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_x_5():
    """Vérifie que la particule de coordonnées en x=5 est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[5,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(5,0,0)
    expected=4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_x_6():
    """Vérifie que la particule de coordonnées en x=6 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[6,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(6,0,0)
    expected=4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_x_10():
    """Vérifie que la particule de coordonnées en x=10 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[10,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(10,0,0)
    expected=4
    np.testing.assert_(state,expected)
    

#=======================================================================
#vérification des numérotations suivant l'axe des ordonnées
#=======================================================================

def test_compute_state_coord_y_0():
    """vérifie que la particule de coordonnées en y=0 est bien dans la partition d'indice 0"""
    coordinates=[[0,0,0],[0,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,0)
    expected=0+P.nx*0
    np.testing.assert_(state,expected)


def test_compute_state_coord_y_1():
    """Vérifie que la particule de coordonnées en y=1 est dans la partion d'indice P.nx*0"""
    coordinates=[[0,0,0],[0,1,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,1,0)
    expected=0+P.nx*0
    np.testing.assert_(state,expected)

def test_compute_state_coord_y_2():
    """Vérifie que la particule de coordonnées en y=2 est dans la partion d'indice P.nx*1"""
    coordinates=[[0,0,0],[0,2,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,2,0)
    expected=0+P.nx*1
    np.testing.assert_(state,expected)

def test_compute_state_coord_y_3():
    """Vérifie que la particule de coordonnées en y=3 est dans la partion d'indice P.nx*2"""
    coordinates=[[0,0,0],[0,3,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,3,0)
    expected=0+P.nx*2
    np.testing.assert_(state,expected)

def test_compute_state_coord_y_4():
    """Vérifie que la particule de coordonnées en y=4 est dans la partion d'indice P.nx*3"""
    coordinates=[[0,0,0],[0,4,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,4,0)
    expected=0+P.nx*3
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_y_5():
    """Vérifie que la particule de coordonnées en y=5 est dans la partion d'indice P.nx*4"""
    coordinates=[[0,0,0],[0,5,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,5,0)
    expected=0+P.nx*4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_y_6():
    """Vérifie que la particule de coordonnées en y=6 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[0,6,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,6,0)
    expected=0+P.nx*4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_y_10():
    """Vérifie que la particule de coordonnées en y=10 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[0,10,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,10,0)
    expected=0+P.nx*4
    np.testing.assert_(state,expected)



#=======================================================================
#vérification des numérotations suivant l'axe des z
#=======================================================================


def test_compute_state_coord_z_0():
    """vérifie que la particule de coordonnées en z=0 est bien dans la partition d'indice 0+P.nx*0+P.nx*P.ny*0"""
    coordinates=[[0,0,0],[0,0,0]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,0)
    expected=0+P.nx*0+P.nx*P.ny*0
    np.testing.assert_(state,expected)


def test_compute_state_coord_z_1():
    """Vérifie que la particule de coordonnées en z=1 est dans la partion d'indice 0+P.nx*0+P.nx*P.ny*0"""
    coordinates=[[0,0,0],[0,0,1]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,1)
    expected=0+P.nx*0+P.nx*P.ny*0
    np.testing.assert_(state,expected)

def test_compute_state_coord_z_2():
    """Vérifie que la particule de coordonnées en z=2 est dans la partion d'indice 0+P.nx*0+P.nx*P.ny*1"""
    coordinates=[[0,0,0],[0,0,2]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,2)
    expected=0+P.nx*0+P.nx*P.ny*1
    np.testing.assert_(state,expected)

def test_compute_state_coord_z_3():
    """Vérifie que la particule de coordonnées en z=3 est dans la partion d'indice 0+P.nx*0+P.nx*P.ny*2"""
    coordinates=[[0,0,0],[0,0,3]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,3)
    expected=0+P.nx*0+P.nx*P.ny*2
    np.testing.assert_(state,expected)

def test_compute_state_coord_z_4():
    """Vérifie que la particule de coordonnées en z=4 est dans la partion d'indice 0+P.nx*0+P.nx*P.ny*3"""
    coordinates=[[0,0,0],[0,0,4]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,4)
    expected=0+P.nx*0+P.nx*P.ny*3
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_z_5():
    """Vérifie que la particule de coordonnées en z=5 est dans la partion d'indice 0+P.nx*0+P.nx*P.ny*4"""
    coordinates=[[0,0,0],[0,0,5]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,5)
    expected=0+P.nx*0+P.nx*P.ny*4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_z_6():
    """Vérifie que la particule de coordonnées en z=6 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 0+P.nx*0+P.nx*P.ny*4"""
    coordinates=[[0,0,0],[0,0,6]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,6)
    expected=0+P.nx*0+P.nx*P.ny*4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_z_10():
    """Vérifie que la particule de coordonnées en z=10 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 0+P.nx*0+P.nx*P.ny*4"""
    coordinates=[[0,0,0],[0,0,10]]
    P=Qua() # le nombre de partitons par défaut est nx=5,ny=5,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,10)
    expected=0+P.nx*0+P.nx*P.ny*4
    np.testing.assert_(state,expected)

