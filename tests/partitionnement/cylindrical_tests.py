import pytest
import numpy as np
from ...partitioners import CylindricalPartitioner as Cyl

"""
Hypothèses sur le partitionnement:la numérotation
la numérotation se fait partant des rayons, puis vers les angles et enfin vers l'are des z 
"""


#=======================================================================
#vérification des numérotations suivant les rayons
#=======================================================================

def test_compute_state_coord_r_0():
    """vérifie que la particule de coordonnées en r=0 est bien dans la partition d'indice 0"""
    coordinates=[[0,0,0],[0,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,0)
    erpected=0
    np.testing.assert_(state,erpected)


def test_compute_state_coord_r_1():
    """Vérifie que la particule de coordonnées en r=1 est dans la partion d'indice 0"""
    coordinates=[[0,0,0],[1,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(1,0,0)
    erpected=0
    np.testing.assert_(state,erpected)

def test_compute_state_coord_r_2():
    """Vérifie que la particule de coordonnées en r=2 est dans la partion d'indice 1"""
    coordinates=[[0,0,0],[2,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(2,0,0)
    erpected=1
    np.testing.assert_(state,erpected)

def test_compute_state_coord_r_3():
    """Vérifie que la particule de coordonnées en r=3 est dans la partion d'indice 2"""
    coordinates=[[0,0,0],[3,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(3,0,0)
    erpected=2
    np.testing.assert_(state,erpected)

def test_compute_state_coord_r_4():
    """Vérifie que la particule de coordonnées en r=4 est dans la partion d'indice 3"""
    coordinates=[[0,0,0],[4,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(4,0,0)
    erpected=3
    np.testing.assert_(state,erpected)
    
def test_compute_state_coord_r_5():
    """Vérifie que la particule de coordonnées en r=5 est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[5,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(5,0,0)
    erpected=4
    np.testing.assert_(state,erpected)
    
def test_compute_state_coord_r_6():
    """Vérifie que la particule de coordonnées en r=6 qui est supérieure nombre de partitions suivant l'are de abcisses est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[6,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(6,0,0)
    erpected=4
    np.testing.assert_(state,erpected)
    
def test_compute_state_coord_r_10():
    """Vérifie que la particule de coordonnées en r=10 qui est supérieure nombre de partitions suivant l'are de abcisses est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[10,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(10,0,0)
    erpected=4
    np.testing.assert_(state,erpected)
    


#=======================================================================
#vérification des numérotations suivant les angles
#=======================================================================


def test_compute_state_coord_theta_0():
    """vérifie que la particule de coordonnées en theta=0 est bien dans la partition d'indice 0"""
    coordinates=[[0,0,0],[0,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,0)
    expected=0+P.nr*0
    np.testing.assert_(state,expected)


def test_compute_state_coord_theta_1():
    """Vérifie que la particule de coordonnées en theta=1 est dans la partion d'indice P.nr*0"""
    coordinates=[[0,0,0],[0,1,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,1,0)
    expected=0+P.nr*0
    np.testing.assert_(state,expected)

def test_compute_state_coord_theta_2():
    """Vérifie que la particule de coordonnées en theta=2 est dans la partion d'indice P.nr*1"""
    coordinates=[[0,0,0],[0,2,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,2,0)
    expected=0+P.nr*1
    np.testing.assert_(state,expected)

def test_compute_state_coord_theta_3():
    """Vérifie que la particule de coordonnées en theta=3 est dans la partion d'indice P.nr*2"""
    coordinates=[[0,0,0],[0,3,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,3,0)
    expected=0+P.nr*2
    np.testing.assert_(state,expected)

def test_compute_state_coord_theta_4():
    """Vérifie que la particule de coordonnées en theta=4 est dans la partion d'indice P.nr*3"""
    coordinates=[[0,0,0],[0,4,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,4,0)
    expected=0+P.nr*3
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_theta_5():
    """Vérifie que la particule de coordonnées en theta=5 est dans la partion d'indice P.nr*4"""
    coordinates=[[0,0,0],[0,5,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,5,0)
    expected=0+P.nr*4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_theta_6():
    """Vérifie que la particule de coordonnées en theta=6 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[0,6,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,6,0)
    expected=0+P.nr*4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_theta_10():
    """Vérifie que la particule de coordonnées en theta=10 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 4"""
    coordinates=[[0,0,0],[0,10,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,10,0)
    expected=0+P.nr*4
    np.testing.assert_(state,expected)
    

#=======================================================================
#vérification des numérotations suivant l'axe des z
#=======================================================================



def test_compute_state_coord_z_0():
    """vérifie que la particule de coordonnées en z=0 est bien dans la partition d'indice 0+P.nr*0+P.nr*P.ntheta*0"""
    coordinates=[[0,0,0],[0,0,0]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,0)
    expected=0+P.nr*0+P.nr*P.ntheta*0
    np.testing.assert_(state,expected)


def test_compute_state_coord_z_1():
    """Vérifie que la particule de coordonnées en z=1 est dans la partion d'indice 0+P.nr*0+P.nr*P.ntheta*0"""
    coordinates=[[0,0,0],[0,0,1]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,1)
    expected=0+P.nr*0+P.nr*P.ntheta*0
    np.testing.assert_(state,expected)

def test_compute_state_coord_z_2():
    """Vérifie que la particule de coordonnées en z=2 est dans la partion d'indice 0+P.nr*0+P.nr*P.ntheta*1"""
    coordinates=[[0,0,0],[0,0,2]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,2)
    expected=0+P.nr*0+P.nr*P.ntheta*1
    np.testing.assert_(state,expected)

def test_compute_state_coord_z_3():
    """Vérifie que la particule de coordonnées en z=3 est dans la partion d'indice 0+P.nr*0+P.nr*P.ntheta*2"""
    coordinates=[[0,0,0],[0,0,3]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,3)
    expected=0+P.nr*0+P.nr*P.ntheta*2
    np.testing.assert_(state,expected)

def test_compute_state_coord_z_4():
    """Vérifie que la particule de coordonnées en z=4 est dans la partion d'indice 0+P.nr*0+P.nr*P.ntheta*3"""
    coordinates=[[0,0,0],[0,0,4]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,4)
    expected=0+P.nr*0+P.nr*P.ntheta*3
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_z_5():
    """Vérifie que la particule de coordonnées en z=5 est dans la partion d'indice 0+P.nr*0+P.nr*P.ntheta*4"""
    coordinates=[[0,0,0],[0,0,5]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,5)
    expected=0+P.nr*0+P.nr*P.ntheta*4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_z_6():
    """Vérifie que la particule de coordonnées en z=6 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 0+P.nr*0+P.nr*P.ntheta*4"""
    coordinates=[[0,0,0],[0,0,6]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,6)
    expected=0+P.nr*0+P.nr*P.ntheta*4
    np.testing.assert_(state,expected)
    
def test_compute_state_coord_z_10():
    """Vérifie que la particule de coordonnées en z=10 qui est supérieure nombre de partitions suivant l'axe de abcisses est dans la partion d'indice 0+P.nr*0+P.nr*P.ntheta*4"""
    coordinates=[[0,0,0],[0,0,10]]
    P=Cyl() # le nombre de partitons par défaut est nr=5,ntheta=8,nz=5
    P=P.fit(coordinates=coordinates)
    state=P.compute_states(0,0,10)
    expected=0+P.nr*0+P.nr*P.ntheta*4
    np.testing.assert_(state,expected)
