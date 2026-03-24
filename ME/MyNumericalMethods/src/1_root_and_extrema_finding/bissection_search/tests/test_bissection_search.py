import numpy as np
import pytest
import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
# from ..implementation import bissection_search
from bissection_search.implementation.bissection_search import bissection_search

def test_bissection_linear():
    f=lambda x:2*x-4
    a,b=0,5
    root=bissection_search(f,a,b)
    expected=2.0
    assert np.isclose(root,expected)

def test_bissection_search():
    f=lambda x:2*x-4
    a,b=0,5
    root=bissection_search(f,a,b)
    excepted=2.0
    assert np.isclose(root,excepted)

def test_bissection_quadratic():
    f=lambda x: x**2 -4
    a,b=0,3
    root=bissection_search(f,a,b)
    excepted=2.0
    assert np.isclose(root,excepted)
    
def test_bissection_sin():
    f=np.sin
    a,b=3,4
    root=bissection_search(f,a,b)
    expected=np.pi
    assert np.isclose(root,expected, atol=1e-5)
def test_bissection_no_root():
    f=lambda x: x**2+1
    a,b=0,1
    with pytest.raises(ValueError):
        bissection_search(f,a,b)


def test_bissection_close_to_root():
    f=lambda x: x**2-2
    a,b=1,2
    root=bissection_search(f,a,b)
    expected=np.sqrt(2)
    assert np.isclose(root,expected)


def test_bissection_negative_interval():
    f=lambda x: x+2
    a,b=-5,0
    root=bissection_search(f,a,b)
    expected=-2
    assert np.isclose(root,expected)
    
def test_bissection_cubic():
    f=lambda x:x**3-2
    a,b=1,2
    root=bissection_search(f,a,b)
    expected=2**(1/3)
    assert np.isclose(root,expected)
    
def test_bissection_exponential()