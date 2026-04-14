import pytest
from DEM_MCM.src.analyze_results import MarkovAnalyzer
import numpy as np


def test_len_snapshots():
    
    analyzer=MarkovAnalyzer()
    snapshots=analyzer.load_dem_snapshots()

    
    expected=500
    np.testing.assert_(expected,len(snapshots))
    