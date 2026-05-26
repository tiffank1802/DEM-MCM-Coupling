import os
import sys
from pathlib import Path
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pyvista as pv
import streamlit as st
from stpyvista import stpyvista
from src.Markov import Markov as MK
from src.Markov.run_sweep import create_partitioner
# Must happen BEFORE any PyVista rendering call
pv.start_xvfb()
pv.OFF_SCREEN = True

st.title("Markov APP")
st.subheader("Visualisation du partitionnement")

method=st.selectbox(
    'Méthode de découpage',
    [
        'cartesian',
        'cylindrical',
        'voronoi',
        # 'physiccs'
        'octree',
        'single',
        'multizone',
        'quantile',
        'adaptive',
     ],
     placeholder='Selectionner la méthode de découpage'
)
    
# st.selectbox(
#     'Selection de la configuration',
#     []
# )

cyl=MK(f"{method}")




vtp = cyl.get_vtp([250])
cyl.visualize()
# st.dataframe(cyl.compute_matrix()[0])
# states_dem=cyl.compute_dem_rsd(list(range(250, 6000, 50)))
# st.dataframe(states_dem)




