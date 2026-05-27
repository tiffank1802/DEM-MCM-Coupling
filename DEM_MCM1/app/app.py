import os
import sys
from pathlib import Path
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pyvista as pv
import numpy as np
import streamlit as st
from stpyvista import stpyvista
from src.Markov import Markov as MK
# Must happen BEFORE any PyVista rendering call
pv.start_xvfb()
pv.OFF_SCREEN = True

st.title("Markov APP")
st.subheader("Visualisation du partitionnement")


cyl = MK("voronoi")
cyl.indices=[300]

vtp = cyl.get_vtp([250])
cyl.visualize()
st.dataframe(cyl.P_matrix()[0])
np.savetxt('P.txt',cyl.P_matrix()[0])
st.dataframe(cyl.propagate())
cyl.default_config