import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
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


# cyl = MK("voronoi")
# 'Cas du découpage en cellules de voronoï'
# cyl.indices=[300]
# f'{cyl.config}'
# vtp = cyl.get_vtp([250])
# cyl.visualize()
# st.dataframe(cyl.P_matrix()[0])
# np.savetxt('P.txt',cyl.P_matrix()[0])
# st.dataframe(cyl.propagate())
# cyl.default_configs
# cyl.config=cyl.default_configs[-3]
# cyl.config
# cyl.visualize()
# cyl.config=cyl.default_configs[-4]
# cyl.config
# cyl.visualize()
# cyl1=MK("cylindrical")
# 'Cylindrical'
# cyl1.visualize()
# fig,ax=plt.subplots()
# s_hist=cyl.S_history
# ax.plot(s_hist[0],label="S0")
# ax.plot(s_hist[-1],label="S-1")
# ax.plot(s_hist[-2],label="S-2")
# ax.legend()
# st.pyplot(fig)

phy=MK("physics")
phy.partitioner.use_velocity=True
phy.config=phy.default_configs[6]
phy.visualize()
st.dataframe(phy.propagate())
phy2=MK("physics")
phy2.config=phy2.default_configs[4]
phy2.partitioner.use_velocity=True
phy2.visualize()
st.dataframe(phy2.propagate())
phy2.default_configs
phy2.partitioner.dem_velocities
# st.dataframe(np.max([i['Diameter'].to_numpy() for i in phy2.datas.values()]))
