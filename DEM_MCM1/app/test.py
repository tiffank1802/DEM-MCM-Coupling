import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import streamlit as st
import pyvista as pv
from stpyvista import stpyvista

mesh = pv.ImageData(dimensions=(300, 300, 1))
mesh2=pv.Sphere()
pl = pv.Plotter()
# pl.add_mesh(mesh)
pl.add_mesh(mesh2)
stpyvista(pl)