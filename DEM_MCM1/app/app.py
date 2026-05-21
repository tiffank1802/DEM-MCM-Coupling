import os
import sys
from pathlib import Path

# On trouve le chemin de la racine (DEM_MCM1) et on l'ajoute à Python
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
from src.Markov import Markov as MK

st.title("Markov modele")
cyl=MK()
cyl.visualize()
