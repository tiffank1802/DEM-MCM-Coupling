import pytest
import numpy as np
import os
import sys

# Configuration du path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from partitioners import PhysicsAwarePartitioner as Phy
from run_sweep import run_markov_sweep, get_configs,ExperimentConfig

class TestPhysics:
    
    @pytest.fixture
    def physics_engine(self):
        """Fixture pour initialiser l'objet Physics et charger les données."""
        engine = Phy(n_cells=30)
        # On charge les snapshots
        engine.load_dem_snapshots(file_indices=[250])
        return engine

    def test_call_compute_state_with_physics(self, mocker, physics_engine):
        # 1. On prépare l'espion sur la méthode de l'instance
        # ATTENTION: On espionne AVANT l'exécution qui est censée l'appeler
        spy = mocker.spy(physics_engine, "compute_states_with_physics")
        
        # 2. On récupère les coordonnées depuis l'objet (chargées via la fixture)
        # Note: j'adapte selon la structure probable de ton objet dem_snapshots
        coords = physics_engine.dem_snapshots[0].get("coords", np.random.randn(100, 3))
        
        # 3. On entraîne le partitionneur
        physics_engine.fit(coords)
        
        # 4. On lance le sweep (qui est censé appeler la méthode en interne)
        # Note: Assure-toi que run_markov_sweep utilise bien l'instance 'physics_engine'
        configs=[ExperimentConfig(
                    method="physics",
                    method_kwargs={"n_cells": 30, "velocity_weight": 0.5},
                )]
        run_markov_sweep("physics", configs)
        
        # 5. Vérification
        # Si run_markov_sweep appelle la méthode, l'assertion passera
        assert spy.called
        # Ou plus précis :
        # spy.assert_called_once()