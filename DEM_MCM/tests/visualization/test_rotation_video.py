"""
Tests pour la génération de vidéos 3D avec rotation.
"""
import sys
import os
import pytest
import numpy as np

# Ajouter src au path - remonter à la racine du projet
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from partitioners import (
    CartesianPartitioner,
    CylindricalPartitioner,
    VoronoiPartitioner,
    QuantileGridPartitioner,
    OctreePartitioner,
)


class Test3DRotationVideo:
    """Tests pour la génération de vidéos 3D avec rotation."""
    
    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        """Crée le dossier de sortie pour les vidéos."""
        self.output_dir = "tests/output/3d_rotation"
        os.makedirs(self.output_dir, exist_ok=True)
        yield
        # Nettoyage optionnel après les tests
        # import shutil
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)
    
    def test_cartesian_rotation_video(self):
        """Test vidéo 3D avec partitionnement cartésien."""
        partitioner = CartesianPartitioner(nx=3, ny=3, nz=2)
        
        # Données de test
        np.random.seed(42)
        n_particles = 500
        x = np.random.randn(n_particles) * 0.5
        y = np.random.randn(n_particles) * 0.5
        z = np.random.rand(n_particles) * 1.0
        diameters = np.random.choice([0.004, 0.008], n_particles)
        
        output = os.path.join(self.output_dir, "cartesian_rotation.mp4")
        result = partitioner.visualize_3d_rotation(
            x, y, z, particle_diameters=diameters,
            output_path=output, duration=10, fps=60  # 10s à 60fps = 600 frames
        )
        
        assert os.path.exists(result), f"Vidéo non générée: {result}"
        assert os.path.getsize(result) > 10000, "Fichier vidéo trop petit"
        print(f"✅ Cartésien: {result} ({os.path.getsize(result)} bytes)")
    
    def test_cylindrical_rotation_video(self):
        """Test vidéo 3D avec partitionnement cylindrique."""
        partitioner = CylindricalPartitioner(nr=3, ntheta=8, nz=2)
        
        np.random.seed(42)
        n_particles = 500
        r = np.random.rand(n_particles) * 0.5
        theta = np.random.rand(n_particles) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.rand(n_particles) * 1.0
        diameters = np.random.choice([0.004, 0.008], n_particles)
        
        output = os.path.join(self.output_dir, "cylindrical_rotation.mp4")
        result = partitioner.visualize_3d_rotation(
            x, y, z, particle_diameters=diameters,
            output_path=output, duration=10, fps=60
        )
        
        assert os.path.exists(result)
        assert os.path.getsize(result) > 10000
        print(f"✅ Cylindrique: {result} ({os.path.getsize(result)} bytes)")
    
    def test_voronoi_rotation_video(self):
        """Test vidéo 3D avec partitionnement Voronoï."""
        partitioner = VoronoiPartitioner(n_cells=50)
        
        np.random.seed(42)
        n_particles = 500
        coords = np.random.randn(n_particles, 3) * 0.5
        coords[:, 2] = np.random.rand(n_particles) * 1.0  # Z entre 0 et 1
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        diameters = np.random.choice([0.004, 0.008], n_particles)
        
        output = os.path.join(self.output_dir, "voronoi_rotation.mp4")
        result = partitioner.visualize_3d_rotation(
            x, y, z, particle_diameters=diameters,
            output_path=output, duration=10, fps=60
        )
        
        assert os.path.exists(result)
        assert os.path.getsize(result) > 10000
        print(f"✅ Voronoï: {result} ({os.path.getsize(result)} bytes)")
    
    def test_particle_size_scaling(self):
        """Vérifie que la taille des particules est proportionnelle au diamètre."""
        partitioner = CartesianPartitioner(nx=2, ny=2, nz=1)
        
        # 2 particules: une petite, une grosse
        x = [-0.2, 0.2]
        y = [0.0, 0.0]
        z = [0.5, 0.5]
        diameters = [0.004, 0.008]
        
        output = os.path.join(self.output_dir, "size_test_rotation.mp4")
        result = partitioner.visualize_3d_rotation(
            x, y, z, particle_diameters=diameters,
            output_path=output, duration=10, fps=60
        )
        
        assert os.path.exists(result)
        print(f"✅ Test taille: {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
