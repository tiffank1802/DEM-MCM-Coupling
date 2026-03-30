"""
===================================================================================
VISUALISATION CAD 3D DU MÉLANGEUR — PyVista + Trame
===================================================================================

Rendu web interactif avec:
- Mélangeur semi-transparent (style CAO)
- Frontières de partitions nettes et colorées
- Particules colorées par cellule
- Contrôles: méthode, transparence, taille, coupes

Usage:
    python mixer_cad_app.py
    → Ouvrir http://localhost:8080

===================================================================================
"""
"""
mixer_cad_app.py — Version headless compatible serveur
"""

import os
# ── Forcer le rendu off-screen AVANT tout import PyVista ──
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_TRAME_SERVER_PROXY_PREFIX"] = "/proxy/8080"

import numpy as np
import polars as pl
import pyvista as pv
from huggingface_hub import HfFileSystem
from partitioners import create_partitioner

# ── Config PyVista headless ──
pv.OFF_SCREEN = True
pv.start_xvfb()  # Lance Xvfb automatiquement si disponible

pv.global_theme.background = "white"
pv.global_theme.font.color = "black"

# ... le reste du code identique ...
pv.global_theme.anti_aliasing = "ssaa"

HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"

# Couleurs pastel pour les cellules
CELL_COLORS = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
    "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a",
    "#ffff99", "#b15928", "#8dd3c7", "#bebada", "#fb8072",
    "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9",
]


class MixerCADApp:
    """Application web de visualisation CAD du mélangeur."""

    def __init__(self):
        self.fs = HfFileSystem()
        self.files = sorted(self.fs.glob(f"{HF_FOLDER}/*.csv"))
        self.coords = None
        self.bounds = None
        self.center = None
        self.scenes = {}  # {method_name: {meshes, partitioner}}
        print(f"📁 {len(self.files)} fichiers DEM disponibles")

    # ═══════════════════════════════════════════════════════════════
    # CHARGEMENT
    # ═══════════════════════════════════════════════════════════════

    def load_particles(self, file_index=100):
        """Charge un snapshot de particules."""
        fname = self.files[file_index]
        print(f"📂 {fname.split('/')[-1]}")

        with self.fs.open(fname, "rb") as f:
            df = pl.read_csv(f)

        self.coords = np.column_stack([
            df["coordinates:0"].to_numpy(),
            df["coordinates:1"].to_numpy(),
            df["coordinates:2"].to_numpy(),
        ])

        eps = 0.001
        self.bounds = [
            self.coords[:, 0].min() - eps, self.coords[:, 0].max() + eps,
            self.coords[:, 1].min() - eps, self.coords[:, 1].max() + eps,
            self.coords[:, 2].min() - eps, self.coords[:, 2].max() + eps,
        ]
        self.center = np.array([
            (self.bounds[0] + self.bounds[1]) / 2,
            (self.bounds[2] + self.bounds[3]) / 2,
            (self.bounds[4] + self.bounds[5]) / 2,
        ])

        print(f"   {len(self.coords)} particules chargées")

    # ═══════════════════════════════════════════════════════════════
    # GÉOMÉTRIE DU MÉLANGEUR (enveloppe)
    # ═══════════════════════════════════════════════════════════════

    def make_mixer_shell(self):
        """Cylindre englobant le mélangeur."""
        xc, yc = self.center[0], self.center[1]
        r = np.sqrt(
            (self.coords[:, 0] - xc) ** 2
            + (self.coords[:, 1] - yc) ** 2
        ).max() * 1.08

        h = (self.bounds[5] - self.bounds[4]) * 1.1
        zc = self.center[2]

        cyl = pv.Cylinder(
            center=(xc, yc, zc),
            direction=(0, 0, 1),
            radius=r,
            height=h,
            resolution=80,
            capping=True,
        )
        return cyl

    # ═══════════════════════════════════════════════════════════════
    # FRONTIÈRES DE PARTITIONS
    # ═══════════════════════════════════════════════════════════════

    def _planes_from_edges(self, x_edges, y_edges, z_edges):
        """Crée des plans semi-transparents aux positions des bords."""
        xmin, xmax = x_edges[0], x_edges[-1]
        ymin, ymax = y_edges[0], y_edges[-1]
        zmin, zmax = z_edges[0], z_edges[-1]

        planes = []

        for x in x_edges[1:-1]:
            planes.append(pv.Plane(
                center=(x, (ymin + ymax) / 2, (zmin + zmax) / 2),
                direction=(1, 0, 0),
                i_size=ymax - ymin, j_size=zmax - zmin,
            ))

        for y in y_edges[1:-1]:
            planes.append(pv.Plane(
                center=((xmin + xmax) / 2, y, (zmin + zmax) / 2),
                direction=(0, 1, 0),
                i_size=xmax - xmin, j_size=zmax - zmin,
            ))

        for z in z_edges[1:-1]:
            planes.append(pv.Plane(
                center=((xmin + xmax) / 2, (ymin + ymax) / 2, z),
                direction=(0, 0, 1),
                i_size=xmax - xmin, j_size=ymax - ymin,
            ))

        return planes

    def make_cartesian_boundaries(self, nx=5, ny=5, nz=5):
        """Plans réguliers."""
        x_e = np.linspace(self.bounds[0], self.bounds[1], nx + 1)
        y_e = np.linspace(self.bounds[2], self.bounds[3], ny + 1)
        z_e = np.linspace(self.bounds[4], self.bounds[5], nz + 1)
        return self._planes_from_edges(x_e, y_e, z_e)

    def make_quantile_boundaries(self, part):
        """Plans aux quantiles."""
        return self._planes_from_edges(
            part._x_edges, part._y_edges, part._z_edges
        )

    def make_cylindrical_boundaries(self, nr=5, ntheta=8, nz=5,
                                     radial_mode="equal_area"):
        """Cylindres concentriques + secteurs + plans horizontaux."""
        xc, yc = self.center[0], self.center[1]
        zmin, zmax = self.bounds[4], self.bounds[5]

        r_max = np.sqrt(
            (self.coords[:, 0] - xc) ** 2
            + (self.coords[:, 1] - yc) ** 2
        ).max() + 0.001

        if radial_mode == "equal_area":
            r_edges = r_max * np.sqrt(np.linspace(0, 1, nr + 1))
        else:
            r_edges = np.linspace(0, r_max, nr + 1)

        z_edges = np.linspace(zmin, zmax, nz + 1)
        theta_edges = np.linspace(0, 2 * np.pi, ntheta + 1)

        meshes = []
        n_res = 80

        # ── Cylindres concentriques (intérieurs) ──
        for r in r_edges[1:-1]:
            theta = np.linspace(0, 2 * np.pi, n_res + 1)
            pts_bot = np.column_stack([
                xc + r * np.cos(theta),
                yc + r * np.sin(theta),
                np.full_like(theta, zmin),
            ])
            pts_top = np.column_stack([
                xc + r * np.cos(theta),
                yc + r * np.sin(theta),
                np.full_like(theta, zmax),
            ])
            pts = np.vstack([pts_bot, pts_top])
            n = len(theta)
            faces = []
            for i in range(n - 1):
                faces.extend([4, i, i + 1, n + i + 1, n + i])
            meshes.append(pv.PolyData(pts, faces=faces))

        # ── Secteurs angulaires (plans radiaux verticaux) ──
        for t in theta_edges[:-1]:
            p = np.array([
                [xc, yc, zmin],
                [xc + r_max * np.cos(t), yc + r_max * np.sin(t), zmin],
                [xc + r_max * np.cos(t), yc + r_max * np.sin(t), zmax],
                [xc, yc, zmax],
            ])
            meshes.append(pv.PolyData(p, faces=[4, 0, 1, 2, 3]))

        # ── Plans horizontaux (intérieurs) ──
        for z in z_edges[1:-1]:
            theta = np.linspace(0, 2 * np.pi, n_res + 1)
            center_pt = np.array([[xc, yc, z]])
            rim = np.column_stack([
                xc + r_max * np.cos(theta),
                yc + r_max * np.sin(theta),
                np.full_like(theta, z),
            ])
            pts = np.vstack([center_pt, rim])
            faces = []
            for i in range(len(theta) - 1):
                faces.extend([3, 0, i + 1, i + 2])
            meshes.append(pv.PolyData(pts, faces=faces))

        return meshes

    def make_voronoi_boundaries(self, part, resolution=50):
        """
        Frontières Voronoï par voxélisation.

        1. Grille 3D fine
        2. Chaque voxel reçoit le label de son centroïde le plus proche
        3. Les faces entre voxels de labels différents = frontières
        """
        grid = pv.ImageData(
            dimensions=(resolution + 1, resolution + 1, resolution + 1),
            spacing=(
                (self.bounds[1] - self.bounds[0]) / resolution,
                (self.bounds[3] - self.bounds[2]) / resolution,
                (self.bounds[5] - self.bounds[4]) / resolution,
            ),
            origin=(self.bounds[0], self.bounds[2], self.bounds[4]),
        )

        centers = grid.cell_centers().points
        labels = part.compute_states(
            centers[:, 0], centers[:, 1], centers[:, 2]
        )
        grid.cell_data["partition"] = labels.astype(float)

        return grid

    def make_octree_boundaries(self, part):
        """Boîtes wireframe pour chaque feuille de l'octree."""
        boxes = []
        for leaf in part._leaves:
            xmin, xmax, ymin, ymax, zmin, zmax = leaf
            box = pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
            boxes.append(box)
        return boxes

    # ═══════════════════════════════════════════════════════════════
    # NUAGE DE PARTICULES
    # ═══════════════════════════════════════════════════════════════

    def make_particle_cloud(self, partitioner):
        """Nuage de points coloré par partition."""
        states = partitioner.compute_states(
            self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        )
        cloud = pv.PolyData(self.coords)
        cloud["cell_id"] = states.astype(float)
        return cloud

    # ═══════════════════════════════════════════════════════════════
    # PRÉ-CALCUL DE TOUTES LES SCÈNES
    # ═══════════════════════════════════════════════════════════════

    def precompute_all(self):
        """Pré-calcule les meshes pour chaque méthode."""
        print("\n🔧 Pré-calcul des scènes...")

        configs = {
            "Cartésien (5³=125)": {
                "method": "cartesian",
                "kwargs": {"nx": 5, "ny": 5, "nz": 5},
                "type": "planes",
            },
            "Cylindrique equal_area": {
                "method": "cylindrical",
                "kwargs": {"nr": 5, "ntheta": 8, "nz": 5, "radial_mode": "equal_area"},
                "type": "cylindrical",
            },
            "Cylindrique equal_dr": {
                "method": "cylindrical",
                "kwargs": {"nr": 5, "ntheta": 8, "nz": 5, "radial_mode": "equal_dr"},
                "type": "cylindrical",
            },
            "Voronoï (64)": {
                "method": "voronoi",
                "kwargs": {"n_cells": 64},
                "type": "voronoi",
            },
            "Voronoï (125)": {
                "method": "voronoi",
                "kwargs": {"n_cells": 125},
                "type": "voronoi",
            },
            "Quantile (5³=125)": {
                "method": "quantile",
                "kwargs": {"nx": 5, "ny": 5, "nz": 5},
                "type": "quantile",
            },
            "Octree (mp=50, d=4)": {
                "method": "octree",
                "kwargs": {"max_particles": 50, "max_depth": 4},
                "type": "octree",
            },
        }

        for name, cfg in configs.items():
            print(f"   • {name}...")

            part = create_partitioner(cfg["method"], **cfg["kwargs"])
            part.fit(self.coords)

            # Particules
            cloud = self.make_particle_cloud(part)

            # Frontières
            if cfg["type"] == "planes":
                boundaries = self.make_cartesian_boundaries(**cfg["kwargs"])
            elif cfg["type"] == "cylindrical":
                boundaries = self.make_cylindrical_boundaries(**cfg["kwargs"])
            elif cfg["type"] == "voronoi":
                boundaries = self.make_voronoi_boundaries(part, resolution=40)
            elif cfg["type"] == "quantile":
                boundaries = self.make_quantile_boundaries(part)
            elif cfg["type"] == "octree":
                boundaries = self.make_octree_boundaries(part)
            else:
                boundaries = []

            diag = part.diagnostics(self.coords)

            self.scenes[name] = {
                "partitioner": part,
                "cloud": cloud,
                "boundaries": boundaries,
                "type": cfg["type"],
                "n_cells": part.n_cells,
                "n_visited": diag["n_visited"],
                "pop_cv": diag["pop_std"] / max(diag["pop_mean"], 1),
            }

        print(f"✅ {len(self.scenes)} scènes prêtes")

    # ═══════════════════════════════════════════════════════════════
    # RENDU D'UNE SCÈNE DANS UN PLOTTER
    # ═══════════════════════════════════════════════════════════════

    def render_scene(self, pl, scene_name,
                     boundary_opacity=0.12,
                     boundary_edge_opacity=0.5,
                     particle_opacity=0.95,
                     particle_size=6,
                     mixer_opacity=0.06,
                     show_boundaries=True,
                     show_particles=True,
                     show_mixer=True):
        """
        Ajoute une scène au plotter PyVista.

        Args:
            pl: pv.Plotter
            scene_name: clé dans self.scenes
            boundary_opacity: opacité des surfaces de frontière
            boundary_edge_opacity: opacité des arêtes
            particle_opacity: opacité des particules
            particle_size: taille des particules
            mixer_opacity: opacité de l'enveloppe
            show_*: toggles
        """
        scene = self.scenes[scene_name]
        cloud = scene["cloud"]
        boundaries = scene["boundaries"]
        btype = scene["type"]
        n_cells = scene["n_cells"]

        # ── Enveloppe du mélangeur ──
        if show_mixer:
            mixer = self.make_mixer_shell()
            pl.add_mesh(
                mixer,
                color="#b0b0b0",
                opacity=mixer_opacity,
                smooth_shading=True,
                specular=0.5,
                specular_power=30,
            )
            pl.add_mesh(
                mixer,
                style="wireframe",
                color="#888888",
                opacity=mixer_opacity * 2,
                line_width=0.5,
            )

        # ── Frontières de partition ──
        if show_boundaries:
            if btype in ("planes", "quantile"):
                for plane in boundaries:
                    # Surface semi-transparente
                    pl.add_mesh(
                        plane,
                        color="#4a90d9",
                        opacity=boundary_opacity,
                        smooth_shading=True,
                    )
                    # Arêtes
                    pl.add_mesh(
                        plane,
                        style="wireframe",
                        color="#1a3a6a",
                        opacity=boundary_edge_opacity,
                        line_width=1.5,
                    )

            elif btype == "cylindrical":
                for mesh in boundaries:
                    pl.add_mesh(
                        mesh,
                        color="#d9944a",
                        opacity=boundary_opacity,
                        smooth_shading=True,
                    )
                    pl.add_mesh(
                        mesh,
                        style="wireframe",
                        color="#6a3a1a",
                        opacity=boundary_edge_opacity,
                        line_width=1.0,
                    )

            elif btype == "voronoi":
                # Grid voxélisé avec couleurs par partition
                pl.add_mesh(
                    boundaries,
                    scalars="partition",
                    cmap="tab20",
                    opacity=boundary_opacity * 1.5,
                    show_scalar_bar=False,
                    show_edges=True,
                    edge_color="#333333",
                    edge_opacity=boundary_edge_opacity * 0.3,
                    line_width=0.3,
                )

                # Centroïdes
                part = scene["partitioner"]
                if hasattr(part, "centroids") and part.centroids is not None:
                    cents = pv.PolyData(part.centroids)
                    pl.add_mesh(
                        cents,
                        color="red",
                        point_size=10,
                        render_points_as_spheres=True,
                        opacity=0.9,
                    )

            elif btype == "octree":
                for box in boundaries:
                    pl.add_mesh(
                        box,
                        color="#9467bd",
                        opacity=boundary_opacity * 0.5,
                        smooth_shading=True,
                    )
                    pl.add_mesh(
                        box,
                        style="wireframe",
                        color="#4a2d6a",
                        opacity=boundary_edge_opacity,
                        line_width=1.5,
                    )

        # ── Particules ──
        if show_particles:
            pl.add_mesh(
                cloud,
                scalars="cell_id",
                cmap="tab20",
                clim=[0, max(n_cells - 1, 1)],
                point_size=particle_size,
                render_points_as_spheres=True,
                opacity=particle_opacity,
                show_scalar_bar=False,
                ambient=0.3,
                diffuse=0.6,
                specular=0.3,
            )

        # ── Texte info ──
        info = (
            f"{scene_name}\n"
            f"{n_cells} cellules | {scene['n_visited']} visitées\n"
            f"CV={scene['pop_cv']:.2f}"
        )
        pl.add_text(info, font_size=10, position="upper_left",
                    color="black", shadow=True)

    # ═══════════════════════════════════════════════════════════════
    # RENDU STATIQUE (SANS TRAME)
    # ═══════════════════════════════════════════════════════════════

    def render_comparison_static(self, figsize=(2400, 1800)):
        """Rendu statique : grille 2×3 ou 2×4 de toutes les méthodes."""
        names = list(self.scenes.keys())
        n = len(names)
        cols = 3
        rows = (n + cols - 1) // cols

        pl = pv.Plotter(
            shape=(rows, cols),
            window_size=figsize,
            off_screen=True,
        )

        for i, name in enumerate(names):
            r, c = divmod(i, cols)
            pl.subplot(r, c)
            self.render_scene(pl, name, particle_size=4)
            pl.camera.azimuth = 45
            pl.camera.elevation = 25

        pl.screenshot("cad_comparison.png")
        print("✅ Sauvegardé: cad_comparison.png")
        pl.close()

    # ═══════════════════════════════════════════════════════════════
    # APPLICATION WEB TRAME
    # ═══════════════════════════════════════════════════════════════

    def serve(self, port=8080):
        """Lance l'application web interactive."""
        from trame.app import get_server
        from trame.ui.vuetify3 import SinglePageLayout
        from trame.widgets import vtk as vtk_widgets
        from trame.widgets import vuetify3 as v3

        server = get_server(client_type="vue3")
        state, ctrl = server.state, server.controller

        # ── État initial ──
        method_names = list(self.scenes.keys())
        state.methods = method_names
        state.active_method = method_names[0]
        state.boundary_opacity = 12       # sur 100
        state.edge_opacity = 50           # sur 100
        state.particle_opacity = 95       # sur 100
        state.particle_size = 6
        state.mixer_opacity = 6           # sur 100
        state.show_boundaries = True
        state.show_particles = True
        state.show_mixer = True

        # ── Plotter PyVista ──
        pl = pv.Plotter(off_screen=True)

        def rebuild_scene(**kwargs):
            """Reconstruit la scène quand un paramètre change."""
            pl.clear()

            self.render_scene(
                pl,
                state.active_method,
                boundary_opacity=state.boundary_opacity / 100,
                boundary_edge_opacity=state.edge_opacity / 100,
                particle_opacity=state.particle_opacity / 100,
                particle_size=state.particle_size,
                mixer_opacity=state.mixer_opacity / 100,
                show_boundaries=state.show_boundaries,
                show_particles=state.show_particles,
                show_mixer=state.show_mixer,
            )

            pl.camera.azimuth = 45
            pl.camera.elevation = 25
            pl.reset_camera()
            ctrl.view_update()

        # Déclencher le rebuild quand l'état change
        state.change("active_method")(rebuild_scene)
        state.change("boundary_opacity")(rebuild_scene)
        state.change("edge_opacity")(rebuild_scene)
        state.change("particle_opacity")(rebuild_scene)
        state.change("particle_size")(rebuild_scene)
        state.change("mixer_opacity")(rebuild_scene)
        state.change("show_boundaries")(rebuild_scene)
        state.change("show_particles")(rebuild_scene)
        state.change("show_mixer")(rebuild_scene)

        # ── Interface utilisateur ──
        with SinglePageLayout(server) as layout:
            layout.title.set_text("🔬 Mélangeur DEM — Partitionnement 3D")

            # Barre d'outils
            with layout.toolbar:
                v3.VSpacer()

                v3.VSelect(
                    v_model=("active_method",),
                    items=("methods",),
                    label="Méthode",
                    density="compact",
                    hide_details=True,
                    style="max-width: 280px;",
                )

                v3.VDivider(vertical=True, classes="mx-2")

                v3.VCheckbox(
                    v_model=("show_particles",),
                    label="Particules",
                    density="compact",
                    hide_details=True,
                )
                v3.VCheckbox(
                    v_model=("show_boundaries",),
                    label="Frontières",
                    density="compact",
                    hide_details=True,
                )
                v3.VCheckbox(
                    v_model=("show_mixer",),
                    label="Mélangeur",
                    density="compact",
                    hide_details=True,
                )

            # Panneau latéral
            with layout.drawer as drawer:
                drawer.width = 300

                with v3.VCard(classes="ma-2", variant="outlined"):
                    v3.VCardTitle("Apparence", classes="text-subtitle-1")
                    with v3.VCardText():
                        v3.VSlider(
                            v_model=("particle_size",),
                            label="Taille particules",
                            min=1, max=15, step=1,
                            thumb_label="always",
                            density="compact",
                        )
                        v3.VSlider(
                            v_model=("particle_opacity",),
                            label="Opacité particules",
                            min=0, max=100, step=5,
                            thumb_label="always",
                            density="compact",
                        )
                        v3.VSlider(
                            v_model=("boundary_opacity",),
                            label="Opacité frontières",
                            min=0, max=50, step=1,
                            thumb_label="always",
                            density="compact",
                        )
                        v3.VSlider(
                            v_model=("edge_opacity",),
                            label="Opacité arêtes",
                            min=0, max=100, step=5,
                            thumb_label="always",
                            density="compact",
                        )
                        v3.VSlider(
                            v_model=("mixer_opacity",),
                            label="Opacité mélangeur",
                            min=0, max=30, step=1,
                            thumb_label="always",
                            density="compact",
                        )

                # Info sur la méthode active
                with v3.VCard(classes="ma-2", variant="outlined"):
                    v3.VCardTitle("Informations", classes="text-subtitle-1")
                    with v3.VCardText():
                        v3.VAlert(
                            type="info",
                            density="compact",
                            text=(
                                "`active_method + ' — '"
                                " + scenes[active_method]?.n_cells + ' cellules'"
                            ),
                        )

            # Vue 3D
            with layout.content:
                with v3.VContainer(fluid=True, classes="fill-height pa-0"):
                    view = vtk_widgets.VtkRemoteView(
                        pl.ren_win,
                        interactive_ratio=1,
                    )
                    ctrl.view_update = view.update

        # ── Scène initiale ──
        rebuild_scene()

        # ── Lancement ──
        print(f"\n{'='*60}")
        print(f"🌐 Application disponible sur:")
        print(f"   http://localhost:{port}")
        print(f"   http://0.0.0.0:{port}")
        print(f"{'='*60}")

        server.start(port=port, open_browser=False)


# ═══════════════════════════════════════════════════════════════════
# SCRIPT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = MixerCADApp()

    # Charger les particules
    app.load_particles(file_index=100)

    # Pré-calculer toutes les scènes
    app.precompute_all()

    # Option 1: Rendu statique (image PNG)
    app.render_comparison_static()

    # Option 2: Application web interactive
    app.serve(port=8080)