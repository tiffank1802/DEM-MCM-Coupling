"""
mixer_cad_plotly.py — Visualisation CAO style Plotly (pas besoin d'X11)
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import HfFileSystem
from partitioners import create_partitioner
import json

HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"


class MixerCADPlotly:
    """Visualisation CAD du mélangeur avec Plotly — compatible headless."""

    def __init__(self):
        self.fs = HfFileSystem()
        self.files = sorted(self.fs.glob(f"{HF_FOLDER}/*.csv"))
        self.coords = None
        self.center = None
        self.r_max = None
        self.zmin = None
        self.zmax = None
        print(f"📁 {len(self.files)} fichiers DEM disponibles")

    def load_particles(self, file_index=100):
        """Charge un snapshot."""
        fname = self.files[file_index]
        print(f"📂 {fname.split('/')[-1]}")

        with self.fs.open(fname, "rb") as f:
            df = pl.read_csv(f)

        self.coords = np.column_stack([
            df["coordinates:0"].to_numpy(),
            df["coordinates:1"].to_numpy(),
            df["coordinates:2"].to_numpy(),
        ])

        eps = 0.002
        self.center = self.coords.mean(axis=0)
        r = np.sqrt(
            (self.coords[:, 0] - self.center[0]) ** 2 +
            (self.coords[:, 1] - self.center[1]) ** 2
        )
        self.r_max = r.max() + eps
        self.zmin = self.coords[:, 2].min() - eps
        self.zmax = self.coords[:, 2].max() + eps

        print(f"   {len(self.coords)} particules")

    # ─────────────────────────────────────────────────────────────
    # ENVELOPPE DU MÉLANGEUR (cylindre transparent)
    # ─────────────────────────────────────────────────────────────

    def _make_cylinder_mesh(self, r, n=60):
        """Cylindre semi-transparent."""
        theta = np.linspace(0, 2 * np.pi, n + 1)
        xc, yc = self.center[0], self.center[1]
        z0, z1 = self.zmin, self.zmax

        # Surface latérale
        x_lat = np.outer(xc + r * np.cos(theta), np.ones(2))
        y_lat = np.outer(yc + r * np.sin(theta), np.ones(2))
        z_lat = np.outer(np.ones(n + 1), np.array([z0, z1]))

        return go.Surface(
            x=x_lat, y=y_lat, z=z_lat,
            colorscale=[[0, "rgba(180,180,190,0.05)"], [1, "rgba(180,180,190,0.05)"]],
            showscale=False,
            hoverinfo="skip",
            name="Mélangeur",
        )

    def _make_disk(self, z, r, n=60, color="rgba(180,180,190,0.04)"):
        """Disque (couvercle haut/bas)."""
        theta = np.linspace(0, 2 * np.pi, n + 1)
        xc, yc = self.center[0], self.center[1]
        r_vals = np.linspace(0, r, 10)

        x = np.outer(xc + np.cos(theta), r_vals)
        y = np.outer(yc + np.sin(theta), r_vals)
        z_vals = np.full_like(x, z)

        return go.Surface(
            x=x, y=y, z=z_vals,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            hoverinfo="skip",
        )

    # ─────────────────────────────────────────────────────────────
    # FRONTIÈRES DE PARTITIONS
    # ─────────────────────────────────────────────────────────────

    def _cartesian_boundaries(self, nx, ny, nz):
        """Plans cartésiens comme des surfaces semi-transparentes."""
        traces = []
        xmin, xmax = self.coords[:, 0].min(), self.coords[:, 0].max()
        ymin, ymax = self.coords[:, 1].min(), self.coords[:, 1].max()
        zmin, zmax = self.zmin, self.zmax

        x_e = np.linspace(xmin, xmax, nx + 1)
        y_e = np.linspace(ymin, ymax, ny + 1)
        z_e = np.linspace(zmin, zmax, nz + 1)

        color = "rgba(70, 130, 220, 0.08)"

        # Plans perpendiculaires à X
        for x in x_e[1:-1]:
            yy, zz = np.meshgrid(
                np.linspace(ymin, ymax, 5),
                np.linspace(zmin, zmax, 5),
            )
            xx = np.full_like(yy, x)
            traces.append(go.Surface(
                x=xx, y=yy, z=zz,
                colorscale=[[0, color], [1, color]],
                showscale=False, hoverinfo="skip",
            ))

        # Plans perpendiculaires à Y
        for y in y_e[1:-1]:
            xx, zz = np.meshgrid(
                np.linspace(xmin, xmax, 5),
                np.linspace(zmin, zmax, 5),
            )
            yy = np.full_like(xx, y)
            traces.append(go.Surface(
                x=xx, y=yy, z=zz,
                colorscale=[[0, color], [1, color]],
                showscale=False, hoverinfo="skip",
            ))

        # Plans perpendiculaires à Z
        for z in z_e[1:-1]:
            xx, yy = np.meshgrid(
                np.linspace(xmin, xmax, 5),
                np.linspace(ymin, ymax, 5),
            )
            zz = np.full_like(xx, z)
            traces.append(go.Surface(
                x=xx, y=yy, z=zz,
                colorscale=[[0, color], [1, color]],
                showscale=False, hoverinfo="skip",
            ))

        # Arêtes des cellules comme des lignes
        for x in x_e:
            for y in y_e:
                traces.append(go.Scatter3d(
                    x=[x, x], y=[y, y], z=[zmin, zmax],
                    mode="lines",
                    line=dict(color="rgba(50,50,100,0.3)", width=1),
                    showlegend=False, hoverinfo="skip",
                ))
        for x in x_e:
            for z in z_e:
                traces.append(go.Scatter3d(
                    x=[x, x], y=[ymin, ymax], z=[z, z],
                    mode="lines",
                    line=dict(color="rgba(50,50,100,0.3)", width=1),
                    showlegend=False, hoverinfo="skip",
                ))
        for y in y_e:
            for z in z_e:
                traces.append(go.Scatter3d(
                    x=[xmin, xmax], y=[y, y], z=[z, z],
                    mode="lines",
                    line=dict(color="rgba(50,50,100,0.3)", width=1),
                    showlegend=False, hoverinfo="skip",
                ))

        return traces

    def _cylindrical_boundaries(self, nr, ntheta, nz, radial_mode):
        """Cylindres concentriques + secteurs + plans Z."""
        traces = []
        xc, yc = self.center[0], self.center[1]
        zmin, zmax = self.zmin, self.zmax

        if radial_mode == "equal_area":
            r_edges = self.r_max * np.sqrt(np.linspace(0, 1, nr + 1))
        else:
            r_edges = np.linspace(0, self.r_max, nr + 1)

        z_edges = np.linspace(zmin, zmax, nz + 1)
        theta_edges = np.linspace(0, 2 * np.pi, ntheta + 1)
        n_res = 60

        # Cercles concentriques
        for r in r_edges[1:]:
            for z in z_edges:
                theta = np.linspace(0, 2 * np.pi, n_res)
                traces.append(go.Scatter3d(
                    x=xc + r * np.cos(theta),
                    y=yc + r * np.sin(theta),
                    z=np.full(n_res, z),
                    mode="lines",
                    line=dict(color="rgba(180,100,30,0.5)", width=2),
                    showlegend=False, hoverinfo="skip",
                ))

        # Secteurs
        for t in theta_edges[:-1]:
            for z in z_edges:
                traces.append(go.Scatter3d(
                    x=[xc, xc + self.r_max * np.cos(t)],
                    y=[yc, yc + self.r_max * np.sin(t)],
                    z=[z, z],
                    mode="lines",
                    line=dict(color="rgba(180,100,30,0.4)", width=1.5),
                    showlegend=False, hoverinfo="skip",
                ))

        # Lignes verticales
        for t in theta_edges[:-1]:
            traces.append(go.Scatter3d(
                x=[xc + self.r_max * np.cos(t)] * 2,
                y=[yc + self.r_max * np.sin(t)] * 2,
                z=[zmin, zmax],
                mode="lines",
                line=dict(color="rgba(180,100,30,0.3)", width=1),
                showlegend=False, hoverinfo="skip",
            ))

        # Cylindres comme surfaces (intérieurs)
        for r in r_edges[1:-1]:
            theta = np.linspace(0, 2 * np.pi, n_res + 1)
            x_cyl = np.outer(xc + r * np.cos(theta), np.ones(2))
            y_cyl = np.outer(yc + r * np.sin(theta), np.ones(2))
            z_cyl = np.outer(np.ones(n_res + 1), np.array([zmin, zmax]))
            traces.append(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl,
                colorscale=[[0, "rgba(210,150,70,0.06)"], [1, "rgba(210,150,70,0.06)"]],
                showscale=False, hoverinfo="skip",
            ))

        return traces

    def _voronoi_boundaries(self, part):
        """Centroïdes + lignes entre voisins."""
        traces = []

        if hasattr(part, "centroids") and part.centroids is not None:
            c = part.centroids

            # Centroïdes
            traces.append(go.Scatter3d(
                x=c[:, 0], y=c[:, 1], z=c[:, 2],
                mode="markers",
                marker=dict(size=5, color="red", symbol="diamond",
                            line=dict(width=1, color="darkred")),
                name="Centroïdes",
                hovertemplate="Centroïde %{pointNumber}<br>"
                              "(%{x:.4f}, %{y:.4f}, %{z:.4f})<extra></extra>",
            ))

            # Lignes entre centroïdes voisins (Delaunay approximatif via distance)
            from scipy.spatial import Delaunay
            try:
                tri = Delaunay(c)
                edges_set = set()
                for simplex in tri.simplices:
                    for i in range(len(simplex)):
                        for j in range(i + 1, len(simplex)):
                            e = tuple(sorted([simplex[i], simplex[j]]))
                            edges_set.add(e)

                for a, b in edges_set:
                    traces.append(go.Scatter3d(
                        x=[c[a, 0], c[b, 0], None],
                        y=[c[a, 1], c[b, 1], None],
                        z=[c[a, 2], c[b, 2], None],
                        mode="lines",
                        line=dict(color="rgba(40,160,40,0.25)", width=1),
                        showlegend=False, hoverinfo="skip",
                    ))
            except Exception:
                pass

        return traces

    def _octree_boundaries(self, part):
        """Boîtes wireframe."""
        traces = []

        for leaf in part._leaves:
            xmin, xmax, ymin, ymax, zmin, zmax = leaf

            # 12 arêtes d'un cube
            edges = [
                ([xmin, xmax], [ymin, ymin], [zmin, zmin]),
                ([xmin, xmax], [ymax, ymax], [zmin, zmin]),
                ([xmin, xmax], [ymin, ymin], [zmax, zmax]),
                ([xmin, xmax], [ymax, ymax], [zmax, zmax]),
                ([xmin, xmin], [ymin, ymax], [zmin, zmin]),
                ([xmax, xmax], [ymin, ymax], [zmin, zmin]),
                ([xmin, xmin], [ymin, ymax], [zmax, zmax]),
                ([xmax, xmax], [ymin, ymax], [zmax, zmax]),
                ([xmin, xmin], [ymin, ymin], [zmin, zmax]),
                ([xmax, xmax], [ymin, ymin], [zmin, zmax]),
                ([xmin, xmin], [ymax, ymax], [zmin, zmax]),
                ([xmax, xmax], [ymax, ymax], [zmin, zmax]),
            ]

            for ex, ey, ez in edges:
                traces.append(go.Scatter3d(
                    x=ex + [None], y=ey + [None], z=ez + [None],
                    mode="lines",
                    line=dict(color="rgba(100,60,150,0.35)", width=1.5),
                    showlegend=False, hoverinfo="skip",
                ))

        return traces

    # ─────────────────────────────────────────────────────────────
    # SCÈNE COMPLÈTE
    # ─────────────────────────────────────────────────────────────

    def make_scene(self, method, method_kwargs, title=None):
        """
        Crée une figure Plotly complète pour une méthode.

        Returns:
            go.Figure
        """
        # Partitionneur
        part = create_partitioner(method, **method_kwargs)
        part.fit(self.coords)

        states = part.compute_states(
            self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        )
        n_states = part.n_cells
        diag = part.diagnostics(self.coords)

        if title is None:
            title = f"{method} — {part.label}"

        traces = []

        # ── Enveloppe transparente ──
        traces.append(self._make_cylinder_mesh(self.r_max * 1.05))
        traces.append(self._make_disk(self.zmin, self.r_max * 1.05))
        traces.append(self._make_disk(self.zmax, self.r_max * 1.05))

        # ── Frontières ──
        if method == "cartesian":
            traces.extend(self._cartesian_boundaries(**method_kwargs))
        elif method == "cylindrical":
            traces.extend(self._cylindrical_boundaries(**method_kwargs))
        elif method == "voronoi":
            traces.extend(self._voronoi_boundaries(part))
        elif method == "quantile":
            # Mêmes bords que cartésien mais aux quantiles
            traces.extend(self._cartesian_boundaries(
                method_kwargs["nx"], method_kwargs["ny"], method_kwargs["nz"]
            ))
            # Remplacer par les vrais bords quantile (lignes)
            # Les plans sont déjà approximatifs, les lignes seront aux bons endroits
        elif method == "octree":
            traces.extend(self._octree_boundaries(part))

        # ── Particules colorées ──
        traces.append(go.Scatter3d(
            x=self.coords[:, 0],
            y=self.coords[:, 1],
            z=self.coords[:, 2],
            mode="markers",
            marker=dict(
                size=3.5,
                color=states,
                colorscale="Turbo",
                cmin=0, cmax=max(n_states - 1, 1),
                opacity=0.85,
                line=dict(width=0),
                colorbar=dict(
                    title="Cellule",
                    thickness=12,
                    len=0.6,
                ),
            ),
            name="Particules",
            hovertemplate=(
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<br>"
                "Cellule=%{marker.color:.0f}<extra></extra>"
            ),
        ))

        # ── Figure ──
        fig = go.Figure(data=traces)

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{title}</b><br>"
                    f"<span style='font-size:12px'>"
                    f"{n_states} cellules | {diag['n_visited']} visitées | "
                    f"CV={diag['pop_std']/max(diag['pop_mean'],1):.2f}"
                    f"</span>"
                ),
                font_size=16,
            ),
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="data",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                bgcolor="rgb(250,250,252)",
            ),
            height=800,
            width=1000,
            showlegend=False,
            margin=dict(l=0, r=0, t=60, b=0),
        )

        return fig

    def make_comparison(self, figsize=(2000, 1200)):
        """Crée les fichiers HTML pour toutes les méthodes."""
        configs = {
            "cartesian_5x5x5": ("cartesian", {"nx": 5, "ny": 5, "nz": 5}),
            "cylindrical_equal_area": ("cylindrical", {"nr": 5, "ntheta": 8, "nz": 5, "radial_mode": "equal_area"}),
            "cylindrical_equal_dr": ("cylindrical", {"nr": 5, "ntheta": 8, "nz": 5, "radial_mode": "equal_dr"}),
            "voronoi_64": ("voronoi", {"n_cells": 64}),
            "voronoi_125": ("voronoi", {"n_cells": 125}),
            "quantile_5x5x5": ("quantile", {"nx": 5, "ny": 5, "nz": 5}),
            "octree_mp50_d4": ("octree", {"max_particles": 50, "max_depth": 4}),
        }

        print("\n🔧 Génération des visualisations CAD...")
        generated = []

        for name, (method, kwargs) in configs.items():
            print(f"   • {name}...")
            try:
                fig = self.make_scene(method, kwargs, title=name)
                filename = f"cad_{name}.html"
                fig.write_html(filename)
                print(f"     ✅ {filename}")
                generated.append(filename)
            except Exception as e:
                print(f"     ❌ {e}")

        # Index HTML
        self._make_index(generated)
        print(f"\n✨ {len(generated)} fichiers générés!")

    def _make_index(self, files):
        """Crée la page d'index."""
        html = """<!DOCTYPE html>
<html><head>
<title>Mélangeur DEM — Visualisation CAD</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px;
         margin: auto; padding: 20px; background: #f8f9fa; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
          gap: 15px; margin-top: 20px; }
  .card { background: white; border-radius: 12px; padding: 20px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.07); transition: transform 0.2s; }
  .card:hover { transform: translateY(-3px); box-shadow: 0 6px 12px rgba(0,0,0,0.12); }
  .card a { text-decoration: none; color: #2980b9; font-size: 18px; font-weight: 600; }
  .card a:hover { color: #e67e22; }
  .card p { color: #7f8c8d; margin: 8px 0 0; font-size: 13px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 11px; margin-right: 4px; }
  .tag-cart { background: #dbeafe; color: #1e40af; }
  .tag-cyl { background: #fef3c7; color: #92400e; }
  .tag-vor { background: #d1fae5; color: #065f46; }
  .tag-quan { background: #fce7f3; color: #9d174d; }
  .tag-oct { background: #ede9fe; color: #5b21b6; }
</style>
</head><body>
<h1>🔬 Mélangeur DEM — Visualisation CAD 3D</h1>
<p>Cliquez sur une méthode pour l'explorer en 3D interactif</p>
<div class="grid">
"""
        tags = {
            "cartesian": ("Cartésien", "tag-cart"),
            "cylindrical": ("Cylindrique", "tag-cyl"),
            "voronoi": ("Voronoï", "tag-vor"),
            "quantile": ("Quantile", "tag-quan"),
            "octree": ("Octree", "tag-oct"),
        }

        for f in files:
            name = f.replace("cad_", "").replace(".html", "")
            method_key = name.split("_")[0]
            tag_text, tag_class = tags.get(method_key, ("", ""))

            html += f"""
  <div class="card">
    <a href="{f}" target="_blank">📐 {name}</a>
    <p><span class="tag {tag_class}">{tag_text}</span> Vue 3D interactive</p>
  </div>
"""

        html += "</div></body></html>"

        with open("cad_index.html", "w") as f:
            f.write(html)
        print("   ✅ cad_index.html")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = MixerCADPlotly()
    app.load_particles(file_index=100)
    app.make_comparison()

    # Servir les fichiers
    import http.server
    import socketserver

    PORT = 8080
    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"\n🌐 http://localhost:{PORT}/cad_index.html")
        httpd.serve_forever()