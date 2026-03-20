"""
===================================================================================
VISUALISATION INTERACTIVE PLOTLY — Comparaison partitionnement + prédiction
===================================================================================
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from huggingface_hub import HfFileSystem

from partitioners import create_partitioner

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"


# =============================================================================
# DASHBOARD
# =============================================================================

class InteractiveDashboard:
    """Dashboard interactif pour l'analyse du mélange."""

    def __init__(self):
        self.fs = HfFileSystem()
        self.files = sorted(self.fs.glob(f"{HF_FOLDER}/*.csv"))
        print(f"📁 {len(self.files)} fichiers disponibles")

        self.coords_history = []
        self.partitioners = {}
        self.ground_truth = {}
        self.P_matrices = {}

    def load_snapshots(self, indices=None, sample_every=3):
        """Charge plusieurs snapshots temporels."""
        if indices is None:
            indices = list(range(0, min(len(self.files), 500), 30))

        print(f"📂 Chargement de {len(indices)} snapshots...")

        for i, idx in enumerate(indices):
            with self.fs.open(self.files[idx], "rb") as f:
                df = pl.read_csv(f)

            coords = np.column_stack([
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ])[::sample_every]

            self.coords_history.append({"t": idx, "coords": coords})
            print(f"   [{i+1}/{len(indices)}] t={idx}: {len(coords)} particules")

        print(f"✅ {len(self.coords_history)} snapshots chargés")

    def setup_partitioners(self):
        """Crée et fit tous les partitionneurs."""
        print("\n🔧 Setup des partitionneurs...")

        all_coords = np.vstack([s["coords"] for s in self.coords_history])
        print(f"   {len(all_coords)} points pour le fit")

        configs = {
            "Cartésien (5×5×5)": {
                "method": "cartesian",
                "kwargs": {"nx": 5, "ny": 5, "nz": 5},
            },
            "Cylindrique equal_area": {
                "method": "cylindrical",
                "kwargs": {"nr": 5, "ntheta": 8, "nz": 5, "radial_mode": "equal_area"},
            },
            "Voronoï (125)": {
                "method": "voronoi",
                "kwargs": {"n_cells": 125},
            },
            "Quantile (5×5×5)": {
                "method": "quantile",
                "kwargs": {"nx": 5, "ny": 5, "nz": 5},
            },
        }

        for name, config in configs.items():
            print(f"   • {name}...")
            part = create_partitioner(config["method"], **config["kwargs"])
            part.fit(all_coords)
            self.partitioners[name] = part

        print(f"✅ {len(self.partitioners)} partitionneurs prêts")

    def compute_ground_truth(self):
        """Calcule les états réels pour chaque snapshot."""
        print("\n📊 Calcul des états réels...")

        for name, part in self.partitioners.items():
            self.ground_truth[name] = {}
            for snap in self.coords_history:
                states = part.compute_states(
                    snap["coords"][:, 0],
                    snap["coords"][:, 1],
                    snap["coords"][:, 2],
                )
                self.ground_truth[name][snap["t"]] = states
            print(f"   ✅ {name}")

    def compute_transition_matrices(self):
        """Calcule une matrice de transition pour chaque méthode."""
        print("\n🔢 Calcul des matrices de transition...")

        for name, part in self.partitioners.items():
            n_states = part.n_cells
            T = np.zeros((n_states, n_states))

            for i in range(len(self.coords_history) - 1):
                t_prev = self.coords_history[i]["t"]
                t_curr = self.coords_history[i + 1]["t"]

                s_prev = self.ground_truth[name][t_prev]
                s_curr = self.ground_truth[name][t_curr]

                n = min(len(s_prev), len(s_curr))
                for j in range(n):
                    T[s_prev[j], s_curr[j]] += 1

            # Normaliser
            row_sums = T.sum(axis=1, keepdims=True)
            P = np.divide(T, row_sums, where=row_sums > 0, out=np.zeros_like(T))

            self.P_matrices[name] = P
            print(f"   ✅ {name}: diag_mean={np.diag(P).mean():.3f}")

    def predict_evolution(self, method_name, t_start, n_steps):
        """Prédit l'évolution via S(t+1) = S(t) @ P."""
        states_init = self.ground_truth[method_name][t_start]
        n_states = self.partitioners[method_name].n_cells

        S = np.bincount(states_init, minlength=n_states).astype(float)
        S = S / S.sum() if S.sum() > 0 else S

        P = self.P_matrices[method_name]
        predictions = {0: S.copy()}

        for step in range(1, n_steps):
            S = S @ P
            predictions[step] = S.copy()

        return predictions

    # ─────────────────────────────────────────────────────────────────
    # VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────

    def plot_3d_particles_comparison(self, t_index=0):
        """Vue 3D des particules colorées par cellule pour chaque méthode."""
        snap = self.coords_history[t_index]
        t = snap["t"]
        coords = snap["coords"]
        names = list(self.partitioners.keys())
        n = len(names)

        cols = 2
        rows = (n + 1) // 2

        specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=names,
            specs=specs,
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
        )

        for i, name in enumerate(names):
            row = i // cols + 1
            col = i % cols + 1

            states = self.ground_truth[name][t]
            n_states = self.partitioners[name].n_cells

            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2.5,
                        color=states,
                        colorscale="Turbo",
                        cmin=0, cmax=max(n_states - 1, 1),
                        showscale=False,
                    ),
                    name=name,
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<br>"
                        "État=%{marker.color}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

        # Mettre à jour chaque scène
        scene_updates = {}
        for i in range(n):
            scene_key = "scene" if i == 0 else f"scene{i+1}"
            scene_updates[scene_key] = dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            )

        fig.update_layout(
            title=dict(text=f"<b>Comparaison des partitionnements — t={t}</b>",
                       font_size=18),
            height=500 * rows,
            width=1400,
            showlegend=False,
            **scene_updates,
        )

        return fig

    def plot_single_method_3d(self, method_name, t_index=0):
        """Vue 3D détaillée d'une seule méthode."""
        snap = self.coords_history[t_index]
        coords = snap["coords"]
        t = snap["t"]

        states = self.ground_truth[method_name][t]
        n_states = self.partitioners[method_name].n_cells

        # Population par cellule
        counts = np.bincount(states, minlength=n_states)

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=states,
                colorscale="Turbo",
                cmin=0, cmax=n_states - 1,
                colorbar=dict(title="État", thickness=15),
            ),
            hovertemplate=(
                f"<b>{method_name}</b><br>"
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<br>"
                "État=%{marker.color}<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=f"<b>{method_name}</b> — t={t} | {n_states} cellules | "
                  f"{(counts > 0).sum()} visitées",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="cube",
            ),
            height=700, width=900,
        )

        return fig

    def plot_transition_matrices(self):
        """Heatmaps des matrices P."""
        names = list(self.P_matrices.keys())
        n = len(names)
        cols = 2
        rows = (n + 1) // 2

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[
                f"{name} (diag={np.diag(P).mean():.3f})"
                for name, P in self.P_matrices.items()
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        for i, (name, P) in enumerate(self.P_matrices.items()):
            row = i // cols + 1
            col = i % cols + 1

            fig.add_trace(
                go.Heatmap(
                    z=P, colorscale="Viridis",
                    zmin=0, zmax=min(P.max(), 1),
                    showscale=(i == 0),
                    hovertemplate="Source=%{y}<br>Dest=%{x}<br>P=%{z:.4f}<extra></extra>",
                ),
                row=row, col=col,
            )

            fig.update_xaxes(title_text="Destination", row=row, col=col)
            fig.update_yaxes(title_text="Source", autorange="reversed", row=row, col=col)

        fig.update_layout(
            title="<b>Matrices de transition P</b>",
            height=450 * rows,
            width=1200,
        )

        return fig

    def plot_prediction_vs_reality(self, method_name, t_start_idx=0, n_steps=None):
        """
        Compare distribution prédite vs réelle.
        Ligne 1: réel | Ligne 2: prédit | Ligne 3: erreur
        """
        if n_steps is None:
            n_steps = min(5, len(self.coords_history) - t_start_idx)

        t_start = self.coords_history[t_start_idx]["t"]
        n_states = self.partitioners[method_name].n_cells

        predictions = self.predict_evolution(method_name, t_start, n_steps)

        # Snapshots à afficher
        snaps = self.coords_history[t_start_idx:t_start_idx + n_steps]
        n_cols = len(snaps)

        titles = []
        for step, snap in enumerate(snaps):
            titles.extend([
                f"Réel t={snap['t']}", f"Prédit (step {step})", f"Erreur step {step}"
            ])

        fig = make_subplots(
            rows=3, cols=n_cols,
            subplot_titles=[f"t={s['t']}" for s in snaps],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        colors = {"real": "#1f77b4", "pred": "#ff7f0e", "error": "#d62728"}

        for col_idx, snap in enumerate(snaps):
            step = col_idx
            t = snap["t"]

            # Distribution réelle
            states_real = self.ground_truth[method_name][t]
            dist_real = np.bincount(states_real, minlength=n_states).astype(float)
            dist_real = dist_real / dist_real.sum() if dist_real.sum() > 0 else dist_real

            # Distribution prédite
            dist_pred = predictions.get(step, np.zeros(n_states))

            # Erreur
            error = np.abs(dist_real - dist_pred)
            l1 = error.sum()

            x_vals = np.arange(n_states)

            # Réel
            fig.add_trace(
                go.Bar(x=x_vals, y=dist_real, marker_color=colors["real"],
                       name="Réel", showlegend=(col_idx == 0),
                       hovertemplate="État %{x}<br>P=%{y:.4f}<extra></extra>"),
                row=1, col=col_idx + 1,
            )

            # Prédit
            fig.add_trace(
                go.Bar(x=x_vals, y=dist_pred, marker_color=colors["pred"],
                       name="Prédit", showlegend=(col_idx == 0),
                       hovertemplate="État %{x}<br>P=%{y:.4f}<extra></extra>"),
                row=2, col=col_idx + 1,
            )

            # Erreur
            fig.add_trace(
                go.Bar(x=x_vals, y=error, marker_color=colors["error"],
                       name=f"L1={l1:.3f}", showlegend=True,
                       hovertemplate="État %{x}<br>|err|=%{y:.4f}<extra></extra>"),
                row=3, col=col_idx + 1,
            )

        # Labels
        for col in range(1, n_cols + 1):
            fig.update_xaxes(title_text="État", row=3, col=col)
        fig.update_yaxes(title_text="P (réel)", row=1, col=1)
        fig.update_yaxes(title_text="P (prédit)", row=2, col=1)
        fig.update_yaxes(title_text="|erreur|", row=3, col=1)

        fig.update_layout(
            title=f"<b>{method_name}</b> — Prédiction vs Réalité",
            height=800,
            width=300 * n_cols,
            barmode="overlay",
        )

        return fig

    def plot_error_evolution(self):
        """Erreur L1 au fil du temps pour toutes les méthodes."""
        fig = go.Figure()

        method_colors = {
            "Cartésien (5×5×5)": "#1f77b4",
            "Cylindrique equal_area": "#ff7f0e",
            "Voronoï (125)": "#2ca02c",
            "Quantile (5×5×5)": "#d62728",
        }

        for name in self.partitioners.keys():
            t_start = self.coords_history[0]["t"]
            n_steps = len(self.coords_history)
            predictions = self.predict_evolution(name, t_start, n_steps)
            n_states = self.partitioners[name].n_cells

            errors = []
            steps = []

            for step, snap in enumerate(self.coords_history):
                t = snap["t"]

                states_real = self.ground_truth[name][t]
                dist_real = np.bincount(states_real, minlength=n_states).astype(float)
                dist_real = dist_real / dist_real.sum() if dist_real.sum() > 0 else dist_real

                dist_pred = predictions.get(step, np.zeros(n_states))

                l1 = np.abs(dist_real - dist_pred).sum()
                errors.append(l1)
                steps.append(step)

            fig.add_trace(go.Scatter(
                x=steps, y=errors,
                mode="lines+markers",
                name=name,
                line=dict(width=3, color=method_colors.get(name, None)),
                marker=dict(size=8),
                hovertemplate=f"<b>{name}</b><br>Step %{{x}}<br>L1=%{{y:.4f}}<extra></extra>",
            ))

        fig.update_layout(
            title="<b>Erreur de prédiction au fil du temps</b> (L1 distance)",
            xaxis_title="Pas de prédiction",
            yaxis_title="Erreur L1",
            height=500, width=1200,
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        )

        return fig

    def plot_population_distribution(self, t_index=0):
        """Distribution de population par cellule pour chaque méthode."""
        snap = self.coords_history[t_index]
        t = snap["t"]
        names = list(self.partitioners.keys())
        n = len(names)

        fig = make_subplots(
            rows=1, cols=n,
            subplot_titles=names,
            horizontal_spacing=0.06,
        )

        for i, name in enumerate(names):
            states = self.ground_truth[name][t]
            n_states = self.partitioners[name].n_cells

            counts = np.bincount(states, minlength=n_states)
            counts_nz = counts[counts > 0]
            cv = counts_nz.std() / counts_nz.mean() if counts_nz.mean() > 0 else 0

            fig.add_trace(
                go.Histogram(
                    x=counts_nz, nbinsx=30,
                    marker_color="steelblue",
                    name=f"{name} (CV={cv:.2f})",
                    hovertemplate="Pop=%{x}<br>Nb cellules=%{y}<extra></extra>",
                ),
                row=1, col=i + 1,
            )

            fig.update_xaxes(title_text="Particules/cellule", row=1, col=i + 1)

        fig.update_yaxes(title_text="Nb cellules", row=1, col=1)

        fig.update_layout(
            title=f"<b>Distribution de population par cellule</b> — t={t}",
            height=400, width=350 * n,
            showlegend=True,
        )

        return fig

    def plot_diagonal_comparison(self):
        """Compare les diagonales de P (probabilité de rester)."""
        fig = go.Figure()

        for name, P in self.P_matrices.items():
            diag = np.diag(P)
            x_vals = np.arange(len(diag))

            fig.add_trace(go.Bar(
                x=x_vals, y=diag, name=name,
                opacity=0.7,
                hovertemplate=f"<b>{name}</b><br>État %{{x}}<br>P(rester)=%{{y:.4f}}<extra></extra>",
            ))

        fig.update_layout(
            title="<b>Probabilité de rester dans le même état</b> (diagonale de P)",
            xaxis_title="Index d'état",
            yaxis_title="P(rester)",
            height=500, width=1200,
            barmode="group",
            legend=dict(x=0.01, y=0.99),
        )

        return fig

    def plot_mixing_animation(self, method_name):
        """Animation 3D du mélange au fil du temps."""
        n_states = self.partitioners[method_name].n_cells

        frames = []
        slider_steps = []

        for i, snap in enumerate(self.coords_history):
            t = snap["t"]
            coords = snap["coords"]
            states = self.ground_truth[method_name][t]

            frame = go.Frame(
                data=[go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=3, color=states,
                        colorscale="Turbo",
                        cmin=0, cmax=n_states - 1,
                        showscale=True,
                        colorbar=dict(title="État"),
                    ),
                    hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<br>"
                                 "État=%{marker.color}<extra></extra>",
                )],
                name=str(t),
                layout=go.Layout(title_text=f"<b>{method_name}</b> — t={t}"),
            )
            frames.append(frame)

            slider_steps.append(dict(
                args=[[str(t)], dict(frame=dict(duration=300, redraw=True),
                                     mode="immediate")],
                label=str(t),
                method="animate",
            ))

        # Figure initiale
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="cube",
            ),
            title=f"<b>{method_name}</b> — Animation temporelle",
            height=700, width=900,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=500, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=True),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=slider_steps,
                currentvalue=dict(prefix="t=", font_size=14),
                pad=dict(t=60),
            )],
        )

        return fig

    def plot_eigenvalues(self):
        """Spectre des valeurs propres de chaque matrice P."""
        fig = go.Figure()

        for name, P in self.P_matrices.items():
            n_eig = min(20, P.shape[0])
            eigenvalues = np.sort(np.abs(np.linalg.eigvals(P)))[::-1][:n_eig]

            fig.add_trace(go.Scatter(
                x=np.arange(n_eig), y=eigenvalues,
                mode="lines+markers",
                name=name,
                hovertemplate=f"<b>{name}</b><br>λ_%{{x}}=%{{y:.4f}}<extra></extra>",
            ))

        fig.update_layout(
            title="<b>Spectre des valeurs propres</b> (|λ₂| contrôle la vitesse de mélange)",
            xaxis_title="Index",
            yaxis_title="|λ|",
            height=500, width=1000,
            hovermode="x unified",
        )

        return fig

    def plot_all_methods_prediction_overlay(self, step=2):
        """
        Superpose la distribution réelle et prédite pour chaque méthode
        à un step donné.
        """
        names = list(self.partitioners.keys())
        n = len(names)

        fig = make_subplots(
            rows=1, cols=n,
            subplot_titles=names,
            horizontal_spacing=0.06,
        )

        for i, name in enumerate(names):
            t_start = self.coords_history[0]["t"]
            predictions = self.predict_evolution(name, t_start, step + 1)
            n_states = self.partitioners[name].n_cells

            # Réel
            snap = self.coords_history[min(step, len(self.coords_history) - 1)]
            states_real = self.ground_truth[name][snap["t"]]
            dist_real = np.bincount(states_real, minlength=n_states).astype(float)
            dist_real = dist_real / dist_real.sum() if dist_real.sum() > 0 else dist_real

            # Prédit
            dist_pred = predictions.get(step, np.zeros(n_states))

            x = np.arange(n_states)

            fig.add_trace(
                go.Bar(x=x, y=dist_real, name="Réel", marker_color="steelblue",
                       opacity=0.6, showlegend=(i == 0)),
                row=1, col=i + 1,
            )
            fig.add_trace(
                go.Bar(x=x, y=dist_pred, name="Prédit", marker_color="orange",
                       opacity=0.6, showlegend=(i == 0)),
                row=1, col=i + 1,
            )

            # L1
            l1 = np.abs(dist_real - dist_pred).sum()
            fig.add_annotation(
                text=f"L1={l1:.3f}",
                xref=f"x{i+1}" if i > 0 else "x",
                yref=f"y{i+1}" if i > 0 else "y",
                x=n_states // 2, y=max(dist_real.max(), dist_pred.max()) * 0.9,
                showarrow=False, font=dict(size=14, color="red"),
            )

            fig.update_xaxes(title_text="État", row=1, col=i + 1)

        fig.update_yaxes(title_text="Probabilité", row=1, col=1)

        fig.update_layout(
            title=f"<b>Réel vs Prédit</b> — step={step}",
            height=450, width=350 * n,
            barmode="overlay",
        )

        return fig

    # ─────────────────────────────────────────────────────────────────
    # DASHBOARD COMPLET
    # ─────────────────────────────────────────────────────────────────

    def create_full_dashboard(self, output_dir="."):
        """Génère tous les fichiers HTML interactifs."""
        print("\n📊 Génération du dashboard...")

        plots = [
            ("1_comparison_3d", self.plot_3d_particles_comparison, {}),
            ("2_matrices_P", self.plot_transition_matrices, {}),
            ("3_error_evolution", self.plot_error_evolution, {}),
            ("4_population", self.plot_population_distribution, {}),
            ("5_diagonal", self.plot_diagonal_comparison, {}),
            ("6_eigenvalues", self.plot_eigenvalues, {}),
            ("7_prediction_overlay_step1", self.plot_all_methods_prediction_overlay, {"step": 1}),
            ("8_prediction_overlay_step3", self.plot_all_methods_prediction_overlay, {"step": 3}),
        ]

        # Prédiction par méthode
        for i, name in enumerate(self.partitioners.keys(), start=9):
            plots.append((
                f"{i}_prediction_{name.replace(' ', '_').replace('(', '').replace(')', '')}",
                self.plot_prediction_vs_reality,
                {"method_name": name},
            ))

        # Animation par méthode
        for i, name in enumerate(self.partitioners.keys(), start=20):
            plots.append((
                f"{i}_animation_{name.replace(' ', '_').replace('(', '').replace(')', '')}",
                self.plot_mixing_animation,
                {"method_name": name},
            ))

        # Générer
        generated = []
        for filename, func, kwargs in plots:
            try:
                fig = func(**kwargs)
                filepath = f"{output_dir}/dashboard_{filename}.html"
                fig.write_html(filepath)
                print(f"   ✅ {filepath}")
                generated.append(filepath)
            except Exception as e:
                print(f"   ❌ {filename}: {e}")

        # Index HTML
        self._generate_index_html(generated, output_dir)

        print(f"\n✨ {len(generated)} fichiers générés!")
        print(f"   Ouvrez {output_dir}/index.html dans votre navigateur")

    def _generate_index_html(self, files, output_dir):
        """Crée une page d'index avec des liens vers tous les plots."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Dashboard Markov — Mélangeur DEM</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #2ca02c; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .card { background: white; border-radius: 8px; padding: 15px 20px; margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card a { text-decoration: none; color: #1f77b4; font-size: 16px; font-weight: bold; }
        .card a:hover { color: #ff7f0e; }
        .card p { color: #777; margin: 5px 0 0; font-size: 13px; }
        .section { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>🔬 Dashboard Markov — Mélangeur DEM</h1>

    <div class="section">
        <h2>📊 Comparaisons</h2>
"""
        descriptions = {
            "comparison_3d": "Vue 3D interactive des particules colorées par cellule",
            "matrices_P": "Heatmaps des matrices de transition P",
            "error_evolution": "Erreur L1 de prédiction au fil du temps",
            "population": "Distribution du nombre de particules par cellule",
            "diagonal": "Probabilité de rester dans le même état (diagonale de P)",
            "eigenvalues": "Spectre des valeurs propres (vitesse de mélange)",
            "prediction_overlay": "Distribution réelle vs prédite superposées",
            "prediction_": "Prédiction vs réalité détaillée",
            "animation_": "Animation 3D du mélange au cours du temps",
        }

        for filepath in files:
            filename = filepath.split("/")[-1].replace("dashboard_", "").replace(".html", "")

            desc = "Visualisation interactive"
            for key, d in descriptions.items():
                if key in filename:
                    desc = d
                    break

            html += f"""
        <div class="card">
            <a href="{filepath.split('/')[-1]}" target="_blank">📈 {filename}</a>
            <p>{desc}</p>
        </div>
"""

        html += """
    </div>
</body>
</html>"""

        index_path = f"{output_dir}/index.html"
        with open(index_path, "w") as f:
            f.write(html)
        print(f"   ✅ {index_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    dashboard = InteractiveDashboard()

    # Charger les données (10 snapshots, 1 tous les 30 fichiers)
    dashboard.load_snapshots(
        indices=list(range(0, 300, 30)),
        sample_every=1,  # toutes les particules
    )

    # Setup + calculs
    dashboard.setup_partitioners()
    dashboard.compute_ground_truth()
    dashboard.compute_transition_matrices()

    # Générer le dashboard
    dashboard.create_full_dashboard(output_dir="dashboard")