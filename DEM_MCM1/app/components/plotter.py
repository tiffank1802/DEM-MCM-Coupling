"""
components/plotter.py
=====================
Matplotlib plotting utilities for matrix analysis and Markov statistics.

Provides functions to:
- Plot transition matrices as heatmaps
- Plot eigenvalue spectra
- Plot RSD evolution curves
- Compare multiple models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import logging
from typing import Optional, Dict, List, Tuple

from src.Markov._config import LoadedModel
from src.Markov.markov_math import (
    analyze_transition_matrix,
    compute_rsd,
)

logger = logging.getLogger(__name__)


class MatrixPlotter:
    """
    Plot transition matrices and related statistics.
    
    Usage:
        >>> plotter = MatrixPlotter()
        >>> fig = plotter.plot_heatmap(model, matrix)
        >>> st.pyplot(fig)
    """
    
    def __init__(self, figsize_default: Tuple[int, int] = (10, 8)):
        self.figsize_default = figsize_default
    
    def plot_heatmap(
        self,
        model: LoadedModel,
        matrix: np.ndarray,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> plt.Figure:
        """
        Plot transition matrix as heatmap.
        
        Args:
            model: LoadedModel with metadata
            matrix: Transition matrix (n_states, n_states)
            title: Custom title (if None, auto-generated)
            figsize: Figure size (width, height) in inches
            cmap: Colormap name
            vmin, vmax: Colorbar value limits
            
        Returns:
            matplotlib Figure object
            
        Example:
            >>> fig = plotter.plot_heatmap(model, M)
            >>> plt.savefig("heatmap.png", dpi=150, bbox_inches="tight")
        """
        if figsize is None:
            figsize = self.figsize_default
        
        if title is None:
            title = f"Transition Matrix: {model.method} ({model.n_states} states)"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Labels and title
        ax.set_xlabel("To State", fontsize=11, fontweight='bold')
        ax.set_ylabel("From State", fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label="Transition Probability")
        
        # Grid
        ax.grid(False)
        
        # Ticks
        n_states = matrix.shape[0]
        if n_states <= 20:
            # Show all ticks for small matrices
            ax.set_xticks(np.arange(0, n_states, 1))
            ax.set_yticks(np.arange(0, n_states, 1))
        else:
            # Sample ticks for large matrices
            tick_step = max(1, n_states // 10)
            ax.set_xticks(np.arange(0, n_states, tick_step))
            ax.set_yticks(np.arange(0, n_states, tick_step))
        
        plt.tight_layout()
        
        logger.info(f"✅ Heatmap created: {matrix.shape}")
        return fig
    
    def plot_heatmap_comparison(
        self,
        models: List[LoadedModel],
        matrices: Dict[str, np.ndarray],
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Plot multiple transition matrices side-by-side.
        
        Args:
            models: List of LoadedModel instances
            matrices: Dict mapping model.folder_name → matrix array
            figsize: Total figure size
            
        Returns:
            matplotlib Figure object
            
        Example:
            >>> models = [model1, model2, model3]
            >>> mats = {m.folder_name: M for m, M in zip(models, matrices_list)}
            >>> fig = plotter.plot_heatmap_comparison(models, mats)
        """
        n_models = len(models)
        
        if figsize is None:
            figsize = (5 * n_models, 4)
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for ax, model in zip(axes, models):
            matrix = matrices.get(model.folder_name)
            if matrix is None:
                ax.text(0.5, 0.5, "Not loaded", ha='center', va='center')
                continue
            
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_title(
                f"{model.method}\n({model.n_states} states)",
                fontsize=10, fontweight='bold'
            )
            ax.set_xlabel("To", fontsize=9)
            ax.set_ylabel("From", fontsize=9)
            plt.colorbar(im, ax=ax, label="P")
        
        plt.tight_layout()
        logger.info(f"✅ Comparison heatmaps created: {n_models} models")
        return fig
    
    def plot_eigenvalues(
        self,
        model: LoadedModel,
        matrix: np.ndarray,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Plot eigenvalue spectrum of transition matrix.
        
        Args:
            model: LoadedModel with metadata
            matrix: Transition matrix (n_states, n_states)
            title: Custom title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
            
        Notes:
            - Shows magnitude and phase of eigenvalues
            - Eigenvalues on complex plane
            - Spectral gap highlighted
        """
        if figsize is None:
            figsize = (12, 5)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)
        
        # Sort by magnitude (descending)
        eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ---- Left: Eigenvalues on complex plane ----
        ax = axes[0]
        
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        magnitudes = np.abs(eigenvalues)
        
        scatter = ax.scatter(
            real_parts, imag_parts,
            c=magnitudes, s=100, alpha=0.6,
            cmap='plasma', edgecolors='black', linewidth=0.5
        )
        
        # Unit circle for reference
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel("Real Part", fontsize=11, fontweight='bold')
        ax.set_ylabel("Imaginary Part", fontsize=11, fontweight='bold')
        ax.set_title("Eigenvalue Spectrum (Complex Plane)", fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax, label="|λ|")
        
        # ---- Right: Magnitude decay ----
        ax = axes[1]
        
        magnitudes_sorted = np.abs(eigenvalues)
        x_range = np.arange(len(magnitudes_sorted))
        
        ax.semilogy(x_range, magnitudes_sorted, 'o-', linewidth=2, markersize=6)
        
        # Highlight spectral gap
        if len(magnitudes_sorted) > 1:
            gap = magnitudes_sorted[0] - magnitudes_sorted[1]
            ax.axhline(
                y=magnitudes_sorted[1],
                color='red', linestyle='--', linewidth=1.5,
                label=f'2nd largest: {magnitudes_sorted[1]:.4f}'
            )
        
        ax.set_xlabel("Eigenvalue Index", fontsize=11, fontweight='bold')
        ax.set_ylabel("|λ| (log scale)", fontsize=11, fontweight='bold')
        ax.set_title("Eigenvalue Magnitude Decay", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        
        # Compute statistics
        stats = analyze_transition_matrix(matrix)
        
        fig.suptitle(
            f"{model.method} - λ₁={magnitudes_sorted[0]:.4f}, "
            f"κ={stats.get('condition_number', np.inf):.2f}",
            fontsize=11, fontweight='bold', y=1.02
        )
        
        plt.tight_layout()
        logger.info(f"✅ Eigenvalue plot created")
        return fig
    
    def plot_eigenvalue_comparison(
        self,
        models: List[LoadedModel],
        matrices: Dict[str, np.ndarray],
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Compare eigenvalue spectra of multiple models.
        
        Args:
            models: List of LoadedModel instances
            matrices: Dict mapping model.folder_name → matrix
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = (12, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for color, model in zip(colors, models):
            matrix = matrices.get(model.folder_name)
            if matrix is None:
                continue
            
            eigenvalues = np.linalg.eigvals(matrix)
            magnitudes = np.abs(eigenvalues)
            magnitudes = magnitudes[np.argsort(-magnitudes)]
            
            x_range = np.arange(len(magnitudes))
            ax.semilogy(
                x_range, magnitudes,
                'o-', linewidth=2, markersize=6,
                label=f"{model.method} ({model.n_states})",
                color=color
            )
        
        ax.set_xlabel("Eigenvalue Index", fontsize=11, fontweight='bold')
        ax.set_ylabel("|λ| (log scale)", fontsize=11, fontweight='bold')
        ax.set_title("Eigenvalue Comparison", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        logger.info(f"✅ Eigenvalue comparison created")
        return fig


class EvolutionPlotter:
    """
    Plot state evolution and mixing kinetics.
    
    Usage:
        >>> plotter = EvolutionPlotter()
        >>> fig = plotter.plot_rsd_evolution(times, rsd_values)
        >>> st.pyplot(fig)
    """
    
    def plot_rsd_evolution(
        self,
        times: np.ndarray,
        rsd_values: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (12, 6),
        title: str = "RSD Evolution",
        xlabel: str = "Time (s)",
        ylabel: str = "RSD (%)",
    ) -> plt.Figure:
        """
        Plot RSD evolution for one or multiple trajectories.
        
        Args:
            times: Time array (seconds), shape (n_timesteps,)
            rsd_values: Dict mapping label → RSD array (%)
            figsize: Figure size
            title: Plot title
            xlabel, ylabel: Axis labels
            
        Returns:
            matplotlib Figure object
            
        Example:
            >>> times = np.array([0, 1, 2, 3, 4, 5])
            >>> rsd_dict = {
            ...     "model1": np.array([80, 60, 40, 25, 15, 10]),
            ...     "model2": np.array([85, 55, 30, 12, 5, 2]),
            ... }
            >>> fig = plotter.plot_rsd_evolution(times, rsd_dict)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(rsd_values)))
        
        for color, (label, rsd) in zip(colors, rsd_values.items()):
            ax.plot(times, rsd, 'o-', linewidth=2.5, markersize=6,
                   label=label, color=color, alpha=0.8)
        
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        logger.info(f"✅ RSD evolution plot created")
        return fig
    
    def plot_concentration_profiles(
        self,
        times: np.ndarray,
        concentrations: np.ndarray,  # shape (n_timesteps, n_states)
        title: str = "Concentration Profiles",
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot concentration in each partition over time.
        
        Args:
            times: Time array
            concentrations: Array (n_timesteps, n_states)
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
            
        Notes:
            - Heatmap showing C(t, i) evolution
            - Useful for detecting segregation patterns
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(
            concentrations.T,  # Transpose for (states, time) layout
            aspect='auto',
            cmap='RdYlGn',
            origin='lower',
            interpolation='nearest'
        )
        
        ax.set_xlabel("Time (s)", fontsize=11, fontweight='bold')
        ax.set_ylabel("State Index", fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        # X-axis: time labels
        n_timesteps = len(times)
        if n_timesteps > 10:
            tick_step = n_timesteps // 10
            ax.set_xticks(np.arange(0, n_timesteps, tick_step))
            ax.set_xticklabels([f"{times[i]:.1f}" for i in range(0, n_timesteps, tick_step)])
        
        cbar = plt.colorbar(im, ax=ax, label="Concentration")
        
        plt.tight_layout()
        logger.info(f"✅ Concentration profile plot created")
        return fig
    
    def plot_segregation_metrics(
        self,
        times: np.ndarray,
        rsd: np.ndarray,
        entropy: np.ndarray,
        intensity_seg: np.ndarray,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Plot multiple segregation metrics over time.
        
        Args:
            times: Time array (seconds)
            rsd: RSD array (0-1 or 0-100%)
            entropy: Entropy array (0-1)
            intensity_seg: Segregation intensity
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Normalize RSD to 0-1 if needed
        if rsd.max() > 10:
            rsd = rsd / 100
        
        # ---- RSD ----
        axes[0].plot(times, rsd, 'o-', linewidth=2, markersize=5, color='#FF6B6B')
        axes[0].fill_between(times, 0, rsd, alpha=0.3, color='#FF6B6B')
        axes[0].set_ylabel("RSD", fontsize=11, fontweight='bold')
        axes[0].set_title("Segregation Metrics Evolution", fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, max(1, rsd.max() * 1.1)])
        
        # ---- Entropy ----
        axes[1].plot(times, entropy, 's-', linewidth=2, markersize=5, color='#4ECDC4')
        axes[1].fill_between(times, 0, entropy, alpha=0.3, color='#4ECDC4')
        axes[1].set_ylabel("Entropy", fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # ---- Intensity of Segregation ----
        axes[2].plot(times, intensity_seg, '^-', linewidth=2, markersize=5, color='#45B7D1')
        axes[2].fill_between(times, 0, intensity_seg, alpha=0.3, color='#45B7D1')
        axes[2].set_xlabel("Time (s)", fontsize=11, fontweight='bold')
        axes[2].set_ylabel("Intensity", fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, max(1, intensity_seg.max() * 1.1)])
        
        plt.tight_layout()
        logger.info(f"✅ Segregation metrics plot created")
        return fig
