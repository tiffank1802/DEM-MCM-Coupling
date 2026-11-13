//! Shared UI state & configuration descriptors for both `egui` (`gui-egui`)
//! and the Live Preview CAD-style workbench (`dem-mcm-gui`).

use std::collections::HashMap;

use crate::gui::camera3d::Camera3D;
use crate::io::json::JsonValue;

/// Exclusive top-level workbench mode preventing simultaneous button clutter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkbenchMode {
    Explore,
    ResearchStudy,
    CompareAB,
}

/// Top-level views separating the compact Summary Dashboard from full-viewport Focus Views.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceTab {
    Summary,
    Mesh3D,
    Teneur,
    Particles,
    Errors,
    MatrixEvolution,
    Matrices,
    Benchmark,
    Study,
    EguiPlan,
    // Compatibility aliases preserved for existing callers
    TeneurParticles,
    ErrorsChains,
    Homogeneous,
    Inhomogeneous,
    DualCompare,
}

/// CAD-inspired collapsible drawer & dropdown visibility flags.
#[derive(Debug, Clone)]
pub struct CollapsiblePanelsState {
    pub ribbon_expanded: bool,
    pub feature_tree_expanded: bool,
    pub timeline_expanded: bool,
    pub node_source_open: bool,
    pub node_mesh_open: bool,
    pub node_markov_open: bool,
    pub node_criterion_open: bool,
    pub node_layers_open: bool,
    pub node_inspector_open: bool,
}

impl Default for CollapsiblePanelsState {
    fn default() -> Self {
        Self {
            ribbon_expanded: true,
            feature_tree_expanded: true,
            timeline_expanded: true,
            node_source_open: true,
            node_mesh_open: true,
            node_markov_open: true,
            node_criterion_open: true,
            node_layers_open: false,
            node_inspector_open: true,
        }
    }
}

/// Interactive filters for the dedicated inhomogeneous matrix evolution view (`p_{i->j}(t)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionViewMode {
    Overlay,
    GridCommonScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionScope {
    OffDiagonal,
    Diagonal,
    All,
    BySourceCell,
}

#[derive(Debug, Clone)]
pub struct MatrixEvolutionFilter {
    pub mode: EvolutionViewMode,
    pub scope: EvolutionScope,
    pub source_cell: usize,
    pub threshold: f64,
    pub show_polynomial_fits: bool,
    pub common_y_scale: bool,
    pub page: usize,
    pub zoomed_pair: Option<(usize, usize)>,
}

impl Default for MatrixEvolutionFilter {
    fn default() -> Self {
        Self {
            mode: EvolutionViewMode::Overlay,
            scope: EvolutionScope::OffDiagonal,
            source_cell: 0,
            threshold: 0.01,
            show_polynomial_fits: true,
            common_y_scale: true,
            page: 0,
            zoomed_pair: None,
        }
    }
}

/// How 3-D cells (or faces) are shaded in the 3-D viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellColorMode {
    CellId,
    Population,
    DemConcentration,
    HomConcentration,
    InhomConcentration,
    HomError,
    InhomError,
    CellRmseDiscrepancy,
}

impl CellColorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CellId => "cell_id",
            Self::Population => "population",
            Self::DemConcentration => "dem_teneur",
            Self::HomConcentration => "hom_teneur",
            Self::InhomConcentration => "inhom_teneur",
            Self::HomError => "hom_err",
            Self::InhomError => "inhom_err",
            Self::CellRmseDiscrepancy => "cell_rmse",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "population" => Self::Population,
            "dem_conc" | "dem_teneur" => Self::DemConcentration,
            "hom_conc" | "hom_teneur" => Self::HomConcentration,
            "inhom_conc" | "inhom_teneur" => Self::InhomConcentration,
            "hom_err" => Self::HomError,
            "inhom_err" => Self::InhomError,
            "cell_rmse" => Self::CellRmseDiscrepancy,
            _ => Self::CellId,
        }
    }
}

/// Which species transition chain / content target to inspect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveSpecies {
    Small,
    Large,
    All,
}

impl ActiveSpecies {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Large => "large",
            Self::All => "all",
        }
    }
}

/// Request payload specifying a mesh + Markov homogeneous/inhomogeneous prediction run.
///
/// Separates **model construction** (`start_index`, `nlt`, `step`, `dt`, `tau`, `permanent_start`)
/// from **full-duration DEM prediction** (`predict_full_duration = true` propagates from
/// `t_min` of the DEM all the way to `t_max`, matching `postprocess.prepare_species`).
#[derive(Debug, Clone)]
pub struct SimulationRequest {
    pub dataset: String, // "synthetic_drum" | "real_chunks"
    pub method: String,
    pub kwargs: HashMap<String, JsonValue>,
    pub nlt: usize,
    pub step: usize,
    pub dt: usize,
    pub tau: usize,
    pub start_index: usize,
    pub permanent_start: usize,
    pub no_species: bool,
    pub target_species: String, // "small" | "large" — criterion: teneur en une espèce
    pub predict_full_duration: bool,
    pub active_block: usize,
}

impl Default for SimulationRequest {
    fn default() -> Self {
        let mut kwargs = HashMap::new();
        kwargs.insert("nx".to_string(), JsonValue::Number(3.0));
        kwargs.insert("ny".to_string(), JsonValue::Number(3.0));
        kwargs.insert("nz".to_string(), JsonValue::Number(2.0));
        Self {
            dataset: "synthetic_drum".to_string(),
            method: "cartesian".to_string(),
            kwargs,
            nlt: 3,
            step: 30,
            dt: 1,
            tau: 12,
            start_index: 40,
            permanent_start: 20,
            no_species: false,
            target_species: "small".to_string(),
            predict_full_duration: true,
            active_block: 0,
        }
    }
}

impl SimulationRequest {
    pub fn default_kwargs_for(method: &str) -> HashMap<String, JsonValue> {
        let mut kw = HashMap::new();
        match method {
            "cartesian" | "quantile" => {
                kw.insert("nx".to_string(), JsonValue::Number(3.0));
                kw.insert("ny".to_string(), JsonValue::Number(3.0));
                kw.insert("nz".to_string(), JsonValue::Number(2.0));
            }
            "cylindrical" => {
                kw.insert("nr".to_string(), JsonValue::Number(3.0));
                kw.insert("ntheta".to_string(), JsonValue::Number(6.0));
                kw.insert("nz".to_string(), JsonValue::Number(1.0));
                kw.insert(
                    "radial_mode".to_string(),
                    JsonValue::String("equal_area".to_string()),
                );
            }
            "octree" => {
                kw.insert("max_particles".to_string(), JsonValue::Number(120.0));
                kw.insert("max_depth".to_string(), JsonValue::Number(3.0));
                kw.insert(
                    "oblique_method".to_string(),
                    JsonValue::String("pca".to_string()),
                );
            }
            "voronoi" => {
                kw.insert("n_cells".to_string(), JsonValue::Number(16.0));
            }
            "physics" | "physics_full_vel" | "spectral" | "gmm" => {
                kw.insert("n_cells".to_string(), JsonValue::Number(16.0));
                kw.insert("velocity_weight".to_string(), JsonValue::Number(0.5));
            }
            "dbscan" => {
                kw.insert("eps".to_string(), JsonValue::Number(0.45));
                kw.insert("min_samples".to_string(), JsonValue::Number(4.0));
                kw.insert("velocity_weight".to_string(), JsonValue::Number(0.6));
            }
            "spectral_biclustering" => {
                kw.insert("n_cells".to_string(), JsonValue::Number(12.0));
                kw.insert("n_col_clusters".to_string(), JsonValue::Number(2.0));
                kw.insert("velocity_weight".to_string(), JsonValue::Number(0.5));
            }
            "adaptive" => {
                kw.insert("y_split".to_string(), JsonValue::Number(0.55));
                kw.insert(
                    "y_split_mode".to_string(),
                    JsonValue::String("quantile".to_string()),
                );
                kw.insert(
                    "top_method".to_string(),
                    JsonValue::String("single".to_string()),
                );
                kw.insert(
                    "bottom_method".to_string(),
                    JsonValue::String("cylindrical".to_string()),
                );
                let mut bkw = HashMap::new();
                bkw.insert("nr".to_string(), JsonValue::Number(2.0));
                bkw.insert("ntheta".to_string(), JsonValue::Number(6.0));
                bkw.insert("nz".to_string(), JsonValue::Number(1.0));
                kw.insert(
                    "bottom_kwargs".to_string(),
                    JsonValue::Object(bkw.into_iter().collect()),
                );
            }
            "multizone" => {
                let mut z1_kw = JsonValue::object();
                z1_kw.insert("nx".to_string(), JsonValue::Number(2.0));
                z1_kw.insert("ny".to_string(), JsonValue::Number(1.0));
                z1_kw.insert("nz".to_string(), JsonValue::Number(2.0));
                let mut z1 = JsonValue::object();
                z1.insert(
                    "method".to_string(),
                    JsonValue::String("cartesian".to_string()),
                );
                z1.insert("y_min".to_string(), JsonValue::Number(0.0));
                z1.insert("y_max".to_string(), JsonValue::Number(0.5));
                z1.insert("kwargs".to_string(), z1_kw);

                let mut z2_kw = JsonValue::object();
                z2_kw.insert("n_cells".to_string(), JsonValue::Number(6.0));
                let mut z2 = JsonValue::object();
                z2.insert(
                    "method".to_string(),
                    JsonValue::String("voronoi".to_string()),
                );
                z2.insert("y_min".to_string(), JsonValue::Number(0.5));
                z2.insert("y_max".to_string(), JsonValue::Number(1.0));
                z2.insert("kwargs".to_string(), z2_kw);

                kw.insert(
                    "y_mode".to_string(),
                    JsonValue::String("quantile".to_string()),
                );
                kw.insert("zones".to_string(), JsonValue::Array(vec![z1, z2]));
            }
            _ => {}
        }
        kw
    }

    pub fn from_json(j: &JsonValue) -> Self {
        let mut def = Self::default();
        if let Some(d) = j.get("dataset").and_then(|v| v.as_str()) {
            def.dataset = d.to_string();
        }
        if let Some(m) = j.get("method").and_then(|v| v.as_str()) {
            def.method = m.to_string();
        }
        if let Some(obj) = j.get("kwargs").and_then(|v| v.as_object()) {
            def.kwargs = obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        } else {
            def.kwargs = Self::default_kwargs_for(&def.method);
        }
        if let Some(v) = j.get("nlt").and_then(|v| v.as_usize()) {
            def.nlt = v.max(1);
        }
        if let Some(v) = j.get("step").and_then(|v| v.as_usize()) {
            def.step = v.max(1);
        }
        if let Some(v) = j.get("dt").and_then(|v| v.as_usize()) {
            def.dt = v.max(1);
        }
        if let Some(v) = j.get("tau").and_then(|v| v.as_usize()) {
            def.tau = v.max(1);
        }
        if let Some(v) = j.get("start_index").and_then(|v| v.as_usize()) {
            def.start_index = v;
        }
        if let Some(v) = j.get("permanent_start").and_then(|v| v.as_usize()) {
            def.permanent_start = v;
        }
        if let Some(v) = j.get("no_species").and_then(|v| v.as_bool()) {
            def.no_species = v;
        }
        if let Some(sp) = j.get("target_species").and_then(|v| v.as_str()) {
            def.target_species = sp.to_string();
        }
        if let Some(v) = j.get("predict_full_duration").and_then(|v| v.as_bool()) {
            def.predict_full_duration = v;
        }
        if let Some(v) = j.get("active_block").and_then(|v| v.as_usize()) {
            def.active_block = v;
        }
        def
    }
}

/// Complete client/desktop interactive viewer state (serializable & egui-ready).
#[derive(Debug, Clone)]
pub struct ViewerAppState {
    pub workbench_mode: WorkbenchMode,
    pub active_tab: WorkspaceTab,
    pub panels: CollapsiblePanelsState,
    pub evol_filter: MatrixEvolutionFilter,
    pub request: SimulationRequest,
    pub secondary_method: String,
    pub color_mode: CellColorMode,
    pub species: ActiveSpecies,
    pub frame_cursor: usize,
    pub playing: bool,
    pub show_wireframe: bool,
    pub show_faces: bool,
    pub show_particles: bool,
    pub show_small_particles: bool,
    pub show_large_particles: bool,
    pub show_velocities: bool,
    pub show_markov_edges: bool,
    pub show_drum: bool,
    pub selected_cell: Option<usize>,
    pub camera: Camera3D,
}

impl Default for ViewerAppState {
    fn default() -> Self {
        Self {
            workbench_mode: WorkbenchMode::Explore,
            active_tab: WorkspaceTab::Summary,
            panels: CollapsiblePanelsState::default(),
            evol_filter: MatrixEvolutionFilter::default(),
            request: SimulationRequest::default(),
            secondary_method: "voronoi".to_string(),
            color_mode: CellColorMode::DemConcentration,
            species: ActiveSpecies::Small,
            frame_cursor: 0,
            playing: false,
            show_wireframe: true,
            show_faces: true,
            show_particles: true,
            show_small_particles: true,
            show_large_particles: true,
            show_velocities: false,
            show_markov_edges: true,
            show_drum: true,
            selected_cell: None,
            camera: Camera3D::default(),
        }
    }
}

/// Metadata entry describing one of the 14 supported partitioning methods.
#[derive(Debug, Clone)]
pub struct MethodDescriptor {
    pub id: &'static str,
    pub name: &'static str,
    pub family: &'static str,
    pub uses_velocity: bool,
    pub summary: &'static str,
}

/// Catalogue of all 14 partitioning approaches rendered in UI selectors.
pub fn catalog_methods() -> Vec<MethodDescriptor> {
    vec![
        MethodDescriptor {
            id: "cartesian",
            name: "Cartésien (nx × ny × nz)",
            family: "Géométrique",
            uses_velocity: false,
            summary: "Grille régulière orthogonale en boîtes hexaédriques 3D.",
        },
        MethodDescriptor {
            id: "cylindrical",
            name: "Cylindrique (nr × nθ × nz)",
            family: "Géométrique",
            uses_velocity: false,
            summary: "Secteurs annulaires adaptés au tambour (aire égale ou Δr égal).",
        },
        MethodDescriptor {
            id: "quantile",
            name: "Quantile (éq.-population)",
            family: "Géométrique",
            uses_velocity: false,
            summary: "Grille 3D à bornes par quantiles pour équilibrer l'occupation.",
        },
        MethodDescriptor {
            id: "octree",
            name: "Octree / Arbre Oblique",
            family: "Géométrique",
            uses_velocity: false,
            summary: "Raffinement hiérarchique 3D (axial ou demi-espaces PCA/SVM/KMeans2).",
        },
        MethodDescriptor {
            id: "voronoi",
            name: "Voronoï 3D (K-Means)",
            family: "Géométrique",
            uses_velocity: false,
            summary: "Polyèdres convexes 3D exacts par intersection de demi-espaces.",
        },
        MethodDescriptor {
            id: "physics",
            name: "Physics-Aware (pos + |V|)",
            family: "Physique",
            uses_velocity: true,
            summary: "Partitionnement 4D (x, y, z, w·|V|) adapté aux zones de cisaillement.",
        },
        MethodDescriptor {
            id: "physics_full_vel",
            name: "Physics Full-Vel (pos + Vx,Vy,Vz)",
            family: "Physique",
            uses_velocity: true,
            summary: "Partitionnement 6D (x, y, z, Vx, Vy, Vz) séparant montée et avalanche.",
        },
        MethodDescriptor {
            id: "spectral",
            name: "Spectral Clustering (kNN)",
            family: "Clustering",
            uses_velocity: true,
            summary: "Partitionnement spectral par graphe de voisinage kNN espace-vitesse.",
        },
        MethodDescriptor {
            id: "dbscan",
            name: "DBSCAN (densité 6D)",
            family: "Clustering",
            uses_velocity: true,
            summary: "Clusters de densité arbitraire avec repli KdTree pour affectation.",
        },
        MethodDescriptor {
            id: "gmm",
            name: "GMM (mélange gaussien EM)",
            family: "Clustering",
            uses_velocity: true,
            summary: "Modèle de mélange gaussien EM avec assignation probabiliste MAP.",
        },
        MethodDescriptor {
            id: "spectral_biclustering",
            name: "Spectral Biclustering",
            family: "Clustering",
            uses_velocity: true,
            summary: "Co-clustering lignes/colonnes espace-cinématique (SVD + KMeans).",
        },
        MethodDescriptor {
            id: "adaptive",
            name: "Adaptatif (haut / lit)",
            family: "Hybride",
            uses_velocity: false,
            summary: "Coupe horizontale y_split : maillage fin dans le lit + zone haute.",
        },
        MethodDescriptor {
            id: "multizone",
            name: "MultiZone (strates N-zones)",
            family: "Hybride",
            uses_velocity: false,
            summary: "Empilement multi-strates combinant des partitionneurs hétérogènes.",
        },
        MethodDescriptor {
            id: "single",
            name: "Domaine unique (1 cellule)",
            family: "Référence",
            uses_velocity: false,
            summary: "Cellule englobante unique de référence (témoin conservation de masse).",
        },
    ]
}
