// Centralized UI State Machine ("Zero-Clutter" progressive disclosure).
// Single source of truth determining which panels, buttons, dropdowns, and
// schema-driven study fields are visible, enabled, or hidden.

const METHOD_CAPABILITIES = {
  cartesian: { usesVelocity: false, resolutionLabel: 'nx, ny, nz', resolutionHint: 'Grille 3D régulière' },
  quantile: { usesVelocity: false, resolutionLabel: 'nx, ny, nz', resolutionHint: 'Quantiles équi-peuplés' },
  cylindrical: { usesVelocity: false, resolutionLabel: 'nr, nθ, nz', resolutionHint: 'Anneaux cylindriques' },
  octree: { usesVelocity: false, resolutionLabel: 'max_particles', resolutionHint: 'Raffinement adaptatif' },
  voronoi: { usesVelocity: false, resolutionLabel: 'n_cells', resolutionHint: 'K-Means 3D' },
  physics: { usesVelocity: true, resolutionLabel: 'n_cells', resolutionHint: 'Position + |V|' },
  physics_full_vel: { usesVelocity: true, resolutionLabel: 'n_cells', resolutionHint: 'Position + (Vx,Vy,Vz)' },
  spectral: { usesVelocity: true, resolutionLabel: 'n_cells', resolutionHint: 'Graphe kNN espace-vitesse' },
  gmm: { usesVelocity: true, resolutionLabel: 'n_cells', resolutionHint: 'Mélange gaussien EM' },
  dbscan: { usesVelocity: true, resolutionLabel: 'eps (densité)', resolutionHint: 'Clusters densité 6D' },
  spectral_biclustering: { usesVelocity: true, resolutionLabel: 'n_cells', resolutionHint: 'Co-clustering SVD' },
  adaptive: { usesVelocity: false, resolutionLabel: 'y_split / lit', resolutionHint: 'Stratification haut/lit' },
  multizone: { usesVelocity: false, resolutionLabel: 'strates', resolutionHint: 'Zones hétérogènes' },
  single: { usesVelocity: false, resolutionLabel: '1 cellule', resolutionHint: 'Domaine global témoin' },
};

const EXPLORE_VIEWS = new Set([
  'summary', 'mesh3d', 'teneur', 'particles', 'errors', 'matrix_evol', 'matrices', 'plan',
]);

function getMethodCapabilities(methodId) {
  return METHOD_CAPABILITIES[methodId] || {
    usesVelocity: false,
    resolutionLabel: 'n_cells',
    resolutionHint: '',
  };
}

/**
 * Pure state-derivation function returning exact UI element visibility & button states.
 * Guarantees that unrelated controls/buttons are never displayed simultaneously.
 */
function deriveUiVisibility(state) {
  const mode = state.workbenchMode || 'explore'; // 'explore' | 'study' | 'compare'
  const view = state.activeView || 'summary';
  const exec = state.executionStatus || 'idle'; // 'idle' | 'computing' | 'error'
  const study = state.study || {};
  const axis1 = study.axis1Kind || 'start_index';
  const axis2 = study.axis2Kind || 'none';
  const isComputing = exec === 'computing';

  const isExplore = mode === 'explore';
  const isStudy = mode === 'study';
  const isCompare = mode === 'compare';

  // 3D viewport controls only appear when the active view actually renders 3D
  const viewNeeds3D = (isExplore && (view === 'summary' || view === 'mesh3d')) || isCompare;

  // Primary contextual action button label (1 button instead of N separate Run buttons)
  let primaryActionLabel = 'Exécuter';
  if (isStudy) primaryActionLabel = isComputing ? 'Étude en cours…' : "Lancer l'étude";
  else if (isCompare) primaryActionLabel = isComputing ? 'Comparaison…' : 'Lancer Benchmark';
  else if (isComputing) primaryActionLabel = 'Calcul…';

  // Study form schema visibility (only controls valid for the chosen axis & law mode)
  const axis1IsMethod = axis1 === 'method';
  const axis1IsSpeciesMode = axis1 === 'species_mode';
  const axis1IsNumeric = !axis1IsMethod && !axis1IsSpeciesMode;
  const showTauCoupling = isStudy && axis1 === 'tau';
  const showLawControls = isStudy && Boolean(study.enableLawStudy || study.enableLaws);
  const showAxis2Input = isStudy && axis2 !== 'none';

  return {
    workbenchMode: mode,
    primaryAction: {
      label: primaryActionLabel,
      disabled: isComputing,
    },
    panels: {
      showShellLayoutToggles: !isStudy, // Hide Arbre/Ruban/Frise buttons in focused Study mode
      showFeatureTree: isExplore && Boolean(state.ui?.treeOpen),
      showRibbon: !isStudy && Boolean(state.ui?.ribbonOpen),
      showTimeline: (isExplore || isCompare) && Boolean(state.ui?.timelineOpen),
      showBottomViewTabs: isExplore,
    },
    ribbon: {
      showCameraAndLayers: viewNeeds3D,
      showCriterionDropdown: isExplore,
      showQuickViewJumps: isExplore,
    },
    studyForm: {
      showAxis1NumericInput: isStudy && axis1IsNumeric,
      showAxis1MethodPicker: isStudy && axis1IsMethod,
      showAxis1SpeciesNote: isStudy && axis1IsSpeciesMode,
      showTauCoupling,
      showAxis2Values: showAxis2Input,
      showLawControls,
    },
    allowedViews: isStudy
      ? ['study']
      : (isCompare ? ['benchmark'] : Array.from(EXPLORE_VIEWS)),
    methodCaps: getMethodCapabilities(state.activeMethod || 'cartesian'),
  };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { deriveUiVisibility, getMethodCapabilities, EXPLORE_VIEWS };
}
