// `APP` (reactive store proxy) is defined in state.js and loaded before
// this file. Every APP.x read/write here transparently maps to the central
// store, so a change made anywhere propagates to all subscribers.

const PALETTE = [
  '#8faec9', '#7dd3bd', '#eec078', '#e5989b', '#b8b8e6', '#d8a48f',
  '#88c9c2', '#a5bcd6', '#cfa8c4', '#b4cc8e', '#c9a6e0', '#e8cf7a',
];

function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function saveUiLayout() {
  try {
    const nodes = {};
    document.querySelectorAll('details.tree-node').forEach(el => {
      if (el.id) nodes[el.id] = el.open;
    });
    localStorage.setItem('dem_mcm_cad_layout', JSON.stringify({
      workbenchMode: APP.workbenchMode,
      activeView: APP.activeView,
      treeOpen: APP.ui.treeOpen,
      ribbonOpen: APP.ui.ribbonOpen,
      timelineOpen: APP.ui.timelineOpen,
      nodes,
    }));
  } catch (_) {}
}

function restoreUiLayout() {
  try {
    const raw = localStorage.getItem('dem_mcm_cad_layout');
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved.workbenchMode === 'explore' || saved.workbenchMode === 'study' || saved.workbenchMode === 'compare') {
      APP.workbenchMode = saved.workbenchMode;
    }
    if (typeof saved.treeOpen === 'boolean') APP.ui.treeOpen = saved.treeOpen;
    if (typeof saved.ribbonOpen === 'boolean') APP.ui.ribbonOpen = saved.ribbonOpen;
    if (typeof saved.timelineOpen === 'boolean') APP.ui.timelineOpen = saved.timelineOpen;
    if (saved.nodes && typeof saved.nodes === 'object') {
      Object.entries(saved.nodes).forEach(([id, isOpen]) => {
        const el = document.getElementById(id);
        if (el) el.open = Boolean(isOpen);
      });
    }
    if (saved.activeView && document.getElementById('pane-' + saved.activeView)) {
      APP.activeView = saved.activeView;
    }
  } catch (_) {}
}

function applyPanelVisibility() {
  const studySt = (typeof getStudyBuilderState === 'function') ? getStudyBuilderState() : {};
  const vis = deriveUiVisibility({
    workbenchMode: APP.workbenchMode,
    activeView: APP.activeView,
    executionStatus: APP.executionStatus,
    activeMethod: APP.method,
    ui: APP.ui,
    study: studySt,
  });

  // Top mode switcher active state
  document.querySelectorAll('#workbench-mode-switch .mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === vis.workbenchMode);
  });

  // Single primary contextual run button
  const runBtn = document.getElementById('run-btn');
  if (runBtn) {
    runBtn.textContent = vis.primaryAction.label;
    runBtn.disabled = vis.primaryAction.disabled;
  }

  // Shell layout toggles (hidden in focused Study mode so zero 3D toggles clutter the top bar)
  const tree = document.getElementById('feature-tree');
  const ribbon = document.getElementById('ribbon-bar');
  const timeline = document.getElementById('timeline-bar');
  const viewTabs = document.getElementById('view-tabs');
  const btnTree = document.getElementById('toggle-tree-btn');
  const btnRibbon = document.getElementById('toggle-ribbon-btn');
  const btnTimeline = document.getElementById('toggle-timeline-btn');

  [btnTree, btnRibbon, btnTimeline].forEach(b => {
    if (b) b.style.display = vis.panels.showShellLayoutToggles ? 'inline-flex' : 'none';
  });

  if (tree) tree.classList.toggle('collapsed', !vis.panels.showFeatureTree);
  if (ribbon) ribbon.classList.toggle('collapsed', !vis.panels.showRibbon);
  if (timeline) timeline.classList.toggle('collapsed', !vis.panels.showTimeline);
  if (viewTabs) viewTabs.style.display = vis.panels.showBottomViewTabs ? 'flex' : 'none';

  if (btnTree) {
    btnTree.textContent = APP.ui.treeOpen ? '◂ Arbre' : '▸ Arbre';
    btnTree.classList.toggle('active-toggle', APP.ui.treeOpen);
  }
  if (btnRibbon) {
    btnRibbon.textContent = APP.ui.ribbonOpen ? '▴ Ruban' : '▾ Ruban';
    btnRibbon.classList.toggle('active-toggle', APP.ui.ribbonOpen);
  }
  if (btnTimeline) {
    btnTimeline.textContent = APP.ui.timelineOpen ? '▾ Frise' : '▴ Frise';
    btnTimeline.classList.toggle('active-toggle', APP.ui.timelineOpen);
  }

  // Contextual Ribbon groups: hide 3D camera & layers when not viewing a 3D tab
  const r3d = document.getElementById('ribbon-group-3d');
  const rCrit = document.getElementById('ribbon-group-criterion');
  const rViews = document.getElementById('ribbon-group-views');
  if (r3d) r3d.style.display = vis.ribbon.showCameraAndLayers ? 'flex' : 'none';
  if (rCrit) rCrit.style.display = vis.ribbon.showCriterionDropdown ? 'flex' : 'none';
  if (rViews) rViews.style.display = vis.ribbon.showQuickViewJumps ? 'flex' : 'none';

  // Schema-driven Study Builder field visibility
  const setDisp = (id, show, flex = false) => {
    const el = document.getElementById(id);
    if (el) el.style.display = show ? (flex ? 'flex' : 'block') : 'none';
  };
  setDisp('study-axis1-num-box', vis.studyForm.showAxis1NumericInput, true);
  setDisp('study-axis1-method-box', vis.studyForm.showAxis1MethodPicker);
  setDisp('study-axis1-species-note', vis.studyForm.showAxis1SpeciesNote);
  setDisp('study-tau-coupling-row', vis.studyForm.showTauCoupling, true);
  setDisp('study-axis2-val-box', vis.studyForm.showAxis2Values, true);
  setDisp('study-law-controls-box', vis.studyForm.showLawControls, true);

  const studySide = document.getElementById('study-sidebar');
  if (studySide) {
    studySide.classList.toggle('collapsed', !APP.ui.studyBuilderOpen);
  }
}

function defaultKwargs(method) {
  return {
    cartesian: { nx: 3, ny: 3, nz: 2 },
    quantile: { nx: 3, ny: 3, nz: 2 },
    cylindrical: { nr: 3, ntheta: 6, nz: 1, radial_mode: 'equal_area' },
    octree: { max_particles: 120, max_depth: 3, oblique_method: 'pca' },
    voronoi: { n_cells: 16 },
    physics: { n_cells: 16, velocity_weight: 0.5 },
    physics_full_vel: { n_cells: 16, velocity_weight: 0.5 },
    spectral: { n_cells: 16, velocity_weight: 0.5 },
    gmm: { n_cells: 14, velocity_weight: 0.5 },
    dbscan: { eps: 0.45, min_samples: 4, velocity_weight: 0.6 },
    spectral_biclustering: { n_cells: 12, n_col_clusters: 2, velocity_weight: 0.5 },
    adaptive: {
      y_split: 0.55,
      y_split_mode: 'quantile',
      top_method: 'single',
      bottom_method: 'cylindrical',
      bottom_kwargs: { nr: 2, ntheta: 6, nz: 1 },
    },
    multizone: {
      y_mode: 'quantile',
      zones: [
        { method: 'cartesian', y_min: 0, y_max: 0.5, kwargs: { nx: 2, ny: 1, nz: 2 } },
        { method: 'voronoi', y_min: 0.5, y_max: 1, kwargs: { n_cells: 6 } },
      ],
    },
    single: {},
  }[method] || {};
}

function renderMethodParams(method) {
  const box = document.getElementById('method-params');
  const m = APP.methods.find(x => x.id === method);
  const famLbl = document.getElementById('method-family-lbl');
  if (m && famLbl) {
    famLbl.textContent = m.family + (m.uses_velocity ? ' · cinématique' : '') + ' — ' + m.summary;
  }
  const fields = {
    cartesian: [['nx', 'nx', 3, 1, 10], ['ny', 'ny', 3, 1, 10], ['nz', 'nz', 2, 1, 8]],
    quantile: [['nx', 'nx', 3, 1, 10], ['ny', 'ny', 3, 1, 10], ['nz', 'nz', 2, 1, 8]],
    cylindrical: [['nr', 'nr', 3, 1, 8], ['ntheta', 'nθ', 6, 2, 24], ['nz', 'nz', 1, 1, 6]],
    octree: [['max_particles', 'max_pts', 120, 15, 2000], ['max_depth', 'prof.', 3, 1, 5]],
    voronoi: [['n_cells', 'n_cells', 16, 2, 64]],
    physics: [['n_cells', 'n_cells', 16, 2, 64], ['velocity_weight', 'w_V', 0.5, 0, 5, 0.1]],
    physics_full_vel: [['n_cells', 'n_cells', 16, 2, 64], ['velocity_weight', 'w_V', 0.5, 0, 5, 0.1]],
    spectral: [['n_cells', 'n_cells', 16, 2, 40], ['velocity_weight', 'w_V', 0.5, 0, 5, 0.1]],
    gmm: [['n_cells', 'n_cells', 14, 2, 40], ['velocity_weight', 'w_V', 0.5, 0, 5, 0.1]],
    dbscan: [['eps', 'eps(z)', 0.45, 0.05, 1.5, 0.02], ['min_samples', 'min_pts', 4, 2, 50]],
    spectral_biclustering: [['n_cells', 'n_row', 12, 2, 32], ['n_col_clusters', 'n_col', 2, 2, 6]],
    adaptive: [['y_split', 'y_split', 0.55, 0.2, 0.85, 0.05]],
  }[method] || [];

  let html = fields.map(([k, lbl, val, mn, mx, st]) => `
    <div class="mini-field">
      <label>${lbl}</label>
      <input type="number" data-kw="${k}" value="${val}" min="${mn}" max="${mx}" step="${st || 1}">
    </div>
  `).join('');

  if (method === 'cylindrical') {
    html += `<div class="mini-field"><label>Mode</label>
      <select data-kw="radial_mode"><option value="equal_area">equal_area</option><option value="equal_dr">equal_dr</option></select></div>`;
  }
  if (method === 'octree') {
    html += `<div class="mini-field"><label>Coupes</label>
      <select data-kw="oblique_method">
        <option value="pca">pca</option>
        <option value="axis">axis</option>
        <option value="kmeans2">kmeans2</option>
        <option value="2medians">2medians</option>
        <option value="svm">svm</option>
        <option value="random">random</option>
      </select></div>`;
  }
  box.innerHTML = html;
}

function gatherKwargs(method) {
  const kw = {};
  document.querySelectorAll('#method-params [data-kw]').forEach(el => {
    const k = el.dataset.kw;
    kw[k] = el.tagName === 'SELECT' ? el.value : parseFloat(el.value);
  });
  if (method === 'adaptive') {
    kw.y_split_mode = 'quantile';
    kw.top_method = 'single';
    kw.bottom_method = 'cylindrical';
    kw.bottom_kwargs = { nr: 2, ntheta: 6, nz: 1 };
  } else if (method === 'multizone') {
    kw.y_mode = 'quantile';
    kw.zones = [
      { method: 'cartesian', y_min: 0.0, y_max: 0.5, kwargs: { nx: 2, ny: 1, nz: 2 } },
      { method: 'voronoi', y_min: 0.5, y_max: 1.0, kwargs: { n_cells: 6 } },
    ];
  }
  return kw;
}

function currentRequest(method, kwargs) {
  // Read shared variables from the central reactive store (APP.*) so that a
  // value updated in ANY part of the software is what gets sent to the API —
  // the DOM controls are kept in sync via bindControl.
  return {
    dataset: APP.dataset,
    method,
    kwargs,
    nlt: APP.nlt || 3,
    step: APP.step || 30,
    dt: APP.dt || 1,
    tau: APP.tau || 12,
    start_index: APP.startIndex || 0,
    no_species: !!APP.noSpecies,
    target_species: APP.targetSpecies || 'small',
    predict_full_duration: !!APP.predictFullDuration,
  };
}

async function fetchSimulate(method, kwargs) {
  const payload = currentRequest(method, kwargs);
  if (!APP.staticMode) {
    try {
      const resp = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        return data;
      }
      if (resp.status !== 404 && resp.status !== 405) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || 'Erreur serveur Rust');
      }
    } catch (e) {
      if (!String(e).includes('Failed to fetch') && !String(e).includes('NetworkError')) {
        throw e;
      }
    }
    APP.staticMode = true;
  }
  const ds = payload.dataset;
  const sp = payload.target_species || 'small';
  const candidates = [
    `static_data/sim_${ds}_${method}_${sp}.json`,
    `static_data/sim_synthetic_drum_${method}_${sp}.json`,
    `static_data/sim_synthetic_drum_cartesian_${sp}.json`,
  ];
  for (const url of candidates) {
    const r = await fetch(url);
    if (r.ok) return await r.json();
  }
  throw new Error('Donnée statique indisponible');
}

async function fetchBenchmark(payload) {
  if (!APP.staticMode) {
    const resp = await fetch('/api/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => null);
    if (resp && resp.ok) {
      const data = await resp.json();
      if (data.error) throw new Error(data.error);
      return data;
    }
  }
  const sp = payload.target_species || 'small';
  const ds = payload.dataset || 'synthetic_drum';
  const r1 = await fetch(`static_data/benchmark_${ds}_${sp}.json`);
  const r = r1.ok ? r1 : await fetch(`static_data/benchmark_synthetic_drum_${sp}.json`);
  if (!r.ok) throw new Error('Benchmark statique indisponible');
  return await r.json();
}

function nearestEvalRow(bundle, ts) {
  const times = bundle.hom_metrics.times_eval;
  if (!times || !times.length) return 0;
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < times.length; i++) {
    const d = Math.abs(times[i] - ts);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

async function fetchStudyPresets() {
  if (!APP.staticMode) {
    try {
      const r = await fetch('/api/study/presets');
      if (r.ok) return await r.json();
    } catch (_) {}
  }
  try {
    const r = await fetch('static_data/study_presets.json');
    if (r.ok) return await r.json();
  } catch (_) {}
  return { presets: typeof STUDY_PRESET_FALLBACKS !== 'undefined' ? STUDY_PRESET_FALLBACKS : [] };
}

async function fetchStudyPresetSpec(presetId) {
  const method = APP.method;
  const base_request = currentRequest(method, gatherKwargs(method));
  if (!APP.staticMode) {
    try {
      const r = await fetch('/api/study/preset_spec', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset_id: presetId, base_request }),
      });
      if (r.ok) return await r.json();
    } catch (_) {}
  }
  return null;
}

async function fetchStudyRun(specPayload) {
  if (!APP.staticMode) {
    try {
      const resp = await fetch('/api/study/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(specPayload),
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        return data;
      }
      if (resp.status !== 404 && resp.status !== 405) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || 'Erreur étude Rust');
      }
    } catch (e) {
      if (!String(e).includes('Failed to fetch') && !String(e).includes('NetworkError')) {
        throw e;
      }
    }
    APP.staticMode = true;
  }
  const ds = specPayload.base_request?.dataset || 'synthetic_drum';
  const pid = specPayload.study_id || 'hom_start';
  for (const url of [
    `static_data/study_${pid}_${ds}.json`,
    `static_data/study_${pid}_synthetic_drum.json`,
    `static_data/study_hom_start_synthetic_drum.json`,
  ]) {
    const r = await fetch(url);
    if (r.ok) return await r.json();
  }
  throw new Error('Étude statique indisponible');
}
