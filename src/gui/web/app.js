function populateSelectors(b) {
  const bSel = document.getElementById('block-select');
  if (bSel) {
    bSel.innerHTML = b.blocks.map(bl =>
      `<option value="${bl.block_index}">B${bl.block_index} [${bl.t_start}..${bl.t_end}]</option>`
    ).join('');
  }

  const cSel = document.getElementById('teneur-cell-select');
  if (cSel) {
    cSel.innerHTML = `<option value="all">Top-${Math.min(5, b.top_cells.length)} cellules (superposées)</option>` +
      b.top_cells.map(c => `<option value="${c}">Cellule #${c}</option>`).join('');
  }

  const sSel = document.getElementById('evol-source-cell');
  if (sSel) {
    const nCells = b.mesh?.n_cells || 0;
    const ids = (b.active_cell_ids && b.active_cell_ids.length)
      ? b.active_cell_ids
      : Array.from({ length: nCells }, (_, i) => i);
    sSel.innerHTML = ids.map(cid => `<option value="${cid}">Source C#${cid}</option>`).join('');
  }

  const span = Math.max(1, b.dem_t_max - b.dem_t_min);
  const left = ((b.model_t_start - b.dem_t_min) / span) * 100;
  const width = ((b.model_t_end - b.model_t_start) / span) * 100;
  const bar = document.getElementById('model-window-bar');
  if (bar) {
    bar.style.left = `${Math.max(0, Math.min(100, left))}%`;
    bar.style.width = `${Math.max(2, Math.min(100 - left, width))}%`;
  }
}

function renderTelemetry(b) {
  const hm = b.hom_metrics, im = b.inhom_metrics;
  const fr = b.frames[APP.frameIdx] || b.frames[0];
  document.getElementById('horizon-badge').textContent =
    `Modèle [${b.model_t_start}..${b.model_t_end}] → Prédiction [${b.pred_t_start}..${b.pred_t_end}]`;
  document.getElementById('telemetry').innerHTML = `
    <div class="tpill"><span class="k">Maillage</span><span class="v soft">${b.method} (${b.diagnostics.n_visited}/${b.mesh.n_cells})</span></div>
    <div class="tpill"><span class="k">Particules</span><span class="v">${fr.n_small_total}(4mm)+${fr.n_large_total}(8mm)=${fr.n_total}</span></div>
    <div class="tpill"><span class="k">RMSE Teneur</span><span class="v soft">${hm.rmse_teneur.toFixed(4)}</span><span class="v ok">/ ${im.rmse_teneur.toFixed(4)}</span></div>
    <div class="tpill"><span class="k">L¹ moy</span><span class="v soft">${hm.l1_mean_target.toFixed(4)}</span><span class="v ok">/ ${im.l1_mean_target.toFixed(4)}</span></div>
    <div class="tpill"><span class="k">RMS disc</span><span class="v soft">${hm.rms_discrepancy_mean.toFixed(4)}</span><span class="v ok">/ ${im.rms_discrepancy_mean.toFixed(4)}</span></div>
  `;
}

function renderTimelineLabels() {
  const b = APP.bundleA;
  if (!b) return;
  const frame = b.frames[Math.min(APP.frameIdx, b.frames.length - 1)] || b.frames[0];
  document.getElementById('time-label').textContent = `t = ${frame.timestep}`;
  document.getElementById('cnt-small').textContent = frame.n_small_total;
  document.getElementById('cnt-large').textContent = frame.n_large_total;
  document.getElementById('cnt-total').textContent = frame.n_total;
}

function render3DAll() {
  renderViewCube();
  const b = APP.bundleA;
  if (!b) return;
  renderTimelineLabels();
  if (APP.activeView === 'summary') {
    renderViewport3D(document.getElementById('canvas3d-mini'), b, false);
    const frame = b.frames[Math.min(APP.frameIdx, b.frames.length - 1)] || b.frames[0];
    const lbl = document.getElementById('mini-3d-lbl');
    if (lbl) lbl.innerHTML = `<b style="color:#fff">${b.method}</b> · ${b.diagnostics.n_visited}/${b.mesh.n_cells} cell. · t=${frame.timestep}`;
  } else if (APP.activeView === 'mesh3d') {
    renderMixer3DView(b);
  } else if (APP.activeView === 'benchmark') {
    renderDualView();
  }
}

// ── Right inspector: live 3D legend showing the parameters that influence
//    the 3-D display. Reads from the central store, so it updates automatically
//    whenever any influencing variable changes (propagation).
const COLOR_MODE_SWATCH = {
  dem_teneur: '#9ec0dc', hom_teneur: '#7dd3bd', inhom_teneur: '#eec078',
  hom_err: '#e5989b', inhom_err: '#d8a48f', cell_rmse: '#b8b8e6',
  population: '#88c9c2', cell_id: '#c9a6e0',
};
const COLOR_MODE_LABEL = {
  dem_teneur: 'Teneur DEM (réf.)', hom_teneur: 'Teneur Homogène',
  inhom_teneur: 'Teneur Inhomogène', hom_err: 'Erreur Hom |ΔT|',
  inhom_err: 'Erreur Inhom |ΔT|', cell_rmse: 'Discrepancy RMSE',
  population: 'Population N_i', cell_id: 'Identifiant cellule',
};

function renderInspectorLegend() {
  const wrap = document.getElementById('inspector-legend');
  if (!wrap) return;
  const view = APP.activeView || 'summary';
  const is3D = view === 'summary' || view === 'mesh3d' || view === 'benchmark';
  const b = APP.bundleA;

  if (!b) {
    wrap.innerHTML = `<div class="legend-section"><h4>Inspecteur</h4>`
      + `<div class="legend-empty">Aucun modèle chargé. Lancez une simulation pour afficher la légende 3D.</div></div>`;
    return;
  }

  const frame = b.frames[Math.min(APP.frameIdx, b.frames.length - 1)] || b.frames[0];
  const rows = [];
  const row = (k, v) => `<div class="legend-row"><span class="lk">${k}</span><span class="lv">${v}</span></div>`;

  if (is3D) {
    const cm = APP.colorMode || 'dem_teneur';
    const sw = COLOR_MODE_SWATCH[cm] || '#8faec9';
    rows.push(`<div class="legend-row"><span class="lk">Coloration</span>`
      + `<span class="lv"><span class="legend-swatch" style="background:${sw};"></span>${COLOR_MODE_LABEL[cm] || cm}</span></div>`);
    rows.push(row('Méthode', APP.method));
    rows.push(row('Jeu de données', APP.dataset));
    rows.push(row('Espèce cible', APP.targetSpecies));
    rows.push(row('Horizon τ', APP.tau));
    const layer = (label, key, color) => {
      const on = APP[key];
      const c = color ? `<span class="legend-swatch" style="background:${color};"></span>` : '';
      return `<div class="legend-row"><span class="lk">${c}${label}</span><span class="lv" style="color:${on ? 'var(--accent-teal)' : 'var(--dim)'}">${on ? 'ON' : 'OFF'}</span></div>`;
    };
    rows.push(layer('Petites (4 mm)', 'showSmall', '#7dd3c8'));
    rows.push(layer('Grosses (8 mm)', 'showLarge', '#eec078'));
    rows.push(layer('Faces', 'showFaces'));
    rows.push(layer('Fils (arêtes)', 'showWire'));
    rows.push(layer('Flux Markov', 'showEdges'));
    rows.push(layer('Vecteurs V', 'showVel'));
    rows.push(layer('Paroi tambour', 'showDrum'));
    rows.push(row('Cellules', `${b.mesh.n_cells} · visitées ${b.diagnostics.n_visited}`));
    rows.push(row('Frame', `${APP.frameIdx} / ${b.frames.length - 1}`));
    if (frame) rows.push(row('t', frame.timestep));
  } else {
    rows.push(row('Vue', view));
    rows.push(row('Méthode', APP.method));
    rows.push(row('Jeu de données', APP.dataset));
    rows.push(row('Espèce cible', APP.targetSpecies));
    rows.push(row('Cellules', `${b.mesh.n_cells} · visitées ${b.diagnostics.n_visited}`));
    rows.push(row('Frame', `${APP.frameIdx} / ${b.frames.length - 1}`));
  }

  // Selection properties (shown when a cell is picked in the 3-D view).
  let selectionHtml = '';
  if (APP.selectedCell != null && frame) {
    const cid = APP.selectedCell;
    const cell = (b.mesh.cells || []).find(cc => cc.id === cid);
    const ei = nearestEvalRow(b, frame.timestep);
    const tD = b.dem_conc_traj[ei]?.[cid] ?? frame.cell_conc_dem?.[cid] ?? 0;
    const tH = b.hom_conc_traj[ei]?.[cid] ?? 0;
    const tI = b.inhom_conc_traj[ei]?.[cid] ?? 0;
    const crm = b.inhom_metrics.cell_rmse_per_cell?.[cid] ?? 0;
    const pop = frame.cell_pop?.[cid] ?? 0;
    selectionHtml = `<div class="legend-section"><h4>Sélection · Cellule #${cid}</h4>`
      + row('Zone', cell ? cell.zone : '—')
      + row('Particules N', pop)
      + row('Teneur DEM', Number(tD).toFixed(3))
      + row('Teneur Hom', Number(tH).toFixed(3))
      + row('Teneur Inhom', Number(tI).toFixed(3))
      + row('RMSE cell.', Number(crm).toFixed(4))
      + `</div>`;
  }

  wrap.innerHTML = selectionHtml + `<div class="legend-section"><h4>Paramètres d'affichage</h4>${rows.join('')}</div>`;
}

// Re-render the legend whenever an influencing variable or the active view changes.
['activeView', 'colorMode', 'method', 'dataset', 'targetSpecies', 'tau', 'nlt', 'step', 'dt', 'selectedCell',
  'showSmall', 'showLarge', 'showWire', 'showFaces', 'showEdges', 'showVel', 'showDrum',
  'frameIdx'].forEach(key => {
  STORE.subscribe(key, () => renderInspectorLegend());
});

// ── Status bar (Fusion 360-style contextual status) ──────────────────────
const VIEW_NAMES = {
  summary: 'Sommaire', mesh3d: 'Mélangeur 3D', teneur: 'Teneur', particles: 'Particules',
  errors: 'Erreurs', matrix_evol: 'Évolution P(t)', matrices: 'Matrices',
  benchmark: 'Comparateur', study: 'Études', plan: 'Architecture',
};
const MODE_NAMES = { explore: 'Explorer', study: 'Études', compare: 'Comparer' };

function renderStatusBar() {
  const set = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };
  set('sb-view', `Vue : <b>${VIEW_NAMES[APP.activeView] || APP.activeView || '—'}</b>`);
  set('sb-mode', `Mode : <b>${MODE_NAMES[APP.workbenchMode] || APP.workbenchMode || '—'}</b>`);
  const sel = APP.selectedCell;
  set('sb-selection', sel != null ? `Sélection : <b>Cellule #${sel}</b>` : 'Sélection : —');
  const b = APP.bundleA;
  if (b && b.frames.length) {
    const nf = b.frames.length - 1;
    set('sb-frame', `Frame <b>${Math.min(APP.frameIdx, nf)}</b> / ${nf}`);
    const fp = b.frames[Math.min(APP.frameIdx, nf)] || b.frames[0];
    const total = (fp.cell_pop || []).reduce((a, x) => a + (x || 0), 0);
    set('sb-counts', `${total} particules`);
  } else {
    set('sb-frame', 'Frame <b>0</b> / 0');
    set('sb-counts', '<b>0</b> particules');
  }
  set('sb-theme', `Thème : <b>${APP.theme === 'light' ? 'clair' : 'sombre'}</b>`);
}

// Keep the status bar in sync with any contextual change.
['activeView', 'workbenchMode', 'selectedCell', 'frameIdx', 'theme'].forEach(key => {
  STORE.subscribe(key, () => renderStatusBar());
});

function renderActiveView() {
  renderInspectorLegend();
  renderStatusBar();
  const b = APP.bundleA;
  if (!b) return;
  renderTelemetry(b);
  renderTimelineLabels();

  if (APP.activeView === 'summary') renderSummaryView(b);
  else if (APP.activeView === 'mesh3d') renderMixer3DView(b);
  else if (APP.activeView === 'teneur') renderTeneurView(b);
  else if (APP.activeView === 'particles') renderParticlesView(b);
  else if (APP.activeView === 'errors') renderErrorsView(b);
  else if (APP.activeView === 'matrix_evol') renderMatrixEvolutionView(b);
  else if (APP.activeView === 'matrices') renderMatricesView(b);
  else if (APP.activeView === 'benchmark') renderBenchmarkView();
  else if (APP.activeView === 'study') renderStudyView();
}

function setWorkbenchMode(mode) {
  if (mode !== 'explore' && mode !== 'study' && mode !== 'compare') return;
  APP.workbenchMode = mode;
  if (mode === 'study') {
    switchView('study');
    if (!APP.studyBundle && typeof runStudy === 'function') {
      runStudy();
    }
  } else if (mode === 'compare') {
    switchView('benchmark');
  } else {
    if (APP.activeView === 'study' || APP.activeView === 'benchmark') {
      switchView('summary');
    } else {
      applyPanelVisibility();
      saveUiLayout();
    }
  }
}

function switchView(viewId) {
  if (!document.getElementById('pane-' + viewId)) return;
  APP.activeView = viewId;
  if (viewId === 'study') APP.workbenchMode = 'study';
  else if (viewId === 'benchmark') APP.workbenchMode = 'compare';
  else if (APP.workbenchMode !== 'explore') APP.workbenchMode = 'explore';

  document.querySelectorAll('.vtab').forEach(b => b.classList.toggle('active', b.dataset.view === viewId));
  document.querySelectorAll('.view-pane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + viewId));
  applyPanelVisibility();
  saveUiLayout();

  if (viewId === 'benchmark' && !APP.bundleB) {
    runDual();
  } else if (viewId === 'study' && !APP.studyBundle) {
    runStudy();
  } else {
    setTimeout(() => renderActiveView(), 18);
  }
}

async function runContextualAction() {
  if (APP.workbenchMode === 'study') return runStudy();
  if (APP.workbenchMode === 'compare') return runDual();
  return runPrimary();
}

async function runPrimary() {
  const loader = document.getElementById('loader');
  const token = beginRun();
  APP.executionStatus = 'computing';
  applyPanelVisibility();
  loader.classList.add('show');
  try {
    const method = APP.method;
    const bundle = await fetchSimulate(method, gatherKwargs(method));
    if (!runIsCurrent(token)) return; // superseded by a newer run
    APP.bundleA = bundle;
    toast(`Simulation OK · ${method} · ${bundle.diagnostics.n_visited}/${bundle.mesh.n_cells} cellules visitées`, 'success', 4000);
    APP.frameIdx = 0;
    APP.selectedCell = null;
    APP.selectedTransitionKey = null;
    APP.evolZoomedKey = null;
    APP.evolPage = 0;
    fitCameraToMesh(bundle.mesh);
    const slider = document.getElementById('time-slider');
    slider.max = Math.max(0, bundle.frames.length - 1);
    slider.value = 0;
    populateSelectors(bundle);
    updateCellInspector(null);
    APP.executionStatus = 'idle';
    renderActiveView();
  } catch (e) {
    if (!runIsCurrent(token)) return; // stale failure, ignore
    APP.executionStatus = 'error';
    toast('Erreur Rust : ' + e.message, 'error', 6000);
  } finally {
    if (runIsCurrent(token)) { loader.classList.remove('show'); applyPanelVisibility(); }
  }
}

async function runStudy() {
  const loader = document.getElementById('loader');
  const token = beginRun();
  APP.executionStatus = 'computing';
  applyPanelVisibility();
  loader.classList.add('show');
  try {
    const spec = gatherStudySpecPayload();
    const res = await fetchStudyRun(spec);
    if (!runIsCurrent(token)) return; // superseded by a newer run
    APP.studyBundle = res;
    APP.studySelectedCell = { idx1: 0, idx2: 0 };
    toast(`Étude OK · ${res.method} · ${res.diagnostics.n_visited} cellules visitées`, 'success', 4000);
    APP.executionStatus = 'idle';
    renderStudyView();
  } catch (e) {
    if (!runIsCurrent(token)) return; // stale failure, ignore
    APP.executionStatus = 'error';
    toast('Erreur Étude : ' + e.message, 'error', 6000);
  } finally {
    if (runIsCurrent(token)) { loader.classList.remove('show'); applyPanelVisibility(); }
  }
}

async function runDual() {
  const loader = document.getElementById('loader');
  const token = beginRun();
  APP.executionStatus = 'computing';
  applyPanelVisibility();
  loader.classList.add('show');
  try {
    const mA = APP.method;
    const mB = APP.methodB;
    const [bA, bB] = await Promise.all([
      fetchSimulate(mA, gatherKwargs(mA)),
      fetchSimulate(mB, defaultKwargs(mB)),
    ]);
    if (!runIsCurrent(token)) return; // superseded by a newer run
    APP.bundleA = bA;
    APP.bundleB = bB;
    toast(`Dual-Mesh OK · ${mA} ⊗ ${mB}`, 'success', 4000);
    APP.executionStatus = 'idle';
    renderActiveView();
  } catch (e) {
    if (!runIsCurrent(token)) return; // stale failure, ignore
    APP.executionStatus = 'error';
    toast('Erreur Dual-Mesh : ' + e.message, 'error', 6000);
  } finally {
    if (runIsCurrent(token)) { loader.classList.remove('show'); applyPanelVisibility(); }
  }
}

async function runBenchmark() {
  const loader = document.getElementById('loader');
  const token = beginRun();
  APP.executionStatus = 'computing';
  applyPanelVisibility();
  loader.classList.add('show');
  try {
    const method = APP.method;
    const payload = currentRequest(method, gatherKwargs(method));
    payload.methods = ['cartesian', 'cylindrical', 'quantile', 'octree', 'voronoi', 'physics', 'adaptive'];
    const data = await fetchBenchmark(payload);
    if (!runIsCurrent(token)) return; // superseded by a newer run
    APP.benchRows = data.rows;
    toast(`Benchmark OK · ${data.rows.length} méthodes évaluées`, 'success', 4000);
    APP.executionStatus = 'idle';
    renderBenchmarkView();
  } catch (e) {
    if (!runIsCurrent(token)) return; // stale failure, ignore
    APP.executionStatus = 'error';
    toast('Erreur Benchmark : ' + e.message, 'error', 6000);
  } finally {
    if (runIsCurrent(token)) { loader.classList.remove('show'); applyPanelVisibility(); }
  }
}

// ── Theme (dark / light) ──────────────────────────────────────────────────
function applyTheme(theme) {
  const t = theme === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = t;
  const icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = t === 'light' ? '☀' : '☾';
  const btn = document.getElementById('theme-toggle-btn');
  if (btn) btn.title = t === 'light' ? 'Passer en sombre' : 'Passer en clair';
}
// Persist the theme choice and keep the DOM in sync with the store so the
// theme updates propagate from any part of the software.
STORE.subscribe('theme', applyTheme);

async function initApp() {
  restoreUiLayout();
  // Restore the saved simulation workspace (method, dataset, integration
  // params) before binding controls so bound inputs display the saved values.
  restoreWorkspace();
  // Restore the user's theme choice (persisted) and apply it before first paint.
  try {
    const savedTheme = localStorage.getItem('dem_mcm_theme');
    if (savedTheme === 'light' || savedTheme === 'dark') APP.theme = savedTheme;
  } catch (_) {}
  applyTheme(APP.theme);
  // Toggle button: flips the store theme (which propagates via the subscription).
  const themeBtn = document.getElementById('theme-toggle-btn');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      APP.theme = APP.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem('dem_mcm_theme', APP.theme); } catch (_) {}
    });
  }
  applyPanelVisibility();

  let data = null;
  try {
    const res = await fetch('/api/methods');
    if (res.ok) data = await res.json();
  } catch (_) {}
  if (!data) {
    APP.staticMode = true;
    const res = await fetch('static_data/methods.json');
    data = await res.json();
    const badge = document.querySelector('#brand-badge span:last-child');
    if (badge) badge.textContent = 'DEM-MCM Studio · Statique';
  }
  APP.methods = data.methods;

  const opts = APP.methods.map(m => `<option value="${m.id}">${m.name}</option>`).join('');
  document.getElementById('method').innerHTML = opts;
  document.getElementById('method-b').innerHTML = opts;
  document.getElementById('method-b').value = 'voronoi';
  renderMethodParams(APP.method);

  await loadPlanText();

  // Collapsible Panel Toggles (CAD Shell)
  document.getElementById('toggle-tree-btn').addEventListener('click', () => {
    APP.ui.treeOpen = !APP.ui.treeOpen;
    applyPanelVisibility();
    saveUiLayout();
    setTimeout(renderActiveView, 180);
  });
  document.getElementById('toggle-ribbon-btn').addEventListener('click', () => {
    APP.ui.ribbonOpen = !APP.ui.ribbonOpen;
    applyPanelVisibility();
    saveUiLayout();
    setTimeout(renderActiveView, 40);
  });
  document.getElementById('toggle-timeline-btn').addEventListener('click', () => {
    APP.ui.timelineOpen = !APP.ui.timelineOpen;
    applyPanelVisibility();
    saveUiLayout();
    setTimeout(renderActiveView, 40);
  });
  document.getElementById('inspector-toggle-btn').addEventListener('click', () => {
    APP.ui.inspectorOpen = !APP.ui.inspectorOpen;
    applyPanelVisibility();
    saveUiLayout();
    setTimeout(renderInspectorLegend, 40);
  });
  document.getElementById('collapse-all-nodes-btn').addEventListener('click', () => {
    const nodes = Array.from(document.querySelectorAll('details.tree-node'));
    const anyOpen = nodes.some(n => n.open);
    nodes.forEach(n => { n.open = !anyOpen; });
    saveUiLayout();
  });
  document.querySelectorAll('details.tree-node').forEach(n => {
    n.addEventListener('toggle', saveUiLayout);
  });

  // Close open ribbon dropdowns when clicking outside
  document.addEventListener('click', e => {
    document.querySelectorAll('details.cad-dropdown[open]').forEach(d => {
      if (!d.contains(e.target)) d.open = false;
    });
  });

  // Navigation Tabs & Expand/Jump Buttons
  document.querySelectorAll('.vtab').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });
  document.querySelectorAll('[data-jump-view]').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.jumpView));
  });
  document.querySelectorAll('[data-cam]').forEach(btn => {
    btn.addEventListener('click', () => {
      setCamPreset(btn.dataset.cam);
      render3DAll();
    });
  });

  // View cube: clicking a face snaps the camera to that orientation.
  document.querySelectorAll('.vc-face').forEach(btn => {
    btn.addEventListener('click', () => {
      APP.cam.yaw = parseFloat(btn.dataset.yaw) || 0;
      APP.cam.pitch = parseFloat(btn.dataset.pitch) || 0;
      render3DAll();
    });
  });

  // Debounced contextual run: parameter changes (typing numerics, switching
  // method/species/dataset) must NOT fire one full server simulation per
  // keystroke. A short pause coalesces bursts into a single up-to-date run.
  const runSoon = debounce(() => runContextualAction(), 350);

  // Dataset: both controls share ONE store key — changing either one
  // updates the store and propagates to the other control + the run action.
  bindControl('dataset', document.getElementById('dataset'));
  bindControl('dataset', document.getElementById('dataset-top'));
  bindControl('startIndex', document.getElementById('start_index'), { parse: v => parseInt(v) || 0 });
  STORE.subscribe('dataset', val => {
    const si = val === 'real_chunks' ? 157 : 40;
    STORE.set('startIndex', si);
    runSoon();
  });

  // Color mode: both controls share ONE store key and re-render 3D.
  bindControl('colorMode', document.getElementById('color-mode'));
  bindControl('colorMode', document.getElementById('ribbon-color-mode'));
  STORE.subscribe('colorMode', () => render3DAll());

  // Exclusive 3-mode switcher
  document.querySelectorAll('#workbench-mode-switch .mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setWorkbenchMode(btn.dataset.mode));
  });

  // Bind all shared simulation controls to the central store. Each control
  // writes into the store on change; the store then propagates the value to
  // every other part of the software (views, other controls, run actions).
  bindControl('method', document.getElementById('method'));
  bindControl('methodB', document.getElementById('method-b'));
  bindControl('targetSpecies', document.getElementById('target-species'));
  bindControl('noSpecies', document.getElementById('no-species'));
  bindControl('predictFullDuration', document.getElementById('full-duration'));
  bindControl('nlt', document.getElementById('nlt'), { parse: v => parseInt(v) || 0 });
  bindControl('step', document.getElementById('step'), { parse: v => parseInt(v) || 0 });
  bindControl('dt', document.getElementById('dt'), { parse: v => parseInt(v) || 0 });
  bindControl('tau', document.getElementById('tau'), { parse: v => parseInt(v) || 0 });
  // Persist the workspace: save on any parameter change (debounced) and
  // again on reload/close so nothing is lost.
  WORKSPACE_KEYS.forEach(key => STORE.subscribe(key, () => _workspaceSave()));
  window.addEventListener('beforeunload', saveWorkspace);

  bindControl('showSmall', document.getElementById('show-small'));
  bindControl('showLarge', document.getElementById('show-large'));
  bindControl('showWire', document.getElementById('show-wire'));
  bindControl('showFaces', document.getElementById('show-faces'));
  bindControl('showVel', document.getElementById('show-vel'));
  bindControl('showEdges', document.getElementById('show-edges'));
  bindControl('showDrum', document.getElementById('show-drum'));

  // Primary contextual run triggers
  document.getElementById('run-btn').addEventListener('click', () => { runSoon.cancel(); runContextualAction(); });
  document.getElementById('run-dual-btn').addEventListener('click', runDual);
  document.getElementById('run-bench-btn').addEventListener('click', runBenchmark);

  // Method change → refresh param fields, panel visibility, then re-run the
  // CONTEXTUAL action (runStudy in study mode, runDual in compare mode,
  // runPrimary in explore mode) so the correct executor runs.
  STORE.subscribe('method', () => {
    renderMethodParams(STORE.get('method'));
    applyPanelVisibility();
    runSoon();
  });
  // Species / duration / no-species / numeric params → re-run the CONTEXTUAL
  // action (NOT always runPrimary) so a change in study mode re-executes the
  // study, in compare mode re-runs the dual sim, and in explore the primary.
  ['targetSpecies', 'predictFullDuration', 'noSpecies', 'nlt', 'step', 'dt', 'tau'].forEach(key => {
    STORE.subscribe(key, () => runSoon());
  });
  // Partitioning parameters (method kwargs) live in the store and are shared
  // with the visualization: editing one re-runs the contextual simulation so
  // the 3-D visuals stay synchronized with the parameters.
  STORE.subscribe('methodKwargs', () => runSoon());
  // 3-D layer toggles → re-render 3D from the store values
  ['showSmall', 'showLarge', 'showWire', 'showFaces', 'showVel', 'showEdges', 'showDrum'].forEach(key => {
    STORE.subscribe(key, () => render3DAll());
  });

  // Study Workshop presets & schema-driven form gating
  const studyPresetData = await fetchStudyPresets();
  APP.studyPresets = studyPresetData.presets || [];
  const presetSel = document.getElementById('study-preset');
  if (presetSel) {
    presetSel.innerHTML = APP.studyPresets
      .map(p => `<option value="${p.id}" title="${p.description || ''}">${p.title || p.name || p.id}</option>`)
      .join('');
    presetSel.addEventListener('change', async () => {
      const spec = await fetchStudyPresetSpec(presetSel.value);
      if (spec) populateStudyFormFromSpec(spec);
      else applyPresetFallbackToBuilder(presetSel.value);
      applyPanelVisibility();
    });
  }
  ['study-axis1-kind', 'study-axis2-kind', 'study-enable-laws'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', () => applyPanelVisibility());
  });
  ['study-curve-chain', 'study-sens-metric', 'study-heat-metric', 'study-transition-select'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', () => renderStudyView());
  });
  const rangeBtn = document.getElementById('study-apply-range-btn');
  if (rangeBtn) {
    rangeBtn.addEventListener('click', () => {
      const a = parseFloat(document.getElementById('study-range-min').value) || 0;
      const b = parseFloat(document.getElementById('study-range-max').value) || 100;
      const n = Math.max(2, Math.min(15, parseInt(document.getElementById('study-range-count').value) || 6));
      const vals = [];
      for (let i = 0; i < n; i++) {
        const v = a + ((b - a) * i) / (n - 1);
        vals.push(Number.isInteger(a) && Number.isInteger(b) ? Math.round(v) : Number(v.toFixed(3)));
      }
      document.getElementById('study-axis1-values').value = vals.join(', ');
    });
  }
  const studyToggleBtn = document.getElementById('study-toggle-builder-btn');
  if (studyToggleBtn) {
    studyToggleBtn.addEventListener('click', () => {
      APP.ui.studyBuilderOpen = !APP.ui.studyBuilderOpen;
      applyPanelVisibility();
      setTimeout(renderStudyView, 160);
    });
  }
  const studyCollapseBtn = document.getElementById('study-collapse-all-btn');
  if (studyCollapseBtn) {
    studyCollapseBtn.addEventListener('click', () => {
      const nodes = Array.from(
        document.querySelectorAll('#study-sidebar details.tree-node')
      );
      const anyOpen = nodes.some(n => n.open);
      nodes.forEach(n => { n.open = !anyOpen; });
      saveUiLayout();
    });
  }
  const expCsv = document.getElementById('study-export-csv');
  if (expCsv) expCsv.addEventListener('click', () => exportStudyData('csv'));
  const expJson = document.getElementById('study-export-json');
  if (expJson) expJson.addEventListener('click', () => exportStudyData('json'));

  // View-specific filters
  ['teneur-cell-select', 'teneur-global-metric', 'npart-species', 'npart-cell-count', 'l1-scope', 'heat-disc-chain', 'mat-view', 'block-select'].forEach(id => {
    document.getElementById(id).addEventListener('change', renderActiveView);
  });
  document.getElementById('bench-metric-key').addEventListener('change', renderBenchmarkView);

  // Matrix evolution filters
  ['evol-mode', 'evol-scope', 'evol-source-cell', 'evol-species', 'evol-threshold', 'evol-sort', 'evol-xaxis', 'evol-show-fits', 'evol-common-y'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => {
      APP.evolPage = 0;
      renderMatrixEvolutionView(APP.bundleA);
    });
  });
  document.getElementById('evol-prev-page').addEventListener('click', () => {
    APP.evolPage = Math.max(0, APP.evolPage - 1);
    renderMatrixEvolutionView(APP.bundleA);
  });
  document.getElementById('evol-next-page').addEventListener('click', () => {
    APP.evolPage += 1;
    renderMatrixEvolutionView(APP.bundleA);
  });
  document.getElementById('evol-clear-zoom').addEventListener('click', () => {
    APP.evolZoomedKey = null;
    renderMatrixEvolutionView(APP.bundleA);
  });
  document.getElementById('evol-reset-selection').addEventListener('click', () => {
    APP.selectedTransitionKey = null;
    APP.evolZoomedKey = null;
    renderMatrixEvolutionView(APP.bundleA);
  });

  // Timeline slider & playback: the slider is bound to the central store
  // key `frameIdx`; every change (from the slider OR from playback) updates
  // the store and propagates to the 3-D views and the slider itself.
  bindControl('frameIdx', document.getElementById('time-slider'), { parse: v => parseInt(v) || 0 });
  STORE.subscribe('frameIdx', () => render3DAll());
  document.getElementById('play-btn').addEventListener('click', () => {
    APP.playing = !APP.playing;
    document.getElementById('play-btn').textContent = APP.playing ? 'Pause' : 'Lecture';
    if (APP.playing) {
      APP.playTimer = setInterval(() => {
        if (!APP.bundleA) return;
        APP.frameIdx = (APP.frameIdx + 1) % APP.bundleA.frames.length;
      }, 130);
    } else {
      clearInterval(APP.playTimer);
    }
  });

  // 3D canvas interactions
  wire3DEvents(document.getElementById('canvas3d'), render3DAll, updateCellInspector);
  wire3DEvents(document.getElementById('canvas3d-mini'), render3DAll, updateCellInspector);
  wire3DEvents(document.getElementById('canvas3d-a'), render3DAll, null);
  wire3DEvents(document.getElementById('canvas3d-b'), render3DAll, null);

  window.addEventListener('resize', () => renderActiveView());

  switchView(APP.activeView);
  await runPrimary();
}

initApp();
