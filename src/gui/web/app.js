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

function renderActiveView() {
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
  APP.executionStatus = 'computing';
  applyPanelVisibility();
  loader.classList.add('show');
  try {
    const method = document.getElementById('method').value;
    const bundle = await fetchSimulate(method, gatherKwargs(method));
    APP.bundleA = bundle;
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
    APP.executionStatus = 'error';
    alert('Erreur Rust: ' + e.message);
  } finally {
    loader.classList.remove('show');
    applyPanelVisibility();
  }
}

async function runStudy() {
  const loader = document.getElementById('loader');
  APP.executionStatus = 'computing';
  applyPanelVisibility();
  loader.classList.add('show');
  try {
    const spec = gatherStudySpecPayload();
    const res = await fetchStudyRun(spec);
    APP.studyBundle = res;
    APP.studySelectedCell = { idx1: 0, idx2: 0 };
    APP.executionStatus = 'idle';
    renderStudyView();
  } catch (e) {
    APP.executionStatus = 'error';
    alert('Erreur Étude: ' + e.message);
  } finally {
    loader.classList.remove('show');
    applyPanelVisibility();
  }
}

async function runDual() {
  const loader = document.getElementById('loader');
  APP.executionStatus = 'computing';
  applyPanelVisibility();
  loader.classList.add('show');
  try {
    const mA = document.getElementById('method').value;
    const mB = document.getElementById('method-b').value;
    const [bA, bB] = await Promise.all([
      fetchSimulate(mA, gatherKwargs(mA)),
      fetchSimulate(mB, defaultKwargs(mB)),
    ]);
    APP.bundleA = bA;
    APP.bundleB = bB;
    APP.executionStatus = 'idle';
    renderActiveView();
  } catch (e) {
    APP.executionStatus = 'error';
    alert('Erreur Dual-Mesh: ' + e.message);
  } finally {
    loader.classList.remove('show');
    applyPanelVisibility();
  }
}

async function runBenchmark() {
  const loader = document.getElementById('loader');
  APP.executionStatus = 'computing';
  applyPanelVisibility();
  loader.classList.add('show');
  try {
    const method = document.getElementById('method').value;
    const payload = currentRequest(method, gatherKwargs(method));
    payload.methods = ['cartesian', 'cylindrical', 'quantile', 'octree', 'voronoi', 'physics', 'adaptive'];
    const data = await fetchBenchmark(payload);
    APP.benchRows = data.rows;
    APP.executionStatus = 'idle';
    renderBenchmarkView();
  } catch (e) {
    APP.executionStatus = 'error';
    alert('Erreur Benchmark: ' + e.message);
  } finally {
    loader.classList.remove('show');
    applyPanelVisibility();
  }
}

async function initApp() {
  restoreUiLayout();
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
  renderMethodParams('cartesian');

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

  // Dataset sync between top strip and feature tree
  const syncDataset = val => {
    document.getElementById('dataset').value = val;
    document.getElementById('dataset-top').value = val;
    document.getElementById('start_index').value = val === 'real_chunks' ? 157 : 40;
    runContextualAction();
  };
  document.getElementById('dataset-top').addEventListener('change', e => syncDataset(e.target.value));
  document.getElementById('dataset').addEventListener('change', e => syncDataset(e.target.value));

  // Color mode sync between ribbon dropdown and feature tree
  const syncColorMode = val => {
    document.getElementById('color-mode').value = val;
    document.getElementById('ribbon-color-mode').value = val;
    render3DAll();
  };
  document.getElementById('color-mode').addEventListener('change', e => syncColorMode(e.target.value));
  document.getElementById('ribbon-color-mode').addEventListener('change', e => syncColorMode(e.target.value));

  // Exclusive 3-mode switcher
  document.querySelectorAll('#workbench-mode-switch .mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setWorkbenchMode(btn.dataset.mode));
  });

  // Primary contextual run triggers
  document.getElementById('method').addEventListener('change', e => {
    renderMethodParams(e.target.value);
    applyPanelVisibility();
  });
  document.getElementById('run-btn').addEventListener('click', runContextualAction);
  document.getElementById('run-dual-btn').addEventListener('click', runDual);
  document.getElementById('run-bench-btn').addEventListener('click', runBenchmark);
  document.getElementById('target-species').addEventListener('change', runPrimary);
  document.getElementById('full-duration').addEventListener('change', runPrimary);
  document.getElementById('no-species').addEventListener('change', runPrimary);

  // Study Workshop presets & schema-driven form gating
  const studyPresetData = await fetchStudyPresets();
  APP.studyPresets = studyPresetData.presets || [];
  const presetSel = document.getElementById('study-preset');
  if (presetSel) {
    presetSel.innerHTML = APP.studyPresets
      .map(p => `<option value="${p.id}" title="${p.description}">${p.title}</option>`)
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
  const expCsv = document.getElementById('study-export-csv');
  if (expCsv) expCsv.addEventListener('click', () => exportStudyData('csv'));
  const expJson = document.getElementById('study-export-json');
  if (expJson) expJson.addEventListener('click', () => exportStudyData('json'));

  // Layer toggles
  ['show-small', 'show-large', 'show-wire', 'show-faces', 'show-vel', 'show-edges', 'show-drum'].forEach(id => {
    document.getElementById(id).addEventListener('change', render3DAll);
  });

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

  // Timeline slider & playback
  document.getElementById('time-slider').addEventListener('input', e => {
    APP.frameIdx = parseInt(e.target.value) || 0;
    render3DAll();
  });
  document.getElementById('play-btn').addEventListener('click', () => {
    APP.playing = !APP.playing;
    document.getElementById('play-btn').textContent = APP.playing ? 'Pause' : 'Lecture';
    if (APP.playing) {
      APP.playTimer = setInterval(() => {
        if (!APP.bundleA) return;
        APP.frameIdx = (APP.frameIdx + 1) % APP.bundleA.frames.length;
        document.getElementById('time-slider').value = APP.frameIdx;
        render3DAll();
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
