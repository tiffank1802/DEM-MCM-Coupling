// Dynamic Research Study Workshop (`Mode: Études de Recherche`).
// Handles 1-D parameter sweeps, 2-D crossed grids (start_index x NLT),
// partitioning-method vs RSD comparisons, and multi-family p_{i->j}(t) law evaluation.

const STUDY_PRESET_FALLBACKS = [
  { id: 'hom_start', name: '1. Homogène · Départ d\'apprentissage (start_index)', description: 'Transitoire initial vs régime permanent sur RSD(t) et Teneur.' },
  { id: 'hom_nlt_frob', name: '2. Homogène · Nombre de blocs (NLT) & Frobenius', description: 'Convergence de Frobenius et erreur RSD selon NLT.' },
  { id: 'hom_step_dt', name: '3. Homogène · Raffinage (dt) & Lignes NaN', description: 'Impact du pas dt sur les états non visités (NaN) et le mélange.' },
  { id: 'hom_step', name: '4. Homogène · Écart inter-blocs (step)', description: 'Sensibilité à l\'espacement entre blocs d\'apprentissage.' },
  { id: 'tau_sensitivity', name: '5. Horizon Markov τ · Cinétique RSD & t50', description: 'Effet spatial & temporel de τ sur la prédiction.' },
  { id: 'method_rsd', name: '6. Découpage spatial · Méthodes × RSD(t)', description: 'Comparaison dynamique des méthodes sur le RSD.' },
  { id: 'spatial_resolution', name: '7. Découpage spatial · Résolution (n_cells / nx)', description: 'Compromis résolution spatiale vs erreur RSD et NaN.' },
  { id: 'inhom_laws_grid', name: '8. Inhomogène · Lois p_{i→j}(t) : carte (start × NLT)', description: 'Compare 7 familles de lois selon start_index et NLT.' },
];

function parseAxisValueList(raw, kind) {
  const tokens = String(raw || '')
    .split(/[,;\s]+/)
    .map(s => s.trim())
    .filter(Boolean);
  if (kind === 'method') return tokens;
  if (kind === 'species_mode') return [false, true];
  if (kind === 'velocity_weight') return tokens.map(Number).filter(n => isFinite(n) && n >= 0);
  return tokens.map(Number).filter(n => isFinite(n) && n >= 0).map(n => Math.round(n));
}

function getStudyBuilderState() {
  const presetId = document.getElementById('study-preset')?.value || 'hom_start';
  const axis1Kind = document.getElementById('study-axis1-kind')?.value || 'start_index';
  const axis2Kind = document.getElementById('study-axis2-kind')?.value || 'none';
  const chainMode = document.getElementById('study-chain-mode')?.value || 'homogeneous';
  const enableLawStudy = document.getElementById('study-enable-laws')?.checked || false;
  const coupleTau = document.getElementById('study-couple-tau')?.checked || false;
  const lawCriterion = document.getElementById('study-law-criterion')?.value || 'aicc';

  return {
    presetId,
    axis1Kind,
    axis2Kind,
    chainMode,
    enableLawStudy,
    coupleTau,
    lawCriterion,
  };
}

function populateStudyFormFromSpec(spec) {
  if (!spec) return;
  const ax1 = spec.primary_axis;
  const ax2 = spec.secondary_axis;

  const a1Sel = document.getElementById('study-axis1-kind');
  const a1Vals = document.getElementById('study-axis1-values');
  const coupleEl = document.getElementById('study-couple-tau');
  const chainEl = document.getElementById('study-chain-mode');
  const lawChk = document.getElementById('study-enable-laws');
  const critEl = document.getElementById('study-law-criterion');
  const a2Sel = document.getElementById('study-axis2-kind');
  const a2Vals = document.getElementById('study-axis2-values');

  if (a1Sel && ax1) a1Sel.value = ax1.kind;
  if (a1Vals && ax1) {
    if (ax1.kind === 'method') {
      document.querySelectorAll('#study-methods-list input[type="checkbox"]').forEach(cb => {
        cb.checked = (ax1.values || []).includes(cb.value);
      });
    } else {
      a1Vals.value = (ax1.values || []).join(', ');
    }
  }
  if (coupleEl && ax1) coupleEl.checked = Boolean(ax1.couple_step_to_tau);
  if (chainEl) chainEl.value = spec.chain_mode || 'homogeneous';
  if (lawChk) lawChk.checked = Boolean(spec.enable_law_study);
  if (critEl && spec.law_criterion) critEl.value = spec.law_criterion;

  if (a2Sel) a2Sel.value = ax2 ? ax2.kind : 'none';
  if (a2Vals) a2Vals.value = ax2 ? (ax2.values || []).join(', ') : '3, 5, 8';

  if (Array.isArray(spec.law_families)) {
    const active = new Set(spec.law_families);
    document.querySelectorAll('#study-law-families input[type="checkbox"]').forEach(cb => {
      cb.checked = active.has(cb.value);
    });
  }
}

function applyPresetFallbackToBuilder(presetId) {
  const fallbacks = {
    hom_start: { chain_mode: 'homogeneous', enable_law_study: false, primary_axis: { kind: 'start_index', values: [0, 20, 40, 80, 120, 180] }, secondary_axis: null },
    hom_nlt_frob: { chain_mode: 'homogeneous', enable_law_study: false, primary_axis: { kind: 'nlt', values: [1, 2, 3, 5, 8, 12] }, secondary_axis: null },
    hom_step_dt: { chain_mode: 'homogeneous', enable_law_study: false, primary_axis: { kind: 'dt', values: [1, 2, 3, 6, 12] }, secondary_axis: null },
    hom_step: { chain_mode: 'homogeneous', enable_law_study: false, primary_axis: { kind: 'step', values: [5, 10, 20, 40, 60, 100] }, secondary_axis: null },
    tau_sensitivity: { chain_mode: 'homogeneous', enable_law_study: false, primary_axis: { kind: 'tau', values: [5, 10, 20, 40, 60], couple_step_to_tau: true }, secondary_axis: null },
    method_rsd: { chain_mode: 'homogeneous', enable_law_study: false, primary_axis: { kind: 'method', values: ['cartesian', 'cylindrical', 'quantile', 'octree', 'voronoi', 'physics', 'adaptive'] }, secondary_axis: null },
    spatial_resolution: { chain_mode: 'homogeneous', enable_law_study: false, primary_axis: { kind: 'spatial_resolution', values: [4, 8, 12, 16, 24, 32] }, secondary_axis: null },
    inhom_laws_grid: { chain_mode: 'inhomogeneous', enable_law_study: true, law_criterion: 'aicc', primary_axis: { kind: 'start_index', values: [0, 40, 80, 120] }, secondary_axis: { kind: 'nlt', values: [3, 5, 7] } },
  };
  if (fallbacks[presetId]) populateStudyFormFromSpec(fallbacks[presetId]);
}

function gatherStudySpecPayload() {
  const st = getStudyBuilderState();
  const method = APP.method;
  const base_request = currentRequest(method, gatherKwargs(method));

  let axis1Values = [];
  if (st.axis1Kind === 'method') {
    axis1Values = Array.from(
      document.querySelectorAll('#study-methods-list input[type="checkbox"]:checked')
    ).map(cb => cb.value);
    if (!axis1Values.length) axis1Values = ['cartesian', 'cylindrical', 'voronoi'];
  } else if (st.axis1Kind === 'species_mode') {
    axis1Values = [false, true];
  } else {
    axis1Values = parseAxisValueList(
      document.getElementById('study-axis1-values')?.value || '0, 40, 80',
      st.axis1Kind
    );
    if (!axis1Values.length) axis1Values = [0, 40, 80];
  }

  let secondary_axis = null;
  if (st.axis2Kind && st.axis2Kind !== 'none') {
    const a2Vals = parseAxisValueList(
      document.getElementById('study-axis2-values')?.value || '3, 5, 7',
      st.axis2Kind
    );
    if (a2Vals.length) {
      secondary_axis = {
        kind: st.axis2Kind,
        values: a2Vals,
        couple_step_to_tau: false,
      };
    }
  }

  const law_families = Array.from(
    document.querySelectorAll('#study-law-families input[type="checkbox"]:checked')
  ).map(cb => cb.value);

  const focusRaw = document.getElementById('study-transition-select')?.value || '';
  let focus_transition = null;
  if (focusRaw.includes('->')) {
    const [s, d] = focusRaw.split('->').map(Number);
    if (isFinite(s) && isFinite(d)) focus_transition = { src: s, dst: d };
  }

  return {
    study_id: st.presetId,
    base_request,
    chain_mode: st.chainMode,
    primary_axis: {
      kind: st.axis1Kind,
      values: axis1Values,
      couple_step_to_tau: st.coupleTau,
    },
    secondary_axis,
    use_bed_frame: document.getElementById('study-bed-frame')?.checked || false,
    enable_law_study: st.enableLawStudy,
    law_families: law_families.length ? law_families : undefined,
    law_criterion: st.lawCriterion,
    focus_transition,
    reference_policy: st.axis1Kind === 'dt' ? 'first' : 'last',
    include_curves: true,
  };
}

function renderStudyView() {
  const b = APP.studyBundle;
  const titleEl = document.getElementById('study-title-lbl');
  const subEl = document.getElementById('study-subtitle-lbl');
  if (!b) {
    if (titleEl) titleEl.textContent = "Atelier d'Études Paramétriques & Lois";
    if (subEl) subEl.textContent = "Choisissez un preset ou configurez un axe et cliquez sur « Lancer l'étude ».";
    return;
  }

  if (titleEl) titleEl.textContent = b.title || "Étude";
  if (subEl) subEl.textContent = `${b.subtitle || ''} · ${b.points.length} point(s) évalué(s)`;

  // Update available transition dropdown when law study returned transitions
  const trSel = document.getElementById('study-transition-select');
  if (trSel && Array.isArray(b.available_transitions) && b.available_transitions.length) {
    const activeKey = b.active_transition ? `${b.active_transition.src}->${b.active_transition.dst}` : '';
    const existing = Array.from(trSel.options).map(o => o.value).join(',');
    const nextOpts = b.available_transitions.map(([s, d]) => `${s}->${d}`).join(',');
    if (existing !== nextOpts) {
      trSel.innerHTML = `<option value="">Auto (plus forte variation)</option>` +
        b.available_transitions.map(([s, d, score]) =>
          `<option value="${s}->${d}">p(${s} → ${d}) · score=${score.toFixed(3)}</option>`
        ).join('');
      if (activeKey) trSel.value = activeKey;
    }
  }

  const is2D = Boolean(b.secondary_axis);
  const hasLaws = Boolean(b.enable_law_study && b.law_points && b.law_points.length);

  const sensPanel = document.getElementById('study-sensitivity-panel');
  const heatPanel = document.getElementById('study-heatmap-panel');
  if (sensPanel) sensPanel.style.display = is2D ? 'none' : 'flex';
  if (heatPanel) heatPanel.style.display = is2D ? 'flex' : 'none';

  // Ensure selected cell stays in bounds
  if (!APP.studySelectedCell) APP.studySelectedCell = { idx1: 0, idx2: 0 };
  const validPts = b.points.filter(p => !p.error);
  if (validPts.length && !validPts.some(p => p.idx1 === APP.studySelectedCell.idx1 && p.idx2 === APP.studySelectedCell.idx2)) {
    APP.studySelectedCell = { idx1: validPts[0].idx1, idx2: validPts[0].idx2 };
  }

  // ── Left / Right Large Plots ───────────────────────────────────────────
  if (is2D) {
    const heatMetric = document.getElementById('study-heat-metric')?.value || (hasLaws ? 'best_law_family' : 'rmse_rsd_hom');
    drawStudyHeatmap2D(
      document.getElementById('chart-study-heatmap'),
      b,
      heatMetric,
      APP.studySelectedCell,
      (r, c) => {
        APP.studySelectedCell = { idx1: r, idx2: c };
        renderStudyView();
      }
    );
  } else {
    renderStudySensitivityChart(b);
  }

  if (hasLaws) {
    renderStudyLawComparisonChart(b);
    renderStudyLawTable(b);
  } else {
    renderStudyTemporalOverlayChart(b);
    renderStudyPointsTable(b);
  }
}

function renderStudyTemporalOverlayChart(b) {
  const leftTitle = document.getElementById('study-left-title');
  const curves = b.curves || [];
  if (!curves.length) return;
  const chainChoice = document.getElementById('study-curve-chain')?.value || (b.chain_mode === 'inhomogeneous' ? 'inhom' : 'hom');
  if (leftTitle) {
    leftTitle.textContent = `Superposition RSD(t) — DEM vs ${chainChoice === 'inhom' ? 'Inhomogène' : 'Homogène'} (${b.primary_axis.kind})`;
  }

  const xVals = curves[0].times || [];
  const series = [
    { label: 'DEM (réf.)', data: curves[0].rsd_dem, x: xVals, color: '#ffffff', lw: 2.6 },
  ];
  curves.forEach((cv, idx) => {
    const yData = (chainChoice === 'inhom' && cv.rsd_inhom && cv.rsd_inhom.length)
      ? cv.rsd_inhom
      : cv.rsd_hom;
    series.push({
      label: `${b.primary_axis.kind}=${cv.label}`,
      data: yData,
      x: cv.times,
      color: PALETTE[idx % PALETTE.length],
      lw: 1.8,
    });
  });

  drawLineChart(document.getElementById('chart-study-curves'), xVals, series, { showLegend: true });
}

function renderStudySensitivityChart(b) {
  const rightTitle = document.getElementById('study-right-title');
  const metricKey = document.getElementById('study-sens-metric')?.value || 'rmse_rsd_hom';
  const pts = (b.points || []).filter(p => !p.error);
  if (!pts.length) return;

  const isCat = !b.primary_axis.is_numeric;
  const xs = pts.map((p, i) => isCat ? i : p.param1_num);

  const metricExtract = {
    rmse_rsd_hom: { lbl: 'RMSE RSD (Hom)', fn: p => p.rmse_rsd_hom },
    rmse_teneur_hom: { lbl: 'RMSE Teneur (Hom)', fn: p => p.rmse_teneur_hom },
    l1_mean_hom: { lbl: 'Erreur L¹ moyenne (Hom)', fn: p => p.l1_mean_hom },
    frobenius_rel_err: { lbl: 'Erreur relative Frobenius ||P - P_réf||_F / ||P_réf||_F', fn: p => p.frobenius_rel_err ?? 0 },
    nan_rows_hom: { lbl: 'Nombre de lignes NaN (états non visités)', fn: p => p.nan_rows_hom },
    t50_compare: { lbl: 'Temps de mélange t50', fn: p => p.t50_hom ?? null },
    lambda_max: { lbl: 'Valeur propre max |λ|', fn: p => p.lambda_max },
  }[metricKey] || { lbl: 'RMSE RSD', fn: p => p.rmse_rsd_hom };

  if (rightTitle) {
    rightTitle.textContent = `${metricExtract.lbl} vs ${b.primary_axis.display_name || b.primary_axis.kind}`;
  }

  const series = [
    {
      label: metricExtract.lbl,
      data: pts.map(metricExtract.fn),
      x: xs,
      color: '#7dd3bd',
      lw: 2.3,
      points: true,
      pointRadius: 4.5,
    },
  ];
  if (metricKey === 't50_compare') {
    series.push({
      label: 't50 DEM (réf.)',
      data: pts.map(p => p.t50_dem ?? null),
      x: xs,
      color: '#ffffff',
      lw: 2.0,
      dash: [5, 4],
    });
  }
  if (metricKey === 'rmse_rsd_hom' && pts.some(p => p.rmse_rsd_inhom != null)) {
    series.push({
      label: 'RMSE RSD (Inhom)',
      data: pts.map(p => p.rmse_rsd_inhom ?? null),
      x: xs,
      color: '#eec078',
      lw: 2.0,
      points: true,
      pointRadius: 4,
    });
  }
  if (metricKey === 'rmse_teneur_hom' && pts.some(p => p.rmse_teneur_inhom != null)) {
    series.push({
      label: 'RMSE Teneur (Inhom)',
      data: pts.map(p => p.rmse_teneur_inhom ?? null),
      x: xs,
      color: '#eec078',
      lw: 2.0,
      points: true,
      pointRadius: 4,
    });
  }

  drawLineChart(document.getElementById('chart-study-sensitivity'), xs, series, { showLegend: true });
}

function renderStudyLawComparisonChart(b) {
  const leftTitle = document.getElementById('study-left-title');
  const sel = APP.studySelectedCell || { idx1: 0, idx2: 0 };
  const lp = (b.law_points || []).find(item => item.idx1 === sel.idx1 && item.idx2 === sel.idx2) || b.law_points[0];
  if (!lp) return;

  const cmp = lp.comparison;
  if (leftTitle) {
    leftTitle.textContent = `Lois p(${cmp.src} → ${cmp.dst}) · start=${lp.start_index}, NLT=${lp.nlt} · Meilleure : ${cmp.best_family_name}`;
  }

  const series = [
    {
      label: 'Blocs mesurés P^(b)',
      data: cmp.values,
      x: cmp.times,
      color: '#ffffff',
      lw: 0,
      points: true,
      pointRadius: 5.2,
    },
  ];

  (cmp.candidates || []).forEach(c => {
    const isBest = c.family === cmp.best_family;
    series.push({
      label: `${c.family_name} (R²=${c.r2.toFixed(2)})`,
      data: c.dense_predicted,
      x: c.dense_times,
      color: LAW_FAMILY_COLORS[c.family] || '#8faec9',
      lw: isBest ? 2.9 : 1.5,
      dash: isBest ? [] : [5, 4],
    });
  });

  drawLineChart(document.getElementById('chart-study-curves'), cmp.times, series, { showLegend: true });
}

function renderStudyLawTable(b) {
  const wrap = document.getElementById('study-table-wrap');
  if (!wrap) return;
  const sel = APP.studySelectedCell || { idx1: 0, idx2: 0 };
  const lp = (b.law_points || []).find(item => item.idx1 === sel.idx1 && item.idx2 === sel.idx2) || b.law_points[0];
  if (!lp) { wrap.innerHTML = ''; return; }

  const cmp = lp.comparison;
  const rowsHtml = (cmp.candidates || []).map(c => {
    const isBest = c.family === cmp.best_family;
    const oob = c.out_of_block_rmse != null ? c.out_of_block_rmse.toFixed(4) : '—';
    return `<tr class="${isBest ? 'selected-row' : ''}">
      <td><span class="swatch" style="background:${LAW_FAMILY_COLORS[c.family] || '#8faec9'};margin-right:6px;"></span><b>${c.family_name}</b>${isBest ? ' ★' : ''}</td>
      <td style="font-family:var(--mono);font-size:10.5px;">${c.formula_label}</td>
      <td>${c.n_params}</td>
      <td>${c.r2.toFixed(4)}</td>
      <td>${c.rmse.toFixed(5)}</td>
      <td>${c.aicc.toFixed(2)}</td>
      <td>${oob}</td>
    </tr>`;
  }).join('');

  wrap.innerHTML = `<table>
    <thead><tr>
      <th>Famille (start=${lp.start_index}, NLT=${lp.nlt}, p(${cmp.src}→${cmp.dst}))</th>
      <th>Équation ajustée (u ∈ [0,1])</th>
      <th>k</th>
      <th>R²</th>
      <th>RMSE</th>
      <th>AICc ↓</th>
      <th>Out-of-Block RMSE ↓</th>
    </tr></thead>
    <tbody>${rowsHtml}</tbody>
  </table>`;
}

function renderStudyPointsTable(b) {
  const wrap = document.getElementById('study-table-wrap');
  if (!wrap) return;
  const rowsHtml = (b.points || []).map(p => {
    if (p.error) {
      return `<tr><td>${p.param1_label}</td><td colspan="9" class="err">${p.error}</td></tr>`;
    }
    return `<tr>
      <td><b>${p.param1_label}</b>${b.secondary_axis ? ` × ${p.param2_label}` : ''}</td>
      <td>${p.method}</td>
      <td>${p.n_visited}/${p.n_cells}</td>
      <td class="${p.nan_rows_hom > 0 ? 'warn' : 'ok'}">${p.nan_rows_hom}</td>
      <td>${p.n_pairs_used}</td>
      <td>${p.rmse_rsd_hom.toFixed(4)}</td>
      <td>${p.rmse_teneur_hom.toFixed(4)}</td>
      <td>${p.l1_mean_hom.toFixed(4)}</td>
      <td>${p.frobenius_rel_err != null ? p.frobenius_rel_err.toFixed(4) : '—'}</td>
      <td>${p.t50_dem != null ? p.t50_dem.toFixed(1) : '—'} / ${p.t50_hom != null ? p.t50_hom.toFixed(1) : '—'}</td>
    </tr>`;
  }).join('');

  wrap.innerHTML = `<table>
    <thead><tr>
      <th>Paramètre</th><th>Méthode</th><th>Visités</th><th>Lignes NaN</th><th>Paires</th>
      <th>RMSE RSD</th><th>RMSE Teneur</th><th>L¹ moy</th><th>Frobenius rel.</th><th>t50 DEM / MK</th>
    </tr></thead>
    <tbody>${rowsHtml}</tbody>
  </table>`;
}

function exportStudyData(format) {
  const b = APP.studyBundle;
  if (!b) return;
  let content = '', mime = 'application/json', ext = 'json';
  if (format === 'csv') {
    const hdr = ['param1', 'param2', 'method', 'n_cells', 'n_visited', 'nan_rows_hom', 'start_index', 'nlt', 'step', 'dt', 'tau', 'rmse_rsd_hom', 'rmse_teneur_hom', 'l1_mean_hom', 'frobenius_rel_err', 'best_law_family', 'best_law_r2', 'best_law_aicc'];
    const lines = [hdr.join(',')].concat((b.points || []).map(p =>
      hdr.map(k => {
        if (k === 'param1') return JSON.stringify(p.param1_label);
        if (k === 'param2') return JSON.stringify(p.param2_label);
        return p[k] ?? '';
      }).join(',')
    ));
    content = lines.join('\n');
    mime = 'text/csv;charset=utf-8';
    ext = 'csv';
  } else {
    content = JSON.stringify(b, null, 2);
  }
  const blob = new Blob([content], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `study_${b.study_id || 'custom'}_${b.dataset}.${ext}`;
  a.click();
  URL.revokeObjectURL(a.href);
}
