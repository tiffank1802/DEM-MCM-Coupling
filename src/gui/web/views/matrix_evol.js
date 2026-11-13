function getActiveEvolutionData(b) {
  if (!b) return null;
  const spSel = document.getElementById('evol-species')?.value || 'target';
  const useLarge = spSel === 'large' || (spSel === 'target' && b.target_species === 'large');
  const evol = useLarge
    ? (b.matrix_evolution_large || b.matrix_evolution_small)
    : (b.matrix_evolution_small || b.matrix_evolution_large);
  if (evol && evol.components) return evol;
  // Compatibility fallback
  return {
    species: b.target_species,
    n_states: b.mesh?.n_cells || 0,
    threshold: 0.01,
    block_times: b.blocks.map(bl => bl.t_start),
    block_labels: b.blocks.map(bl => `B${bl.block_index}`),
    global_vmax: 1.0,
    components: (b.p_laws || []).map(l => ({
      ...l,
      is_diagonal: l.src === l.dst,
      dense_times: l.block_times,
      dense_predicted: l.predicted,
      max_prob: Math.max(...l.values),
      mean_prob: l.values.reduce((a, c) => a + c, 0) / Math.max(1, l.values.length),
      range_prob: Math.max(...l.values) - Math.min(...l.values),
    })),
  };
}

function toBlockX(arr, xMin, xMax, nBlocks) {
  if (!arr || nBlocks <= 1 || Math.abs(xMax - xMin) < 1e-9) {
    return (arr || []).map((_, idx) => idx);
  }
  return arr.map(t => ((t - xMin) / (xMax - xMin)) * (nBlocks - 1));
}

function filterEvolutionComponents(evol) {
  const scope = document.getElementById('evol-scope')?.value || 'off_diag';
  const srcCell = parseInt(document.getElementById('evol-source-cell')?.value || '0') || 0;
  const threshold = parseFloat(document.getElementById('evol-threshold')?.value || '0.01') || 0.01;
  const sortBy = document.getElementById('evol-sort')?.value || 'max';

  let list = (evol.components || []).filter(c => c.max_prob >= threshold);
  if (scope === 'off_diag') list = list.filter(c => !c.is_diagonal);
  else if (scope === 'diag') list = list.filter(c => c.is_diagonal);
  else if (scope === 'source') list = list.filter(c => c.src === srcCell);

  list = list.slice();
  if (sortBy === 'range') {
    list.sort((a, b) => (b.range_prob - a.range_prob) || (b.max_prob - a.max_prob));
  } else if (sortBy === 'index') {
    list.sort((a, b) => (a.src - b.src) || (a.dst - b.dst));
  } else {
    list.sort((a, b) => (b.max_prob - a.max_prob) || (a.src - b.src) || (a.dst - b.dst));
  }
  return list;
}

function renderMatrixEvolutionView(b) {
  if (!b) return;
  const evol = getActiveEvolutionData(b);
  if (!evol) return;

  const scopeEl = document.getElementById('evol-scope');
  const srcSel = document.getElementById('evol-source-cell');
  if (srcSel && scopeEl) {
    srcSel.style.display = scopeEl.value === 'source' ? 'inline-block' : 'none';
  }

  const filtered = filterEvolutionComponents(evol);
  const mode = document.getElementById('evol-mode')?.value || 'overlay';
  const showFits = document.getElementById('evol-show-fits')?.checked ?? true;
  const commonY = document.getElementById('evol-common-y')?.checked ?? true;
  const xMode = document.getElementById('evol-xaxis')?.value || 'time';
  const countLbl = document.getElementById('evol-count-lbl');
  if (countLbl) {
    countLbl.textContent = `${filtered.length} transitions (${evol.species}) · Ymax=${evol.global_vmax.toFixed(3)}`;
  }

  const bt = evol.block_times || [];
  const xMin = bt.length ? Math.min(...bt) : 0;
  const xMax = bt.length ? Math.max(...bt) : 1;
  const nBlocks = bt.length;
  const getXs = c => xMode === 'block'
    ? (c.values || []).map((_, i) => i)
    : (c.block_times || bt);
  const getDenseXs = c => xMode === 'block'
    ? toBlockX(c.dense_times || c.block_times, xMin, xMax, nBlocks)
    : (c.dense_times || c.block_times || bt);

  const zoomedItem = APP.evolZoomedKey
    ? (evol.components || []).find(c => `${c.src}->${c.dst}` === APP.evolZoomedKey)
    : null;
  const clearZoomBtn = document.getElementById('evol-clear-zoom');
  if (clearZoomBtn) clearZoomBtn.style.display = zoomedItem ? 'inline-block' : 'none';

  const overlayWrap = document.getElementById('evol-overlay-wrap');
  const gridWrap = document.getElementById('evol-grid-wrap');
  const pager = document.getElementById('evol-pager');

  // Show Overlay Mode when selected OR when user zoomed a specific card from the grid
  if (mode === 'overlay' || zoomedItem) {
    if (overlayWrap) overlayWrap.style.display = 'flex';
    if (gridWrap) gridWrap.style.display = 'none';
    if (pager) pager.style.display = 'none';

    const itemsToDraw = zoomedItem
      ? [zoomedItem]
      : (APP.selectedTransitionKey
          ? filtered.filter(c => `${c.src}->${c.dst}` === APP.selectedTransitionKey)
          : filtered.slice(0, 10));
    const displayList = itemsToDraw.length ? itemsToDraw : filtered.slice(0, 10);

    const series = [];
    let localMax = 0.05;
    displayList.forEach((c, idx) => {
      const col = zoomedItem ? '#7dd3bd' : PALETTE[idx % PALETTE.length];
      const xs = getXs(c);
      const dxs = getDenseXs(c);
      localMax = Math.max(localMax, c.max_prob || 0);
      series.push({
        label: `p(${c.src}→${c.dst})`,
        data: c.values,
        x: xs,
        color: col,
        lw: zoomedItem ? 2.8 : 2.2,
        points: true,
        pointRadius: zoomedItem ? 5 : 3.8,
      });
      if (showFits && c.dense_predicted && c.dense_predicted.length) {
        series.push({
          label: zoomedItem ? `${c.law_label} (R²=${c.r2.toFixed(2)})` : '',
          data: c.dense_predicted,
          x: dxs,
          color: zoomedItem ? '#e5989b' : col,
          lw: zoomedItem ? 2.2 : 1.3,
          dash: [5, 4],
        });
      }
      if (showFits && displayList.length === 1 && Array.isArray(c.law_candidates) && c.law_candidates.length) {
        c.law_candidates
          .filter(lc => lc.family === 'exponential' || lc.family === 'linear' || lc.family === c.best_law_family)
          .slice(0, 3)
          .forEach(lc => {
            series.push({
              label: `${lc.family_name} (AICc=${lc.aicc.toFixed(1)})`,
              data: lc.dense_predicted,
              x: xMode === 'block' ? toBlockX(lc.dense_times, xMin, xMax, nBlocks) : lc.dense_times,
              color: LAW_FAMILY_COLORS[lc.family] || '#eec078',
              lw: 1.6,
              dash: [2, 3],
            });
          });
      }
    });

    const mainTitle = document.getElementById('evol-main-title');
    if (mainTitle) {
      if (zoomedItem) {
        mainTitle.textContent = `Zoom p(${zoomedItem.src} → ${zoomedItem.dst}) · ${zoomedItem.law_label} · R²=${zoomedItem.r2.toFixed(3)} · RMSE=${zoomedItem.rmse.toFixed(4)}`;
      } else {
        mainTitle.textContent = `Évolution des ${displayList.length} principales composantes p_{i→j}(t) sur ${nBlocks} blocs (${evol.species})`;
      }
    }

    const baseXs = displayList.length ? getXs(displayList[0]) : [0, 1];
    const yRange = commonY
      ? [0, Math.min(1.08, Math.max(evol.global_vmax, localMax * 1.12))]
      : null;

    drawLineChart(document.getElementById('chart-evol-main'), baseXs, series, {
      yRange,
      showLegend: true,
    });

    // Clickable Law Table on right
    const tbl = document.getElementById('evol-table-wrap');
    if (tbl) {
      tbl.innerHTML = `
        <table>
          <thead>
            <tr><th>Transition</th><th>Max</th><th>Δp</th><th>Meilleure loi</th><th>R²</th><th>RMSE</th><th>Loi polynomiale</th></tr>
          </thead>
          <tbody>
            ${filtered.map(c => {
              const key = `${c.src}->${c.dst}`;
              const isSel = key === APP.selectedTransitionKey || key === APP.evolZoomedKey;
              return `<tr class="clickable-row ${isSel ? 'selected-row' : ''}" data-evol-key="${key}">
                <td class="${c.is_diagonal ? 'warn' : 'ok'}">${c.src} → ${c.dst}</td>
                <td>${c.max_prob.toFixed(3)}</td>
                <td>${c.range_prob.toFixed(3)}</td>
                <td><span class="tpill" style="padding:1px 5px;">${c.best_law_family || ('d' + c.degree)}</span></td>
                <td>${c.r2.toFixed(2)}</td>
                <td>${c.rmse.toFixed(4)}</td>
                <td style="text-align:left;max-width:190px;overflow:hidden;text-overflow:ellipsis;" title="${c.law_label}">${c.law_label}</td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>`;

      tbl.querySelectorAll('tr[data-evol-key]').forEach(row => {
        row.addEventListener('click', () => {
          const k = row.dataset.evolKey;
          APP.selectedTransitionKey = (APP.selectedTransitionKey === k) ? null : k;
          APP.evolZoomedKey = null;
          renderMatrixEvolutionView(APP.bundleA);
        });
      });
    }
    return;
  }

  // Mode B: Common-Scale Grid (16 subplots per page, click any card to zoom)
  if (overlayWrap) overlayWrap.style.display = 'none';
  if (gridWrap) gridWrap.style.display = 'flex';
  if (pager) pager.style.display = 'flex';

  const pageSize = 16;
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  if (APP.evolPage >= totalPages) APP.evolPage = totalPages - 1;
  if (APP.evolPage < 0) APP.evolPage = 0;

  const pageLbl = document.getElementById('evol-page-lbl');
  if (pageLbl) pageLbl.textContent = `${APP.evolPage + 1} / ${totalPages}`;

  const vmaxLbl = document.getElementById('evol-vmax-lbl');
  if (vmaxLbl) {
    vmaxLbl.textContent = commonY
      ? `Échelle Y commune : [0.000 .. ${Math.min(1.08, evol.global_vmax).toFixed(3)}]`
      : 'Échelle Y auto par composante';
  }

  const slice = filtered.slice(APP.evolPage * pageSize, (APP.evolPage + 1) * pageSize);
  const grid = document.getElementById('evol-grid-container');
  if (!grid) return;

  grid.innerHTML = slice.map((c, idx) => `
    <div class="evol-cell-card" data-zoom-key="${c.src}->${c.dst}" title="Cliquer pour agrandir p(${c.src}→${c.dst}) en plein écran">
      <div class="evol-cell-head">
        <b class="${c.is_diagonal ? 'warn' : 'ok'}">p(${c.src} → ${c.dst})</b>
        <span style="font-family:var(--mono);font-size:10px;color:var(--accent-ice);">R²=${c.r2.toFixed(2)} · RMSE=${c.rmse.toFixed(3)}</span>
      </div>
      <div class="canvas-box"><canvas id="evol-sub-${idx}"></canvas></div>
      <div class="evol-cell-law">${c.law_label}</div>
    </div>
  `).join('');

  const sharedY = commonY ? [0, Math.min(1.08, evol.global_vmax)] : null;
  slice.forEach((c, idx) => {
    const cv = document.getElementById(`evol-sub-${idx}`);
    const xs = getXs(c);
    const dxs = getDenseXs(c);
    const s = [
      { label: 'mesuré', data: c.values, x: xs, color: '#7dd3bd', lw: 2.0, points: true, pointRadius: 3.5 },
    ];
    if (showFits && c.dense_predicted && c.dense_predicted.length) {
      s.push({ label: `fit d${c.degree}`, data: c.dense_predicted, x: dxs, color: '#e5989b', lw: 1.5, dash: [4, 3] });
    }
    drawLineChart(cv, xs, s, { yRange: sharedY, compact: true, showLegend: false });
  });

  grid.querySelectorAll('.evol-cell-card').forEach(card => {
    card.addEventListener('click', () => {
      APP.evolZoomedKey = card.dataset.zoomKey;
      renderMatrixEvolutionView(APP.bundleA);
    });
  });
}
