function renderMatricesView(b) {
  if (!b) return;
  const matMode = document.getElementById('mat-view').value;
  const blkIdx = parseInt(document.getElementById('block-select').value) || 0;
  const isLarge = b.target_species === 'large';

  const singleBox = document.getElementById('mat-single-box');
  const gridBox = document.getElementById('mat-blocks-grid');
  const titleEl = document.getElementById('mat-main-title');

  if (matMode === 'blocks_all') {
    singleBox.style.display = 'none';
    gridBox.style.display = 'grid';
    if (titleEl) titleEl.textContent = `Comparaison visuelle des ${b.blocks.length} blocs P^(b) (${b.target_species})`;

    gridBox.innerHTML = b.blocks.map((bl, idx) => `
      <div class="block-card">
        <div class="subtle-bar" style="margin-bottom:3px;">
          <b>Bloc B${bl.block_index} [${bl.t_start}..${bl.t_end}]</b>
          <span class="soft" style="font-family:var(--mono);font-size:10px;">diag=${bl.diag_mean_small.toFixed(2)}</span>
        </div>
        <div class="canvas-box"><canvas id="heat-blk-${idx}"></canvas></div>
      </div>
    `).join('');

    b.blocks.forEach((bl, idx) => {
      const m = isLarge ? bl.p_large : bl.p_small;
      drawMatrixHeatmap(document.getElementById(`heat-blk-${idx}`), m, false, true);
    });
  } else {
    singleBox.style.display = 'block';
    gridBox.style.display = 'none';
    if (matMode === 'hom') {
      if (titleEl) titleEl.textContent = `Matrice homogène P_hom (${b.target_species})`;
      drawMatrixHeatmap(document.getElementById('heat-matrix'), isLarge ? b.p_hom_large : b.p_hom_small, false, false);
    } else if (matMode === 'block' && b.blocks[blkIdx]) {
      const bl = b.blocks[blkIdx];
      if (titleEl) titleEl.textContent = `Bloc B${bl.block_index} [${bl.t_start}..${bl.t_end}] · |λ|_max=${bl.lambda_max_small.toFixed(4)}`;
      drawMatrixHeatmap(document.getElementById('heat-matrix'), isLarge ? bl.p_large : bl.p_small, false, false);
    } else if (matMode === 'diff' && b.blocks.length >= 2) {
      const k0 = Math.min(blkIdx, b.blocks.length - 2);
      const a = isLarge ? b.blocks[k0].p_large : b.blocks[k0].p_small;
      const c = isLarge ? b.blocks[k0 + 1].p_large : b.blocks[k0 + 1].p_small;
      const diff = a.map((row, r) => row.map((v, j) => (c[r][j] ?? 0) - (v ?? 0)));
      if (titleEl) titleEl.textContent = `Différence ΔP_${k0} = P^(B${k0 + 1}) - P^(B${k0})`;
      drawMatrixHeatmap(document.getElementById('heat-matrix'), diff, true, false);
    }
  }

  // Inter-block L1 norm & max delta chart
  const evol = isLarge ? b.matrix_evolution_large : b.matrix_evolution_small;
  const diffs = (evol && evol.delta_norms_l1) || b.block_rel_diffs || [];
  const maxAbs = (evol && evol.delta_max_abs) || [];
  if (diffs.length) {
    const xs = diffs.map((_, k) => k);
    const s = [
      { label: '||P_{k+1} - P_k||₁ / ||P_k||₁', data: diffs, color: '#7dd3bd', lw: 2.3, points: true, pointRadius: 4 },
    ];
    if (maxAbs.length) {
      s.push({ label: 'max |Δp_{ij}|', data: maxAbs, color: '#eec078', lw: 2.0, points: true, pointRadius: 3.5 });
    }
    drawLineChart(document.getElementById('chart-delta-norms'), xs, s);
  }
}
