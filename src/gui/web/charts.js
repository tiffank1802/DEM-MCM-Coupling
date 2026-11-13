function getTooltipEl() {
  return document.getElementById('chart-tooltip');
}

function hideChartTooltip() {
  const tt = getTooltipEl();
  if (tt) tt.style.display = 'none';
}

function showChartTooltip(x, y, html) {
  const tt = getTooltipEl();
  if (!tt) return;
  tt.innerHTML = html;
  tt.style.display = 'block';
  const vw = window.innerWidth, vh = window.innerHeight;
  const tw = Math.min(310, tt.offsetWidth || 200);
  const th = tt.offsetHeight || 70;
  const px = Math.min(vw - tw - 12, x + 14);
  const py = Math.min(vh - th - 12, y + 14);
  tt.style.left = Math.max(8, px) + 'px';
  tt.style.top = Math.max(8, py) + 'px';
}

function attachChartHover(canvas) {
  if (canvas._hoverWired) return;
  canvas._hoverWired = true;
  canvas.addEventListener('mouseleave', () => hideChartTooltip());
  canvas.addEventListener('mousemove', e => {
    const meta = canvas._chartMeta;
    if (!meta) return;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;

    if (meta.kind === 'line') {
      const { pad, pw, ph, xMin, xMax, yMin, yMax, series, W } = meta;
      if (mx < pad.l || mx > W - pad.r || my < pad.t || my > pad.t + ph) {
        hideChartTooltip();
        return;
      }
      const xTarget = xMin + ((mx - pad.l) / pw) * (xMax - xMin);
      const lines = [];
      let nearestX = xTarget;
      let bestDist = Infinity;

      for (const s of series) {
        const xs = s.x || meta.xVals;
        if (!xs || !xs.length || !s.data) continue;
        let bi = -1, bd = Infinity;
        for (let i = 0; i < xs.length; i++) {
          const yv = s.data[i];
          if (yv == null || !isFinite(yv)) continue;
          const d = Math.abs(xs[i] - xTarget);
          if (d < bd) { bd = d; bi = i; }
        }
        if (bi >= 0) {
          if (bd < bestDist) { bestDist = bd; nearestX = xs[bi]; }
          const val = s.data[bi];
          lines.push(`<span style="color:${s.color}">■</span> ${s.label}: <b>${Math.abs(val) >= 100 ? val.toFixed(1) : val.toFixed(4)}</b>`);
        }
      }
      if (lines.length) {
        const header = `<div style="color:#9ec0dc;margin-bottom:2px;">x = ${nearestX.toFixed(1)}</div>`;
        showChartTooltip(e.clientX, e.clientY, header + lines.slice(0, 10).join('<br>'));
      } else {
        hideChartTooltip();
      }
    } else if (meta.kind === 'rect_heat') {
      const { pad, pw, ph, rows, cols, grid, xTimes, cellIds } = meta;
      const c = Math.floor(((mx - pad.l) / pw) * cols);
      const rIdx = rows - 1 - Math.floor(((my - pad.t) / ph) * rows);
      if (c >= 0 && c < cols && rIdx >= 0 && rIdx < rows) {
        const val = grid[rIdx][c];
        showChartTooltip(
          e.clientX,
          e.clientY,
          `Cellule <b>#${cellIds[rIdx] ?? rIdx}</b> · t = <b>${xTimes[c]}</b><br>|Δp| = <b>${val.toFixed(5)}</b>`
        );
      } else {
        hideChartTooltip();
      }
    } else if (meta.kind === 'matrix') {
      const { ox, oy, size, n, matrix } = meta;
      const c = Math.floor((mx - ox) / (size / n));
      const row = Math.floor((my - oy) / (size / n));
      if (row >= 0 && row < n && c >= 0 && c < n) {
        const v = matrix[row][c];
        showChartTooltip(
          e.clientX,
          e.clientY,
          `Transition <b>${row} → ${c}</b><br>P = <b>${v == null ? 'NaN' : v.toFixed(5)}</b>`
        );
      } else {
        hideChartTooltip();
      }
    }
  });
}

function drawLineChart(canvas, xVals, series, opts = {}) {
  if (!canvas || !xVals || !xVals.length) return;
  const host = canvas.parentElement;
  const rect = host ? host.getBoundingClientRect() : canvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const compact = Boolean(opts.compact);
  const blocks = opts.blocks || [];
  const modelRange = opts.modelRange || null;
  const showLegend = opts.showLegend !== false;
  const pad = compact
    ? { l: 38, r: 8, t: showLegend ? 18 : 8, b: 19 }
    : { l: 50, r: 14, t: showLegend ? 24 : 12, b: 26 };
  const pw = Math.max(10, W - pad.l - pad.r);
  const ph = Math.max(10, H - pad.t - pad.b);

  // Determine global X bounds across xVals & per-series x
  let xMin = Infinity, xMax = -Infinity;
  const scanX = arr => {
    for (const xv of arr) {
      if (xv != null && isFinite(xv)) { xMin = Math.min(xMin, xv); xMax = Math.max(xMax, xv); }
    }
  };
  scanX(xVals);
  for (const s of series) if (s.x) scanX(s.x);
  if (!isFinite(xMin)) { xMin = 0; xMax = 1; }
  if (Math.abs(xMax - xMin) < 1e-9) { xMax = xMin + 1; }

  let yMin = Infinity, yMax = -Infinity;
  if (opts.yRange) {
    yMin = opts.yRange[0];
    yMax = opts.yRange[1];
  } else {
    for (const s of series) {
      for (const v of (s.data || [])) {
        if (v != null && isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
      }
    }
    if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
    const ySpan = Math.max(1e-6, yMax - yMin);
    yMin = Math.max(0, yMin - 0.08 * ySpan);
    yMax = yMax + 0.1 * ySpan;
  }
  if (Math.abs(yMax - yMin) < 1e-9) yMax = yMin + 1;

  const tx = x => pad.l + ((x - xMin) / (xMax - xMin)) * pw;
  const ty = y => pad.t + (1 - (y - yMin) / (yMax - yMin)) * ph;

  if (modelRange) {
    const mx0 = Math.max(pad.l, tx(modelRange[0]));
    const mx1 = Math.min(W - pad.r, tx(modelRange[1]));
    if (mx1 > mx0) {
      ctx.fillStyle = 'rgba(125, 211, 189, 0.08)';
      ctx.fillRect(mx0, pad.t, mx1 - mx0, ph);
      ctx.strokeStyle = 'rgba(125, 211, 189, 0.28)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(mx0, pad.t); ctx.lineTo(mx0, pad.t + ph);
      ctx.moveTo(mx1, pad.t); ctx.lineTo(mx1, pad.t + ph);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  blocks.forEach((b, idx) => {
    const x0 = Math.max(pad.l, tx(b.t_start));
    const x1 = Math.min(W - pad.r, tx(b.t_end));
    if (x1 > x0) {
      ctx.fillStyle = idx % 2 === 0 ? 'rgba(207,224,240,0.045)' : 'rgba(125,211,189,0.055)';
      ctx.fillRect(x0, pad.t, x1 - x0, ph);
      if (!compact) {
        ctx.fillStyle = 'rgba(182,192,206,0.58)';
        ctx.font = '9.5px monospace';
        ctx.fillText(`B${b.block_index}`, x0 + 3, pad.t + 11);
      }
    }
  });

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.font = compact ? '9.5px monospace' : '10px monospace';
  ctx.fillStyle = '#8490a2';
  const yTicks = compact ? 3 : 4;
  for (let i = 0; i <= yTicks; i++) {
    const yv = yMin + (i / yTicks) * (yMax - yMin);
    const yp = ty(yv);
    ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(W - pad.r, yp); ctx.stroke();
    ctx.fillText(Math.abs(yv) >= 100 ? yv.toFixed(0) : (Math.abs(yv) >= 10 ? yv.toFixed(1) : yv.toFixed(3)), 4, yp + 3);
  }
  const xTicks = compact ? 3 : 5;
  for (let i = 0; i <= xTicks; i++) {
    const xv = xMin + (i / xTicks) * (xMax - xMin);
    const xp = tx(xv);
    ctx.beginPath(); ctx.moveTo(xp, pad.t); ctx.lineTo(xp, H - pad.b); ctx.stroke();
    const lbl = Math.abs(xMax - xMin) <= 12 && Number.isInteger(Math.round(xv))
      ? String(Math.round(xv * 10) / 10)
      : String(Math.round(xv));
    ctx.fillText(lbl, Math.min(W - pad.r - 18, Math.max(pad.l, xp - 8)), H - 6);
  }

  for (const s of series) {
    const xs = s.x || xVals;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.lw || 1.9;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < xs.length; i++) {
      const y = s.data[i];
      if (y == null || !isFinite(y)) continue;
      const px = tx(xs[i]), py = ty(y);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    if (s.points) {
      ctx.fillStyle = s.color;
      for (let i = 0; i < xs.length; i++) {
        const y = s.data[i];
        if (y == null || !isFinite(y)) continue;
        const px = tx(xs[i]), py = ty(y);
        ctx.beginPath();
        ctx.arc(px, py, s.pointRadius || 3.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  if (showLegend && series.length) {
    let lx = pad.l + 4;
    let ly = 8;
    const maxLx = W - 90;
    for (const s of series) {
      if (!s.label) continue;
      ctx.font = compact ? '9.5px sans-serif' : '10.5px sans-serif';
      const w = ctx.measureText(s.label).width + 22;
      if (lx + w > maxLx) break;
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, ly - 3, 11, 3.5);
      ctx.fillStyle = '#dce4ee';
      ctx.fillText(s.label, lx + 14, ly + 2);
      lx += w + 4;
    }
  }

  canvas._chartMeta = { kind: 'line', pad, pw, ph, xMin, xMax, yMin, yMax, series, xVals, W };
  attachChartHover(canvas);
}

function drawMatrixHeatmap(canvas, matrix, isDiff = false, compact = false) {
  if (!canvas || !matrix || !matrix.length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const n = matrix.length;
  const padSide = compact ? 14 : 34;
  const size = Math.max(40, Math.min(rect.width - padSide, rect.height - padSide));
  const ox = Math.round((rect.width - size) * 0.5);
  const oy = Math.round((rect.height - size) * 0.5);
  const cw = size / n;

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = matrix[r][c];
      if (v == null || !isFinite(v)) {
        ctx.fillStyle = 'rgba(15,18,24,0.65)';
      } else if (isDiff) {
        const k = Math.max(-1, Math.min(1, v * 3));
        if (k >= 0) {
          ctx.fillStyle = `rgb(${Math.round(45 + 190 * k)},${Math.round(52 + 130 * k)},${Math.round(64 + 90 * k)})`;
        } else {
          ctx.fillStyle = `rgb(${Math.round(45 + 165 * (-k))},${Math.round(52 + 50 * (-k))},${Math.round(64 + 60 * (-k))})`;
        }
      } else {
        const k = Math.min(1, Math.max(0, Math.pow(v, 0.55)));
        const R = Math.round(28 + 205 * k);
        const G = Math.round(38 + 185 * Math.sin(k * Math.PI * 0.78));
        const B = Math.round(56 + 150 * (1 - k * 0.45));
        ctx.fillStyle = `rgb(${R},${G},${B})`;
      }
      ctx.fillRect(ox + c * cw, oy + r * cw, Math.max(1, cw - 0.5), Math.max(1, cw - 0.5));
    }
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.strokeRect(ox, oy, size, size);

  if (!compact && n <= 24) {
    ctx.fillStyle = '#8490a2';
    ctx.font = '9px monospace';
    const step = n > 14 ? 2 : 1;
    for (let i = 0; i < n; i += step) {
      ctx.fillText(String(i), ox + i * cw + cw * 0.3, oy + size + 10);
      ctx.fillText(String(i), Math.max(2, ox - 16), oy + i * cw + cw * 0.65);
    }
  }

  canvas._chartMeta = { kind: 'matrix', ox, oy, size, n, matrix };
  attachChartHover(canvas);
}

function drawRectHeatmap(canvas, grid, xTimes, cellIds) {
  if (!canvas || !grid || !grid.length || !grid[0].length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 40, r: 12, t: 10, b: 22 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const rows = grid.length, cols = grid[0].length;
  let vmax = 1e-6;
  for (const r of grid) for (const v of r) if (v > vmax) vmax = v;
  const cw = pw / cols, ch = ph / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const k = Math.min(1, grid[r][c] / vmax);
      const R = Math.round(32 + 200 * Math.pow(k, 0.7));
      const G = Math.round(45 + 145 * Math.sin(k * Math.PI));
      const B = Math.round(62 + 40 * (1 - k));
      ctx.fillStyle = `rgb(${R},${G},${B})`;
      ctx.fillRect(pad.l + c * cw, pad.t + (rows - 1 - r) * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  ctx.fillStyle = '#8490a2';
  ctx.font = '9.5px monospace';
  ctx.fillText(`C${cellIds[rows - 1] ?? 0}`, 4, pad.t + 10);
  ctx.fillText(`C${cellIds[0] ?? 0}`, 4, H - pad.b - 2);
  ctx.fillText(String(xTimes[0]), pad.l, H - 6);
  ctx.fillText(String(xTimes[cols - 1]), W - pad.r - 32, H - 6);

  canvas._chartMeta = { kind: 'rect_heat', pad, pw, ph, rows, cols, grid, xTimes, cellIds };
  attachChartHover(canvas);
}

function drawCellErrorBars(canvas, cellIds, cellErrors) {
  if (!canvas || !cellErrors || !cellErrors.length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 46, r: 10, t: 16, b: 22 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const n = cellIds.length;
  if (!n) return;
  let ymax = 1e-5;
  cellIds.forEach(cid => { ymax = Math.max(ymax, cellErrors[cid] || 0); });
  ymax *= 1.15;
  const gw = pw / n, bw = Math.max(3, gw * 0.72);

  ctx.font = '9.5px monospace';
  ctx.fillStyle = '#8490a2';
  for (let i = 0; i <= 3; i++) {
    const yv = (i / 3) * ymax;
    const yp = pad.t + (1 - i / 3) * ph;
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(W - pad.r, yp); ctx.stroke();
    ctx.fillText(yv.toFixed(3), 4, yp + 3);
  }
  cellIds.forEach((cid, idx) => {
    const val = cellErrors[cid] || 0;
    const bh = (val / ymax) * ph;
    const x = pad.l + idx * gw + (gw - bw) * 0.5;
    ctx.fillStyle = 'rgba(125, 211, 189, 0.78)';
    ctx.fillRect(x, pad.t + ph - bh, bw, bh);
    if (n <= 24 || idx % 2 === 0) {
      ctx.fillStyle = '#98a4b5';
      ctx.fillText(`C${cid}`, x, H - 6);
    }
  });
}

function drawBarBenchmark(canvas, rows, metricKey = 'rmse_teneur') {
  if (!canvas || !rows || !rows.length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 46, r: 12, t: 22, b: 30 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  const pairFor = r => {
    if (metricKey === 'mae_teneur') return [r.hom_mae_teneur, r.inhom_mae_teneur, 'MAE(Teneur)'];
    if (metricKey === 'l1_mean') return [r.hom_l1_mean, r.inhom_l1_mean, 'L1 moy'];
    if (metricKey === 'rms_disc') return [r.hom_rms_disc, r.inhom_rms_disc, 'RMS disc.'];
    return [r.hom_rmse_teneur, r.inhom_rmse_teneur, 'RMSE(Teneur)'];
  };

  let ymax = 1e-4;
  let metricLbl = 'RMSE(Teneur)';
  rows.forEach(r => {
    const [a, b, lbl] = pairFor(r);
    metricLbl = lbl;
    ymax = Math.max(ymax, a, b);
  });
  ymax *= 1.15;
  const gw = pw / rows.length, bw = Math.max(6, gw * 0.34);

  ctx.font = '10px monospace';
  ctx.fillStyle = '#8490a2';
  for (let i = 0; i <= 3; i++) {
    const yv = (i / 3) * ymax;
    const yp = pad.t + (1 - i / 3) * ph;
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(W - pad.r, yp); ctx.stroke();
    ctx.fillText(yv.toFixed(3), 4, yp + 3);
  }
  rows.forEach((r, idx) => {
    const [vH, vI] = pairFor(r);
    const x0 = pad.l + idx * gw + gw * 0.14;
    const hH = (vH / ymax) * ph;
    const hI = (vI / ymax) * ph;
    ctx.fillStyle = 'rgba(158, 192, 220, 0.78)';
    ctx.fillRect(x0, pad.t + ph - hH, bw, hH);
    ctx.fillStyle = 'rgba(125, 211, 189, 0.85)';
    ctx.fillRect(x0 + bw + 3, pad.t + ph - hI, bw, hI);
    ctx.fillStyle = '#dce4ee';
    ctx.fillText(r.method.slice(0, 9), x0, H - 10);
  });
  ctx.fillStyle = 'rgba(158, 192, 220, 0.85)'; ctx.fillRect(pad.l, 6, 11, 4);
  ctx.fillStyle = '#dce4ee'; ctx.font = '10.5px sans-serif'; ctx.fillText(`${metricLbl} Homogène`, pad.l + 15, 11);
  ctx.fillStyle = 'rgba(125, 211, 189, 0.90)'; ctx.fillRect(pad.l + 165, 6, 11, 4);
  ctx.fillStyle = '#dce4ee'; ctx.fillText(`${metricLbl} Inhomogène`, pad.l + 180, 11);
}

const LAW_FAMILY_COLORS = {
  constant: '#98a4b5',
  linear: '#8faec9',
  quadratic: '#7dd3bd',
  cubic: '#b8b8e6',
  exponential: '#eec078',
  logarithmic: '#e5989b',
  power: '#c9a6e0',
};

function drawStudyHeatmap2D(canvas, bundle, metricKey, selectedCell, onCellClick) {
  if (!canvas || !bundle || !bundle.points || !bundle.points.length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const l1 = bundle.primary_axis?.labels || [];
  const l2 = bundle.secondary_axis?.labels || ['-'];
  const nRows = l1.length;
  const nCols = l2.length;
  if (!nRows || !nCols) return;

  const pad = { l: 74, r: 16, t: 28, b: 34 };
  const pw = Math.max(40, W - pad.l - pad.r);
  const ph = Math.max(40, H - pad.t - pad.b);
  const cw = pw / nCols;
  const ch = ph / nRows;

  const ptMap = new Map();
  let vmin = Infinity, vmax = -Infinity;
  const getVal = p => {
    if (metricKey === 'best_law_r2') return p.best_law_r2 ?? null;
    if (metricKey === 'best_law_aicc') return p.best_law_aicc ?? null;
    if (metricKey === 'best_law_oob_rmse') return p.best_law_oob_rmse ?? p.best_law_rmse ?? null;
    if (metricKey === 'rmse_teneur_inhom') return p.rmse_teneur_inhom ?? p.rmse_teneur_hom;
    if (metricKey === 'rmse_rsd_inhom') return p.rmse_rsd_inhom ?? p.rmse_rsd_hom;
    if (metricKey === 'nan_rows_hom') return p.nan_rows_hom;
    return p.rmse_rsd_hom;
  };
  for (const p of bundle.points) {
    ptMap.set(`${p.idx1}:${p.idx2}`, p);
    const v = getVal(p);
    if (v != null && isFinite(v)) {
      vmin = Math.min(vmin, v);
      vmax = Math.max(vmax, v);
    }
  }
  if (!isFinite(vmin)) { vmin = 0; vmax = 1; }
  const vspan = Math.max(1e-9, vmax - vmin);

  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < nCols; c++) {
      const p = ptMap.get(`${r}:${c}`);
      const x = pad.l + c * cw;
      const y = pad.t + r * ch;
      if (!p || p.error) {
        ctx.fillStyle = 'rgba(229, 152, 155, 0.22)';
        ctx.fillRect(x + 1, y + 1, Math.max(2, cw - 2), Math.max(2, ch - 2));
        ctx.fillStyle = '#e5989b';
        ctx.font = '10px monospace';
        ctx.fillText('N/A', x + cw * 0.35, y + ch * 0.54);
        continue;
      }

      let txt = '';
      if (metricKey === 'best_law_family') {
        const fam = p.best_law_family || 'constant';
        ctx.fillStyle = hexToRgba(LAW_FAMILY_COLORS[fam] || '#8faec9', 0.48);
        txt = `${p.best_law_name || fam}`;
      } else {
        const val = getVal(p);
        const k = val == null ? 0.5 : (val - vmin) / vspan;
        const good = (metricKey === 'best_law_r2') ? k : (1 - k);
        const R = Math.round(180 * (1 - good) + 52 * good);
        const G = Math.round(70 * (1 - good) + 175 * good);
        const B = Math.round(75 * (1 - good) + 150 * good);
        ctx.fillStyle = `rgba(${R},${G},${B},0.72)`;
        txt = val == null ? '—' : (Math.abs(val) >= 100 ? val.toFixed(1) : val.toFixed(3));
      }
      ctx.fillRect(x + 1.5, y + 1.5, Math.max(2, cw - 3), Math.max(2, ch - 3));

      const isSel = selectedCell && selectedCell.idx1 === r && selectedCell.idx2 === c;
      ctx.strokeStyle = isSel ? '#ffffff' : 'rgba(255,255,255,0.16)';
      ctx.lineWidth = isSel ? 2.4 : 1;
      ctx.strokeRect(x + 1.5, y + 1.5, Math.max(2, cw - 3), Math.max(2, ch - 3));

      ctx.fillStyle = '#f2f6fb';
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(txt.slice(0, 12), x + cw * 0.5, y + ch * 0.48);
      if (p.best_law_r2 != null) {
        ctx.fillStyle = 'rgba(220,228,238,0.85)';
        ctx.font = '10px monospace';
        ctx.fillText(`R²=${p.best_law_r2.toFixed(2)}`, x + cw * 0.5, y + ch * 0.68);
      }
      ctx.textAlign = 'left';
    }
  }

  // Axis headers
  ctx.fillStyle = '#98a4b5';
  ctx.font = '10.5px monospace';
  ctx.textAlign = 'center';
  for (let c = 0; c < nCols; c++) {
    ctx.fillText(`${bundle.secondary_axis?.kind || 'A2'}=${l2[c]}`, pad.l + (c + 0.5) * cw, pad.t - 8);
  }
  ctx.textAlign = 'right';
  for (let r = 0; r < nRows; r++) {
    ctx.fillText(`${bundle.primary_axis?.kind || 'A1'}=${l1[r]}`, pad.l - 6, pad.t + (r + 0.55) * ch);
  }
  ctx.textAlign = 'left';

  if (!canvas._studyClickWired) {
    canvas._studyClickWired = true;
    canvas.addEventListener('click', e => {
      const m = canvas._studyHeatMeta;
      if (!m || !m.onCellClick) return;
      const rr = canvas.getBoundingClientRect();
      const mx = e.clientX - rr.left, my = e.clientY - rr.top;
      const c = Math.floor((mx - m.pad.l) / m.cw);
      const r = Math.floor((my - m.pad.t) / m.ch);
      if (r >= 0 && r < m.nRows && c >= 0 && c < m.nCols) {
        m.onCellClick(r, c);
      }
    });
  }
  canvas._studyHeatMeta = { pad, cw, ch, nRows, nCols, onCellClick };
}

