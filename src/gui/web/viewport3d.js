function teneurColor(val, alpha, isErr = false) {
  const t = Math.max(0, Math.min(1, val));
  if (isErr) {
    const r = Math.round(90 + 145 * Math.pow(t, 0.72));
    const g = Math.round(200 * (1 - t * 0.55));
    const b = Math.round(170 * (1 - t * 0.75));
    return `rgba(${r},${g},${b},${alpha})`;
  }
  let r, g, b;
  if (t < 0.5) {
    const k = t * 2;
    r = Math.round(92 + (222 - 92) * k);
    g = Math.round(126 + (228 - 126) * k);
    b = Math.round(156 + (236 - 156) * k);
  } else {
    const k = (t - 0.5) * 2;
    r = Math.round(222 + (218 - 222) * k);
    g = Math.round(228 + (132 - 228) * k);
    b = Math.round(236 + (132 - 236) * k);
  }
  return `rgba(${r},${g},${b},${alpha})`;
}

function fitCameraToMesh(mesh) {
  if (!mesh || !mesh.domain_bounds) return;
  const b = mesh.domain_bounds;
  const diag = Math.max(0.05, Math.hypot(b[1] - b[0], b[3] - b[2], b[5] - b[4]));
  APP.cam.target = [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])];
  APP.cam.distance = diag * 1.38;
}

function setCamPreset(p) {
  if (p === 'iso') { APP.cam.yaw = 0.68; APP.cam.pitch = 0.44; }
  else if (p === 'xy') { APP.cam.yaw = 0.0; APP.cam.pitch = 0.0; }
  else if (p === 'xz') { APP.cam.yaw = 0.0; APP.cam.pitch = 1.54; }
  else if (p === 'yz') { APP.cam.yaw = Math.PI / 2; APP.cam.pitch = 0.0; }
  else if (p === 'fit' && APP.bundleA) { fitCameraToMesh(APP.bundleA.mesh); }
}

function camBasis(cam) {
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const fwd = [-cp * sy, -sp, -cp * cy];
  const rx = -fwd[2], rz = fwd[0];
  const rl = Math.hypot(rx, rz) || 1e-12;
  const right = [rx / rl, 0, rz / rl];
  const up = [
    right[1] * fwd[2] - right[2] * fwd[1],
    right[2] * fwd[0] - right[0] * fwd[2],
    right[0] * fwd[1] - right[1] * fwd[0],
  ];
  const eye = [
    cam.target[0] - fwd[0] * cam.distance,
    cam.target[1] - fwd[1] * cam.distance,
    cam.target[2] - fwd[2] * cam.distance,
  ];
  return { right, up, fwd, eye };
}

function projectPt(pt, basis, w, h, fov) {
  const rx = pt[0] - basis.eye[0], ry = pt[1] - basis.eye[1], rz = pt[2] - basis.eye[2];
  const cx = rx * basis.right[0] + ry * basis.right[1] + rz * basis.right[2];
  const cy = rx * basis.up[0] + ry * basis.up[1] + rz * basis.up[2];
  const depth = rx * basis.fwd[0] + ry * basis.fwd[1] + rz * basis.fwd[2];
  if (depth <= 1e-4) return null;
  const focal = (0.5 * h) / Math.tan((fov * Math.PI / 180) * 0.5);
  const scale = focal / depth;
  return { x: 0.5 * w + cx * scale, y: 0.5 * h - cy * scale, depth, scale };
}

function getCellMetricValue(bundle, cid, frame, colorMode) {
  const evalIdx = nearestEvalRow(bundle, frame.timestep);
  const cDem = (bundle.dem_conc_traj[evalIdx] && bundle.dem_conc_traj[evalIdx][cid]) ?? (frame.cell_conc_dem[cid] ?? 0.5);
  const cHom = (bundle.hom_conc_traj[evalIdx] && bundle.hom_conc_traj[evalIdx][cid]) ?? 0.5;
  const cInh = (bundle.inhom_conc_traj[evalIdx] && bundle.inhom_conc_traj[evalIdx][cid]) ?? 0.5;
  if (colorMode === 'dem_teneur') return { v: cDem, err: false };
  if (colorMode === 'hom_teneur') return { v: cHom, err: false };
  if (colorMode === 'inhom_teneur') return { v: cInh, err: false };
  if (colorMode === 'hom_err') return { v: Math.min(1, Math.abs(cHom - cDem) * 2.5), err: true };
  if (colorMode === 'inhom_err') return { v: Math.min(1, Math.abs(cInh - cDem) * 2.5), err: true };
  if (colorMode === 'cell_rmse') {
    const cr = (bundle.inhom_metrics.cell_rmse_per_cell && bundle.inhom_metrics.cell_rmse_per_cell[cid]) || 0;
    return { v: Math.min(1, cr * 8), err: true };
  }
  if (colorMode === 'population') {
    const maxPop = Math.max(1, ...frame.cell_pop);
    return { v: (frame.cell_pop[cid] || 0) / maxPop, err: false };
  }
  return { v: cid, err: false };
}

function renderViewport3D(canvas, bundle, recordHits = false) {
  if (!canvas || !bundle) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const basis = camBasis(APP.cam);
  const proj = p => projectPt(p, basis, W, H, APP.cam.fov);
  const frame = bundle.frames[Math.min(APP.frameIdx, bundle.frames.length - 1)] || bundle.frames[0];
  const colorMode = document.getElementById('color-mode').value;
  const showWire = document.getElementById('show-wire').checked;
  const showFaces = document.getElementById('show-faces').checked;
  const showSmall = document.getElementById('show-small').checked;
  const showLarge = document.getElementById('show-large').checked;
  const showVel = document.getElementById('show-vel').checked;
  const showEdges = document.getElementById('show-edges').checked;
  const showDrum = document.getElementById('show-drum').checked;

  if (showDrum) {
    const dr = bundle.mesh.drum;
    const N = 40;
    ctx.strokeStyle = 'rgba(207, 224, 240, 0.22)';
    ctx.lineWidth = 1;
    [dr.z_min, dr.z_max].forEach(z => {
      ctx.beginPath();
      for (let i = 0; i <= N; i++) {
        const a = (i / N) * Math.PI * 2;
        const p = proj([dr.radius * Math.cos(a), dr.radius * Math.sin(a), z]);
        if (!p) continue;
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    });
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      const p1 = proj([dr.radius * Math.cos(a), dr.radius * Math.sin(a), dr.z_min]);
      const p2 = proj([dr.radius * Math.cos(a), dr.radius * Math.sin(a), dr.z_max]);
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }
  }

  const items = [];
  if (recordHits) APP.projectedCells = [];

  for (const cell of bundle.mesh.cells) {
    const isSel = APP.selectedCell === cell.id;
    const pop = frame.cell_pop[cell.id] || 0;
    const metric = getCellMetricValue(bundle, cell.id, frame, colorMode);
    const baseHex = PALETTE[(cell.zone * 5 + cell.id) % PALETTE.length];
    const faceFill = colorMode === 'cell_id'
      ? hexToRgba(baseHex, isSel ? 0.38 : 0.11)
      : teneurColor(metric.v, isSel ? 0.44 : (pop > 0 ? 0.18 : 0.04), metric.err);

    const pc = proj(cell.centroid);
    if (recordHits && pc) {
      APP.projectedCells.push({ id: cell.id, x: pc.x, y: pc.y, pop, cell });
    }

    if (showFaces) {
      for (const poly of cell.faces) {
        const pts = [];
        let dSum = 0, ok = true;
        for (const v of poly) {
          const pv = proj(v);
          if (!pv) { ok = false; break; }
          pts.push(pv);
          dSum += pv.depth;
        }
        if (ok && pts.length >= 3) {
          items.push({ type: 'poly', depth: dSum / pts.length, pts, fill: faceFill, stroke: isSel ? '#ffffff' : null });
        }
      }
    }

    if (showWire) {
      const strokeCol = isSel
        ? '#ffffff'
        : (colorMode === 'cell_id' ? hexToRgba(baseHex, 0.58) : 'rgba(207, 224, 240, 0.30)');
      for (const [a, b] of cell.edges) {
        const pa = proj(a), pb = proj(b);
        if (pa && pb) {
          items.push({ type: 'line', depth: 0.5 * (pa.depth + pb.depth) - 1e-4, pa, pb, col: strokeCol, lw: isSel ? 1.8 : 0.8 });
        }
      }
    }
  }

  for (const sp of (bundle.mesh.split_planes || [])) {
    const pts = sp.quad.map(proj).filter(Boolean);
    if (pts.length === 4) {
      const d = pts.reduce((a, p) => a + p.depth, 0) / 4;
      items.push({ type: 'poly', depth: d, pts, fill: 'rgba(125,211,189,0.14)', stroke: 'rgba(125,211,189,0.75)' });
    }
  }

  if (showEdges && bundle.top_edges_small) {
    const cenById = new Map(bundle.mesh.cells.map(c => [c.id, c.centroid]));
    for (const [si, dj, prob] of bundle.top_edges_small) {
      const cs = cenById.get(si), cd = cenById.get(dj);
      if (!cs || !cd) continue;
      const pa = proj(cs), pb = proj(cd);
      if (pa && pb) {
        items.push({
          type: 'arrow',
          depth: 0.5 * (pa.depth + pb.depth) - 2e-4,
          pa, pb,
          col: `rgba(158, 192, 220, ${Math.min(0.85, 0.22 + prob * 1.3)})`,
          lw: 0.9 + prob * 4.0,
        });
      }
    }
  }

  if (frame && (showSmall || showLarge)) {
    const diams = frame.diameters || [];
    for (let i = 0; i < frame.coords.length; i++) {
      const isS = frame.is_small[i];
      if ((isS && !showSmall) || (!isS && !showLarge)) continue;
      const p = proj(frame.coords[i]);
      if (!p) continue;
      const d = diams[i] || (isS ? 0.004 : 0.008);
      const rad = Math.max(isS ? 2.1 : 4.2, Math.min(isS ? 5.0 : 10.0, p.scale * d * 0.56));
      items.push({ type: 'sphere', depth: p.depth - 3e-4, x: p.x, y: p.y, r: rad, isSmall: isS });
      if (showVel && frame.velocities[i]) {
        const v = frame.velocities[i];
        const tip = proj([
          frame.coords[i][0] + v[0] * 0.024,
          frame.coords[i][1] + v[1] * 0.024,
          frame.coords[i][2] + v[2] * 0.024,
        ]);
        if (tip) {
          items.push({ type: 'line', depth: p.depth - 3.5e-4, pa: p, pb: tip, col: 'rgba(207,224,240,0.65)', lw: 1.0 });
        }
      }
    }
  }

  items.sort((a, b) => b.depth - a.depth);
  for (const it of items) {
    if (it.type === 'poly') {
      ctx.beginPath();
      ctx.moveTo(it.pts[0].x, it.pts[0].y);
      for (let k = 1; k < it.pts.length; k++) ctx.lineTo(it.pts[k].x, it.pts[k].y);
      ctx.closePath();
      ctx.fillStyle = it.fill;
      ctx.fill();
      if (it.stroke) { ctx.strokeStyle = it.stroke; ctx.lineWidth = 1.3; ctx.stroke(); }
    } else if (it.type === 'line') {
      ctx.strokeStyle = it.col;
      ctx.lineWidth = it.lw;
      ctx.beginPath();
      ctx.moveTo(it.pa.x, it.pa.y);
      ctx.lineTo(it.pb.x, it.pb.y);
      ctx.stroke();
    } else if (it.type === 'arrow') {
      ctx.strokeStyle = it.col;
      ctx.lineWidth = it.lw;
      ctx.beginPath();
      ctx.moveTo(it.pa.x, it.pa.y);
      ctx.lineTo(it.pb.x, it.pb.y);
      ctx.stroke();
      const ang = Math.atan2(it.pb.y - it.pa.y, it.pb.x - it.pa.x);
      const hl = 5 + it.lw;
      ctx.beginPath();
      ctx.moveTo(it.pb.x, it.pb.y);
      ctx.lineTo(it.pb.x - hl * Math.cos(ang - 0.42), it.pb.y - hl * Math.sin(ang - 0.42));
      ctx.lineTo(it.pb.x - hl * Math.cos(ang + 0.42), it.pb.y - hl * Math.sin(ang + 0.42));
      ctx.closePath();
      ctx.fillStyle = it.col;
      ctx.fill();
    } else if (it.type === 'sphere') {
      const g = ctx.createRadialGradient(it.x - it.r * 0.3, it.y - it.r * 0.3, it.r * 0.12, it.x, it.y, it.r);
      if (it.isSmall) {
        g.addColorStop(0, '#dcfaf5');
        g.addColorStop(0.45, '#7dd3c8');
        g.addColorStop(1, '#2b6a64');
      } else {
        g.addColorStop(0, '#fff3db');
        g.addColorStop(0.45, '#eec078');
        g.addColorStop(1, '#7a5420');
      }
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(it.x, it.y, it.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(10,12,16,0.55)';
      ctx.lineWidth = 0.65;
      ctx.stroke();
    }
  }
}

function wire3DEvents(canvas, onRender3D, onCellSelected) {
  if (!canvas) return;
  canvas.addEventListener('mousedown', e => {
    APP.drag.active = true;
    APP.drag.mode = (e.button === 2 || e.shiftKey) ? 'pan' : 'orbit';
    APP.drag.x = e.clientX;
    APP.drag.y = e.clientY;
  });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  window.addEventListener('mousemove', e => {
    if (!APP.drag.active) return;
    const dx = e.clientX - APP.drag.x, dy = e.clientY - APP.drag.y;
    APP.drag.x = e.clientX;
    APP.drag.y = e.clientY;
    if (APP.drag.mode === 'orbit') {
      APP.cam.yaw += dx * 0.008;
      APP.cam.pitch = Math.max(-1.54, Math.min(1.54, APP.cam.pitch + dy * 0.008));
    } else {
      const b = camBasis(APP.cam);
      const scale = APP.cam.distance * 0.0018;
      APP.cam.target[0] -= (b.right[0] * dx - b.up[0] * dy) * scale;
      APP.cam.target[1] -= (b.right[1] * dx - b.up[1] * dy) * scale;
      APP.cam.target[2] -= (b.right[2] * dx - b.up[2] * dy) * scale;
    }
    onRender3D();
  });
  window.addEventListener('mouseup', () => { APP.drag.active = false; });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    APP.cam.distance = Math.max(0.03, Math.min(20, APP.cam.distance * (e.deltaY > 0 ? 1.1 : 0.9)));
    onRender3D();
  }, { passive: false });
  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 26 * 26;
    for (const c of APP.projectedCells) {
      const d2 = (c.x - x) ** 2 + (c.y - y) ** 2;
      if (d2 < bd) { bd = d2; best = c; }
    }
    APP.selectedCell = best ? best.id : null;
    if (onCellSelected) onCellSelected(best);
    onRender3D();
  });
}
