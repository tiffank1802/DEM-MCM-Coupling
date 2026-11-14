// 3-D viewport rendered with WebGL for fluidity.
//
// The scene (cell faces, wireframe edges, split planes, drum, particles,
// velocity arrows) is uploaded to the GPU each frame and drawn with
// hardware acceleration: a depth buffer handles occlusion, shaders handle
// projection/color, and points/triangles/lines are drawn in single
// batched calls. All shared state is read from the central reactive store
// (APP.*), so any variable updated anywhere propagates to the 3-D view.

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

// ---------------------------------------------------------------------------
// Minimal column-major mat4 helpers (WebGL)
// ---------------------------------------------------------------------------
function mat4Perspective(fovDeg, aspect, near, far) {
  const f = 1 / Math.tan((fovDeg * Math.PI / 180) * 0.5);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function mat4LookAt(eye, center, up) {
  let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
  let zl = Math.hypot(zx, zy, zz) || 1e-12;
  zx /= zl; zy /= zl; zz /= zl;
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  let xl = Math.hypot(xx, xy, xz) || 1e-12;
  xx /= xl; xy /= xl; xz /= xl;
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1,
  ]);
}

function mat4Multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}

function parseColor(c) {
  if (typeof c === 'number') return [((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255, 1];
  if (c && c[0] === '#') {
    let h = c.slice(1);
    if (h.length === 3) h = h.split('').map(x => x + x).join('');
    const n = parseInt(h, 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1];
  }
  const m = c && c.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const p = m[1].split(',').map(s => parseFloat(s));
    return [p[0] / 255, p[1] / 255, p[2] / 255, p.length > 3 ? p[3] : 1];
  }
  return [1, 1, 1, 1];
}

// ---------------------------------------------------------------------------
// WebGL program & per-canvas renderer state
// ---------------------------------------------------------------------------
const VERT_SRC = `
attribute vec3 aPos;
attribute vec4 aColor;
attribute float aSize;
uniform mat4 uVP;
varying vec4 vColor;
void main() {
  gl_Position = uVP * vec4(aPos, 1.0);
  gl_PointSize = aSize;
  vColor = aColor;
}`;

const FRAG_SRC = `
precision mediump float;
varying vec4 vColor;
void main() {
  // Round points; other primitives ignore gl_PointCoord.
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vColor;
}`;

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

const GL_RENDERERS = new WeakMap();

function getGL(canvas) {
  let r = GL_RENDERERS.get(canvas);
  if (r) return r;
  const gl = canvas.getContext('webgl', { antialias: true, alpha: false, depth: true, premultipliedAlpha: false })
    || canvas.getContext('experimental-webgl', { antialias: true, alpha: false, depth: true, premultipliedAlpha: false });
  if (!gl) { GL_RENDERERS.set(canvas, null); return null; }
  const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  const locs = {
    prog,
    aPos: gl.getAttribLocation(prog, 'aPos'),
    aColor: gl.getAttribLocation(prog, 'aColor'),
    aSize: gl.getAttribLocation(prog, 'aSize'),
    uVP: gl.getUniformLocation(prog, 'uVP'),
    tri: gl.createBuffer(),
    line: gl.createBuffer(),
    point: gl.createBuffer(),
  };
  r = { gl, locs };
  GL_RENDERERS.set(canvas, r);
  return r;
}

// Build an interleaved [x,y,z, r,g,b,a, size] buffer from a list of vertices.
function buildBuffer(out, verts) {
  const arr = new Float32Array(verts.length * 8);
  for (let i = 0, j = 0; i < verts.length; i++, j += 8) {
    const v = verts[i];
    arr[j] = v[0]; arr[j + 1] = v[1]; arr[j + 2] = v[2];
    arr[j + 3] = v[3]; arr[j + 4] = v[4]; arr[j + 5] = v[5]; arr[j + 6] = v[6];
    arr[j + 7] = v[7];
  }
  out.push(arr);
}

// Draw a cached buffer: sets up attributes and issues a single draw call.
function drawBuffer(gl, locs, bufferData, mode) {
  if (!bufferData || bufferData.length === 0) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, mode === gl.POINTS ? locs.point : (mode === gl.LINES ? locs.line : locs.tri));
  gl.bufferData(gl.ARRAY_BUFFER, bufferData, gl.DYNAMIC_DRAW);
  const stride = 8 * 4;
  gl.enableVertexAttribArray(locs.aPos);
  gl.vertexAttribPointer(locs.aPos, 3, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(locs.aColor);
  gl.vertexAttribPointer(locs.aColor, 4, gl.FLOAT, false, stride, 3 * 4);
  if (locs.aSize >= 0) {
    gl.enableVertexAttribArray(locs.aSize);
    gl.vertexAttribPointer(locs.aSize, 1, gl.FLOAT, false, stride, 7 * 4);
  }
  gl.drawArrays(mode, 0, bufferData.length / 8);
}

function renderViewport3D(canvas, bundle, recordHits = false) {
  if (!canvas || !bundle) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  const pw = Math.round(rect.width * dpr);
  const ph = Math.round(rect.height * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  const r = getGL(canvas);
  if (!r) return; // No WebGL available; nothing to draw.
  const { gl, locs } = r;

  const W = rect.width, H = rect.height;
  gl.viewport(0, 0, pw, ph);
  gl.clearColor(0.055, 0.075, 0.11, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // View-projection matrix from the centralized camera state.
  const basis = camBasis(APP.cam);
  const vp = mat4Multiply(
    mat4Perspective(APP.cam.fov, pw / ph, 1e-4, 1e3),
    mat4LookAt(basis.eye, APP.cam.target, basis.up)
  );
  gl.useProgram(locs.prog);
  gl.uniformMatrix4fv(locs.uVP, false, vp);

  const frame = bundle.frames[Math.min(APP.frameIdx, bundle.frames.length - 1)] || bundle.frames[0];
  const colorMode = APP.colorMode;
  const showWire = APP.showWire;
  const showFaces = APP.showFaces;
  const showSmall = APP.showSmall;
  const showLarge = APP.showLarge;
  const showVel = APP.showVel;
  const showEdges = APP.showEdges;
  const showDrum = APP.showDrum;

  if (recordHits) APP.projectedCells = [];

  const tri = [];
  const line = [];
  const point = [];
  const col = (c, size = 0) => { const p = parseColor(c); return [p[0], p[1], p[2], p[3], size]; };

  const proj = p => projectPt(p, basis, W, H, APP.cam.fov);

  // Drum (cylinder wireframe)
  if (showDrum) {
    const dr = bundle.mesh.drum;
    const N = 40;
    const drumCol = col('rgba(207, 224, 240, 0.22)');
    // Rings as connected segments (top + bottom) plus radial spokes,
    // in WORLD coordinates — WebGL projects them via the VP matrix.
    const addRing = (z) => {
      let prev = null;
      for (let i = 0; i <= N; i++) {
        const a = (i / N) * Math.PI * 2;
        const pt = [dr.radius * Math.cos(a), dr.radius * Math.sin(a), z];
        if (prev) line.push(prev, [pt[0], pt[1], pt[2], drumCol[0], drumCol[1], drumCol[2], drumCol[3], 0]);
        prev = [pt[0], pt[1], pt[2], drumCol[0], drumCol[1], drumCol[2], drumCol[3], 0];
      }
    };
    addRing(dr.z_min);
    addRing(dr.z_max);
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      const p1 = [dr.radius * Math.cos(a), dr.radius * Math.sin(a), dr.z_min];
      const p2 = [dr.radius * Math.cos(a), dr.radius * Math.sin(a), dr.z_max];
      line.push([p1[0], p1[1], p1[2], drumCol[0], drumCol[1], drumCol[2], drumCol[3], 0], [p2[0], p2[1], p2[2], drumCol[0], drumCol[1], drumCol[2], drumCol[3], 0]);
    }
  }

  // Cells: faces (triangles) + wireframe edges
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
      // Triangulate each polygon as a fan; world coords go straight to the GPU.
      const fc = col(faceFill);
      for (const poly of cell.faces) {
        for (let i = 1; i < poly.length - 1; i++) {
          tri.push(
            [poly[0][0], poly[0][1], poly[0][2], fc[0], fc[1], fc[2], fc[3], 0],
            [poly[i][0], poly[i][1], poly[i][2], fc[0], fc[1], fc[2], fc[3], 0],
            [poly[i + 1][0], poly[i + 1][1], poly[i + 1][2], fc[0], fc[1], fc[2], fc[3], 0]
          );
        }
      }
    }

    if (showWire) {
      const strokeCol = isSel
        ? '#ffffff'
        : (colorMode === 'cell_id' ? hexToRgba(baseHex, 0.58) : 'rgba(207, 224, 240, 0.30)');
      const lc = col(strokeCol);
      for (const [a, b] of cell.edges) {
        line.push([a[0], a[1], a[2], lc[0], lc[1], lc[2], lc[3], 0], [b[0], b[1], b[2], lc[0], lc[1], lc[2], lc[3], 0]);
      }
    }
  }

  // Split planes
  for (const sp of (bundle.mesh.split_planes || [])) {
    const sc = col('rgba(125,211,189,0.14)');
    const stroke = col('rgba(125,211,189,0.75)');
    const quad = sp.quad;
    for (let i = 1; i < quad.length - 1; i++) {
      tri.push(
        [quad[0][0], quad[0][1], quad[0][2], sc[0], sc[1], sc[2], sc[3], 0],
        [quad[i][0], quad[i][1], quad[i][2], sc[0], sc[1], sc[2], sc[3], 0],
        [quad[i + 1][0], quad[i + 1][1], quad[i + 1][2], sc[0], sc[1], sc[2], sc[3], 0]
      );
    }
    for (let i = 0; i < quad.length; i++) {
      const a = quad[i], b = quad[(i + 1) % quad.length];
      line.push([a[0], a[1], a[2], stroke[0], stroke[1], stroke[2], stroke[3], 0], [b[0], b[1], b[2], stroke[0], stroke[1], stroke[2], stroke[3], 0]);
    }
  }

  // Top edges (arrows) as lines + a small cone (triangle) at the tip
  if (showEdges && bundle.top_edges_small) {
    const cenById = new Map(bundle.mesh.cells.map(c => [c.id, c.centroid]));
    for (const [si, dj, prob] of bundle.top_edges_small) {
      const cs = cenById.get(si), cd = cenById.get(dj);
      if (!cs || !cd) continue;
      const colc = parseColor(`rgba(158, 192, 220, ${Math.min(0.85, 0.22 + prob * 1.3)})`);
      const p = [cs[0], cs[1], cs[2], colc[0], colc[1], colc[2], colc[3], 0];
      const p2 = [cd[0], cd[1], cd[2], colc[0], colc[1], colc[2], colc[3], 0];
      line.push(p, p2);
      // Cone head: two short segments forming a V
      const dx = cd[0] - cs[0], dy = cd[1] - cs[1], dz = cd[2] - cs[2];
      const dl = Math.hypot(dx, dy, dz) || 1e-9;
      const ux = dx / dl, uy = dy / dl, uz = dz / dl;
      const hl = 0.012;
      const bx = cd[0] - ux * hl, by = cd[1] - uy * hl, bz = cd[2] - uz * hl;
      // approximate a perpendicular
      const px = Math.abs(ux) < 0.9 ? ux : 0;
      const py = Math.abs(uy) < 0.9 ? uy : 0;
      const pz = Math.abs(uz) < 0.9 ? uz : 0;
      const nx = uy * pz - uz * py, ny = uz * px - ux * pz, nz = ux * py - uy * px;
      const nl = Math.hypot(nx, ny, nz) || 1e-9;
      const e1 = [bx + (nx / nl) * hl * 0.5, by + (ny / nl) * hl * 0.5, bz + (nz / nl) * hl * 0.5];
      const e2 = [bx - (nx / nl) * hl * 0.5, by - (ny / nl) * hl * 0.5, bz - (nz / nl) * hl * 0.5];
      tri.push(
        [cd[0], cd[1], cd[2], colc[0], colc[1], colc[2], colc[3], 0],
        [e1[0], e1[1], e1[2], colc[0], colc[1], colc[2], colc[3], 0],
        [e2[0], e2[1], e2[2], colc[0], colc[1], colc[2], colc[3], 0]
      );
    }
  }

  // Particles / spheres
  if (frame && (showSmall || showLarge)) {
    const diams = frame.diameters || [];
    for (let i = 0; i < frame.coords.length; i++) {
      const isS = frame.is_small[i];
      if ((isS && !showSmall) || (!isS && !showLarge)) continue;
      const p = proj(frame.coords[i]);
      if (!p) continue;
      const d = diams[i] || (isS ? 0.004 : 0.008);
      const rad = Math.max(isS ? 2.1 : 4.2, Math.min(isS ? 5.0 : 10.0, p.scale * d * 0.56));
      const sc = isS ? col('#7dd3c8', rad * dpr) : col('#eec078', rad * dpr);
      point.push([frame.coords[i][0], frame.coords[i][1], frame.coords[i][2], sc[0], sc[1], sc[2], sc[3], sc[4]]);
      if (showVel && frame.velocities[i]) {
        const v = frame.velocities[i];
        const tip = proj([frame.coords[i][0] + v[0] * 0.024, frame.coords[i][1] + v[1] * 0.024, frame.coords[i][2] + v[2] * 0.024]);
        if (tip) {
          const vc = col('rgba(207,224,240,0.65)');
          line.push([frame.coords[i][0], frame.coords[i][1], frame.coords[i][2], vc[0], vc[1], vc[2], vc[3], 0], [frame.coords[i][0] + v[0] * 0.024, frame.coords[i][1] + v[1] * 0.024, frame.coords[i][2] + v[2] * 0.024, vc[0], vc[1], vc[2], vc[3], 0]);
        }
      }
    }
  }

  // Upload & draw: faces first (write depth), then lines, then points.
  const triBuf = [];
  const lineBuf = [];
  const pointBuf = [];
  buildBuffer(triBuf, tri);
  buildBuffer(lineBuf, line);
  buildBuffer(pointBuf, point);
  drawBuffer(gl, locs, triBuf[0], gl.TRIANGLES);
  drawBuffer(gl, locs, lineBuf[0], gl.LINES);
  drawBuffer(gl, locs, pointBuf[0], gl.POINTS);
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
