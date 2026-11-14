// Centralized reactive application state.
//
// Every shared variable (dataset, method, parameters, UI panels, frame
// cursor, color mode, …) lives here in ONE place. A variable updated from
// any part of the software is immediately propagated to every other part:
// views subscribe to the keys they depend on and re-render automatically,
// and DOM controls bound to a key stay in sync through `bindControl`.
//
// `APP` is a reactive proxy over this store, so existing code that reads
// `APP.x` or writes `APP.x = y` keeps working while changes now trigger
// notifications to all subscribers.

function createStore(initial) {
  const state = Object.assign({}, initial);
  const subs = new Map(); // key -> Set<fn>

  function get(key) {
    if (key == null) return state;
    return state[key];
  }

  function set(key, value) {
    if (Object.is(state[key], value)) return false;
    state[key] = value;
    const setForKey = subs.get(key);
    if (setForKey) setForKey.forEach(fn => { try { fn(value, key); } catch (_) {} });
    const all = subs.get('*');
    if (all) all.forEach(fn => { try { fn(state, key); } catch (_) {} });
    return true;
  }

  function subscribe(key, fn) {
    if (!subs.has(key)) subs.set(key, new Set());
    subs.get(key).add(fn);
    return () => { const s = subs.get(key); if (s) s.delete(fn); };
  }

  return { get, set, subscribe, state };
}

const STORE = createStore({
  staticMode: false,
  theme: 'dark', // 'dark' | 'light' — user-selectable, persisted
  workbenchMode: 'explore', // 'explore' | 'study' | 'compare'
  executionStatus: 'idle',  // 'idle' | 'computing' | 'error'
  activeView: 'summary',
  methods: [],
  bundleA: null,
  bundleB: null,
  benchRows: null,
  studyPresets: [],
  studyBundle: null,
  studySelectedCell: { idx1: 0, idx2: 0 },
  frameIdx: 0,
  playing: false,
  playTimer: null,
  selectedCell: null,
  selectedTransitionKey: null, // e.g. "0->3"
  evolPage: 0,
  evolZoomedKey: null,
  cam: { yaw: 0.68, pitch: 0.44, distance: 0.34, target: [0, -0.01, 0], fov: 42 },
  drag: { active: false, mode: 'orbit', x: 0, y: 0 },
  projectedCells: [],
  // Shared simulation parameters (single source of truth for the whole UI)
  dataset: 'synthetic_drum',
  method: 'cartesian',
  methodB: 'voronoi',
  // Partitioning parameters per method (single source of truth shared by the
  // parameter inputs AND the simulation/visualization: method -> kwargs object)
  methodKwargs: {},
  targetSpecies: 'small',
  noSpecies: false,
  predictFullDuration: true,
  nlt: 3,
  step: 30,
  dt: 1,
  tau: 12,
  startIndex: 40,
  // Shared 3-D layer toggles & color mode
  showSmall: true,
  showLarge: true,
  showWire: true,
  showFaces: true,
  showVel: false,
  showEdges: true,
  showDrum: true,
  colorMode: 'dem_teneur',
  // UI layout
  ui: {
    treeOpen: true,
    ribbonOpen: true,
    timelineOpen: true,
    studyBuilderOpen: true,
    inspectorOpen: true,
  },
});

// Reactive proxy: reads/writes transparently map to the store and fire
// subscribers on every change, so any update propagates everywhere.
const APP = new Proxy({}, {
  get: (_t, key) => STORE.get(key),
  set: (_t, key, value) => { STORE.set(key, value); return true; },
});

// Bind a DOM control (select / checkbox / text input) to a store key.
// The control is initialized from the store and updated whenever the store
// changes, from ANY part of the software — guaranteeing one-way propagation
// and keeping every bound element in sync automatically.
function bindControl(key, el, opts = {}) {
  if (!el) return;
  const value = STORE.get(key);
  const apply = (v) => {
    if (el.tagName === 'SELECT') {
      if (opts.optionValue !== undefined) el.value = String(v);
      else el.value = String(v);
    } else if (el.type === 'checkbox') {
      el.checked = Boolean(v);
    } else {
      el.value = v == null ? '' : String(v);
    }
  };
  apply(value);
  STORE.subscribe(key, (v) => apply(v));
  const onInput = () => {
    let v;
    if (el.tagName === 'SELECT') v = el.value;
    else if (el.type === 'checkbox') v = el.checked;
    else v = el.value;
    if (opts.parse) v = opts.parse(v);
    STORE.set(key, v);
  };
  el.addEventListener('change', onInput);
  if (el.type === 'text' || el.tagName === 'INPUT') {
    el.addEventListener('input', onInput);
  }
}

// ── Non-blocking toast notifications ─────────────────────────────────────
// Replaces blocking alert() with a small stacked message queue in the
// bottom-right corner. `toast(msg, type)` where type is one of
// 'info' (default), 'success', 'warn', 'error'.
const TOAST_ICONS = { info: 'ℹ', success: '✓', warn: '⚠', error: '✕' };
function toast(message, type = 'info', timeout = 3500) {
  if (typeof document === 'undefined') return;
  const container = document.getElementById('toast-container');
  if (!container) return;
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML =
    `<span class="t-icon">${TOAST_ICONS[type] || TOAST_ICONS.info}</span>` +
    `<span class="t-msg"></span>`;
  el.querySelector('.t-msg').textContent = message;
  container.appendChild(el);
  let timer = null;
  const remove = () => {
    if (timer) clearTimeout(timer);
    if (!el.parentElement) return;
    el.classList.add('hide');
    setTimeout(() => { if (el.parentElement) el.remove(); }, 220);
  };
  el.addEventListener('click', remove);
  if (timeout > 0) timer = setTimeout(remove, timeout);
}

// ── Run serialization (debounce + stale-response guard) ───────────────────
// Rapid parameter changes would otherwise fire one full server simulation per
// keystroke, overlapping requests and letting a slower older response overwrite
// a fresher one. `debounce` coalesces bursts; `beginRun`/`runIsCurrent` give
// each in-flight run a token so stale completions are discarded.
function debounce(fn, delay = 350) {
  let timer = null;
  const wrapped = (...args) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; fn(...args); }, delay);
  };
  wrapped.cancel = () => { if (timer) clearTimeout(timer); timer = null; };
  return wrapped;
}

let RUN_SEQ = 0;
function beginRun() {
  RUN_SEQ += 1;
  return RUN_SEQ;
}
function runIsCurrent(token) {
  return token === RUN_SEQ;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { createStore, STORE, APP, bindControl, toast, debounce, beginRun, runIsCurrent };
}
