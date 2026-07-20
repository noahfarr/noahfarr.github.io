// Train a small policy on Wordle in your browser, swapping the torso behind one
// fixed PPO. The point of the blog post made interactive: the RecurrentPPO update
// step is compiled to WebAssembly by whlo (https://github.com/noahfarr/whlo), and
// only the network torso changes between the three buttons. The algorithm cannot
// tell which one it is driving.
//
//   /assets/pomdp/<model>/init.mlir        key         -> state
//   /assets/pomdp/<model>/train_step.mlir  state, key  -> state', metric
//
// If those modules are present the trainer runs them for real. Until they are (the
// Wordle env is being finalised), it runs a clearly-labelled illustrative preview
// so the interface is fully alive. The preview invents no benchmark: all three
// torsos learn under the same loop, which is the whole claim.

const MOUNT = "pomdp-trainer";
// Reuse the compiler the MCTS demo already vendors, so there is one whlo on the site.
const WHLO = "/assets/js/mcts/whlo/src/index.mjs";
const MODEL_DIR = (m) => `/assets/pomdp/${m}`;

const MODELS = [
  { id: "gru", label: "GRU", color: "#3b82f6", note: "sequential scan · BPTT" },
  { id: "min_gru", label: "MinGRU", color: "#22c55e", note: "associative scan" },
  { id: "attention", label: "Attention", color: "#f59e0b", note: "KV-cache attention" },
];

const MAX_UPDATES = 160; // stop each run here so the plot stays readable
const METRIC_LABEL = "episodes solved";

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

const el = (tag, cls) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
};
const raf = () => new Promise((r) => requestAnimationFrame(r));
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

function randKey() {
  const k = new Uint32Array(2);
  crypto.getRandomValues(k);
  return k;
}

// A tiny deterministic PRNG so a model's preview curve is stable within a run.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// map whlo positional inputs -> named, run, return positional outputs
function call(exe, inputs) {
  const named = {};
  exe.inputs.forEach((slot, i) => (named[slot.name] = inputs[i]));
  const out = exe.run(named);
  return exe.outputs.map((slot) => out[slot.name]);
}

// ---------------------------------------------------------------------------
// backends: both expose { live, reset(), step() -> metric in [0,1] }
// ---------------------------------------------------------------------------

// Real training: compile init + train_step for one model and step the PPO update.
// train_step is expected to return the new state leaves followed by one scalar
// metric leaf (the rollout solved-rate); its index is read from layout.json.
class WhloBackend {
  constructor(compile, initExe, stepExe, metricIndex) {
    this.live = true;
    this.compile = compile;
    this.initExe = initExe;
    this.stepExe = stepExe;
    this.metricIndex = metricIndex;
    this.reset();
  }
  static async load(modelId) {
    const { compile, initCompiler } = await import(WHLO);
    await initCompiler();
    const dir = MODEL_DIR(modelId);
    const [initSrc, stepSrc, layout] = await Promise.all([
      fetch(`${dir}/init.mlir`).then((r) => (r.ok ? r.text() : Promise.reject(r.status))),
      fetch(`${dir}/train_step.mlir`).then((r) => (r.ok ? r.text() : Promise.reject(r.status))),
      fetch(`${dir}/layout.json`).then((r) => (r.ok ? r.json() : null)),
    ]);
    const [initExe, stepExe] = await Promise.all([compile(initSrc), compile(stepSrc)]);
    // the metric is the last output unless layout.json marks one
    const metricIndex = (layout && layout.metric_index != null)
      ? layout.metric_index
      : stepExe.outputs.length - 1;
    return new WhloBackend(compile, initExe, stepExe, metricIndex);
  }
  reset() {
    this.state = call(this.initExe, [randKey()]);
    this.updates = 0;
  }
  step() {
    // train_step returns [state leaves..., metric]; keep the leaves, read the metric
    const out = call(this.stepExe, [...this.state, randKey()]);
    const metric = Number(out[this.metricIndex][0]);
    this.state = out.slice(0, this.metricIndex);
    this.updates++;
    return clamp01(metric);
  }
}

// Illustrative preview: a plausible PPO learning curve. No claimed benchmark, just
// "it rises under the same loop". Each torso gets a mild, noisy climb to a plateau.
class MockBackend {
  constructor(modelId) {
    this.live = false;
    const m = MODELS.findIndex((x) => x.id === modelId);
    // gentle per-torso spread so the three curves are distinguishable, not a result
    this.tau = 34 + m * 6;
    this.plateau = 0.7 + m * 0.05;
    this.reset(modelId);
  }
  reset(modelId = this.modelId) {
    this.modelId = modelId;
    this.rng = mulberry32(0x9e37 + this.modelId.length * 101 + MODELS.findIndex((x) => x.id === this.modelId));
    this.updates = 0;
    this.smooth = 0.02 + 0.03 * this.rng();
  }
  step() {
    const t = this.updates++;
    const target = this.plateau * (1 - Math.exp(-t / this.tau));
    // random-walk toward the target with exploration noise, plus a small early dip
    const noise = (this.rng() - 0.5) * 0.11 * Math.exp(-t / 90);
    this.smooth += 0.14 * (target - this.smooth) + noise;
    return clamp01(this.smooth);
  }
}

// ---------------------------------------------------------------------------
// plot: dependency-free canvas line chart, one series per model, theme-aware
// ---------------------------------------------------------------------------

class Plot {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = new Map(); // id -> { color, data:[], active }
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas);
  }
  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    this.w = Math.max(1, Math.floor(rect.width));
    this.h = Math.max(1, Math.floor(rect.height));
    this.canvas.width = this.w * dpr;
    this.canvas.height = this.h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }
  ensure(id, color) {
    if (!this.series.has(id)) this.series.set(id, { color, data: [] });
    return this.series.get(id);
  }
  push(id, color, value) {
    this.ensure(id, color).data.push(value);
    this.draw();
  }
  clear(id) {
    if (this.series.has(id)) this.series.get(id).data = [];
    this.draw();
  }
  setActive(id) {
    this.active = id;
    this.draw();
  }
  _textColor() {
    // read the inherited text color so axes adapt to light/dark themes
    return getComputedStyle(this.canvas).color || "#888";
  }
  draw() {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    const padL = 34, padR = 10, padT = 12, padB = 22;
    const x0 = padL, x1 = w - padR, y0 = h - padB, y1 = padT;
    const ink = this._textColor();

    // axes + gridlines
    ctx.lineWidth = 1;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const v = i / 4;
      const y = y0 + (y1 - y0) * v;
      ctx.strokeStyle = withAlpha(ink, i === 0 ? 0.35 : 0.12);
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      ctx.fillStyle = withAlpha(ink, 0.55);
      ctx.textAlign = "right";
      ctx.fillText(Math.round(v * 100) + "%", x0 - 6, y);
    }
    // x-axis label
    ctx.fillStyle = withAlpha(ink, 0.55);
    ctx.textAlign = "center";
    ctx.fillText("training updates →", (x0 + x1) / 2, h - 6);

    const sx = (i) => x0 + (x1 - x0) * (i / (MAX_UPDATES - 1));
    const sy = (v) => y0 + (y1 - y0) * clamp01(v);

    // draw inactive series faded, active on top and bold with a head dot
    const entries = [...this.series.entries()].sort(
      ([a], [b]) => (a === this.active ? 1 : 0) - (b === this.active ? 1 : 0)
    );
    for (const [id, s] of entries) {
      if (!s.data.length) continue;
      const isActive = id === this.active;
      ctx.strokeStyle = withAlpha(s.color, isActive ? 1 : 0.32);
      ctx.lineWidth = isActive ? 2.2 : 1.4;
      ctx.lineJoin = "round";
      ctx.beginPath();
      s.data.forEach((v, i) => (i ? ctx.lineTo(sx(i), sy(v)) : ctx.moveTo(sx(i), sy(v))));
      ctx.stroke();
      if (isActive) {
        const i = s.data.length - 1;
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.arc(sx(i), sy(s.data[i]), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

// blend a hex or rgb color with an alpha, for gridlines/faded lines
function withAlpha(color, alpha) {
  if (color.startsWith("#")) {
    const n = parseInt(color.slice(1), 16);
    const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    return `rgba(${r},${g},${b},${alpha})`;
  }
  const m = color.match(/\d+/g);
  if (m && m.length >= 3) return `rgba(${m[0]},${m[1]},${m[2]},${alpha})`;
  return color;
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

const CSS = `
#${MOUNT} { --panel: rgba(127,127,127,.28); max-width: 42rem; margin: 1.8rem auto; font-variant-numeric: tabular-nums; color: inherit; }
#${MOUNT} .bar { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-bottom: .8rem; }
#${MOUNT} .seg { display: inline-flex; border: 1px solid var(--panel); border-radius: 999px; overflow: hidden; }
#${MOUNT} .seg button { font: inherit; font-weight: 600; font-size: .82rem; padding: .34rem .8rem; border: 0; background: transparent; color: inherit; cursor: pointer; transition: background .15s; }
#${MOUNT} .seg button + button { border-left: 1px solid var(--panel); }
#${MOUNT} .seg button:hover { background: rgba(127,127,127,.1); }
#${MOUNT} .seg button.on { color: #fff; }
#${MOUNT} .spacer { flex: 1 1 auto; }
#${MOUNT} button.act { font: inherit; font-weight: 600; font-size: .82rem; padding: .34rem 1rem; border: 1px solid var(--panel); border-radius: 999px; background: transparent; color: inherit; cursor: pointer; transition: background .15s, border-color .15s; }
#${MOUNT} button.act:hover { background: rgba(127,127,127,.1); }
#${MOUNT} button.act.primary { border-color: transparent; color: #fff; }
#${MOUNT} button.act:disabled { opacity: .45; cursor: default; }
#${MOUNT} .panel { border: 1px solid var(--panel); border-radius: 12px; padding: .6rem .7rem .3rem; }
#${MOUNT} canvas { display: block; width: 100%; height: 200px; }
#${MOUNT} .foot { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem 1rem; margin-top: .5rem; font-size: .8rem; }
#${MOUNT} .readout { font-weight: 600; }
#${MOUNT} .readout .muted { opacity: .55; font-weight: 400; }
#${MOUNT} .badge { font-size: .68rem; font-weight: 600; letter-spacing: .02em; text-transform: uppercase; padding: .12rem .5rem; border-radius: 999px; border: 1px solid var(--panel); opacity: .8; }
#${MOUNT} .badge.live { color: #22c55e; border-color: rgba(34,197,94,.5); }
#${MOUNT} .badge.preview { color: #f59e0b; border-color: rgba(245,158,11,.5); }
#${MOUNT} .cap { margin-top: .55rem; font-size: .74rem; opacity: .62; line-height: 1.5; }
`;

function mount(root) {
  const style = el("style");
  style.textContent = CSS;
  root.appendChild(style);

  const bar = el("div", "bar");
  const seg = el("div", "seg");
  const modelBtns = MODELS.map((m) => {
    const b = el("button");
    b.textContent = m.label;
    b.dataset.id = m.id;
    b.title = m.note;
    seg.appendChild(b);
    return b;
  });
  const spacer = el("div", "spacer");
  const trainBtn = el("button", "act primary");
  trainBtn.textContent = "Train";
  const resetBtn = el("button", "act");
  resetBtn.textContent = "Reset";
  bar.append(seg, spacer, trainBtn, resetBtn);

  const panel = el("div", "panel");
  const canvas = el("canvas");
  panel.appendChild(canvas);

  const foot = el("div", "foot");
  const readout = el("div", "readout");
  const badge = el("div", "badge");
  foot.append(readout, badge);

  const cap = el("div", "cap");
  cap.innerHTML =
    "Same PPO loop, compiled to WebAssembly by " +
    '<a href="https://github.com/noahfarr/whlo">whlo</a>. Only the torso changes ' +
    "between the buttons — the algorithm never knows which network it is training.";

  root.append(bar, panel, foot, cap);
  return { modelBtns, trainBtn, resetBtn, canvas, readout, badge };
}

// ---------------------------------------------------------------------------
// controller
// ---------------------------------------------------------------------------

async function main() {
  const root = document.getElementById(MOUNT);
  if (!root) return;
  const ui = mount(root);
  const plot = new Plot(ui.canvas);

  const state = {
    modelId: MODELS[0].id,
    running: false,
    backends: new Map(), // modelId -> backend
    metrics: new Map(), // modelId -> latest value
  };

  const colorOf = (id) => MODELS.find((m) => m.id === id).color;

  function paintSelector() {
    ui.modelBtns.forEach((b) => {
      const on = b.dataset.id === state.modelId;
      b.classList.toggle("on", on);
      b.style.background = on ? colorOf(b.dataset.id) : "";
    });
    ui.trainBtn.style.background = state.running ? "" : colorOf(state.modelId);
    ui.trainBtn.classList.toggle("primary", !state.running);
    plot.setActive(state.modelId);
  }

  function setBadge(live) {
    ui.badge.className = "badge " + (live ? "live" : "preview");
    ui.badge.textContent = live ? "live in-browser" : "illustrative preview";
  }

  function setReadout() {
    const b = state.backends.get(state.modelId);
    const m = state.metrics.get(state.modelId);
    const label = MODELS.find((x) => x.id === state.modelId).label;
    const upd = b ? b.updates : 0;
    ui.readout.innerHTML =
      `${label} <span class="muted">· update ${upd} · ` +
      (m == null ? "—" : `${Math.round(m * 100)}% ${METRIC_LABEL}`) +
      "</span>";
  }

  // lazily build a backend for a model: try real whlo modules, else preview
  async function backendFor(modelId) {
    if (state.backends.has(modelId)) return state.backends.get(modelId);
    let backend;
    try {
      backend = await WhloBackend.load(modelId);
    } catch (_) {
      backend = new MockBackend(modelId);
    }
    state.backends.set(modelId, backend);
    plot.ensure(modelId, colorOf(modelId));
    return backend;
  }

  async function loop() {
    const backend = await backendFor(state.modelId);
    setBadge(backend.live);
    // preview steps a few times per frame to feel responsive; real training is
    // heavier, so one step per frame keeps the tab live.
    const perFrame = backend.live ? 1 : 3;
    while (state.running && backend.updates < MAX_UPDATES) {
      let metric;
      for (let i = 0; i < perFrame && backend.updates < MAX_UPDATES; i++) {
        metric = backend.step();
        plot.push(state.modelId, colorOf(state.modelId), metric);
      }
      state.metrics.set(state.modelId, metric);
      setReadout();
      await raf();
    }
    if (backend.updates >= MAX_UPDATES) stop();
  }

  function start() {
    if (state.running) return;
    state.running = true;
    ui.trainBtn.textContent = "Pause";
    paintSelector();
    loop();
  }
  function stop() {
    state.running = false;
    ui.trainBtn.textContent = "Train";
    paintSelector();
  }

  ui.trainBtn.addEventListener("click", () => (state.running ? stop() : start()));
  ui.resetBtn.addEventListener("click", async () => {
    stop();
    const backend = await backendFor(state.modelId);
    backend.reset(state.modelId);
    state.metrics.delete(state.modelId);
    plot.clear(state.modelId);
    setReadout();
  });
  ui.modelBtns.forEach((b) =>
    b.addEventListener("click", async () => {
      stop();
      state.modelId = b.dataset.id;
      paintSelector();
      const backend = await backendFor(state.modelId);
      setBadge(backend.live);
      setReadout();
    })
  );

  // first paint
  paintSelector();
  const first = await backendFor(state.modelId);
  setBadge(first.live);
  setReadout();
}

main();
