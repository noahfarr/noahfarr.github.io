// Train a small policy on Wordle in your browser, swapping the torso behind one
// fixed PPO. The blog post made interactive: the RecurrentPPO update step is
// compiled to WebAssembly by whlo (https://github.com/noahfarr/whlo); only the
// network torso changes between the three buttons, and the algorithm cannot tell
// which one it is driving.
//
//   /assets/pomdp/<model>/init.mlir        key         -> state
//   /assets/pomdp/<model>/train_step.mlir  state, key  -> state', metric
//
// If those modules are present the trainer runs them for real. Until they are (the
// Wordle env is being finalised), it runs a clearly-labelled illustrative preview
// so the interface is fully alive. The preview invents no benchmark; all three
// torsos climb under the same loop, which is the whole claim.
//
// Colours are the dataviz reference palette's first three categorical slots
// (blue / green / magenta), validated all-pairs on this site's light and dark
// card surfaces. Theme, ink, and accent come from al-folio's own CSS variables so
// the widget is native to the site's light/dark toggle.

const MOUNT = "pomdp-trainer";
const WHLO = "/assets/js/mcts/whlo/src/index.mjs"; // reuse the vendored compiler
const MODEL_DIR = (m) => `/assets/pomdp/${m}`;

const MODELS = [
  { id: "gru", label: "GRU", varName: "--s1", note: "sequential scan · BPTT" },
  { id: "min_gru", label: "MinGRU", varName: "--s2", note: "associative scan" },
  { id: "attention", label: "Attention", varName: "--s3", note: "KV-cache attention" },
  // revealed only after a policy is uploaded; compiled from the user's own modules
  { id: "custom", label: "Yours", varName: "--s4", note: "uploaded policy", custom: true },
];

const MAX_UPDATES = 160; // stop each run here so the plot stays readable
const METRIC_LABEL = "Episodes solved";
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------------------
// helpers
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

function call(exe, inputs) {
  const named = {};
  exe.inputs.forEach((slot, i) => (named[slot.name] = inputs[i]));
  const out = exe.run(named);
  return exe.outputs.map((slot) => out[slot.name]);
}

// ---------------------------------------------------------------------------
// backends: both expose { live, updates, reset(id?), step() -> [0,1] }
// ---------------------------------------------------------------------------

class WhloBackend {
  constructor(compile, initExe, stepExe, metricIndex) {
    this.live = true;
    this.initExe = initExe;
    this.stepExe = stepExe;
    this.metricIndex = metricIndex;
    this.reset();
  }
  // compile a StableHLO init + train_step pair; used for both bundled models and
  // uploaded policies, which is why the compiler import lives here.
  static async build(initSrc, stepSrc, layout) {
    const { compile, initCompiler } = await import(WHLO);
    await initCompiler();
    const [initExe, stepExe] = await Promise.all([compile(initSrc), compile(stepSrc)]);
    const metricIndex = layout && layout.metric_index != null ? layout.metric_index : stepExe.outputs.length - 1;
    return new WhloBackend(compile, initExe, stepExe, metricIndex);
  }
  static async load(modelId) {
    const dir = MODEL_DIR(modelId);
    const [initSrc, stepSrc, layout] = await Promise.all([
      fetch(`${dir}/init.mlir`).then((r) => (r.ok ? r.text() : Promise.reject(r.status))),
      fetch(`${dir}/train_step.mlir`).then((r) => (r.ok ? r.text() : Promise.reject(r.status))),
      fetch(`${dir}/layout.json`).then((r) => (r.ok ? r.json() : null)),
    ]);
    return WhloBackend.build(initSrc, stepSrc, layout);
  }
  static fromSources(initSrc, stepSrc, layout) {
    return WhloBackend.build(initSrc, stepSrc, layout);
  }
  reset() {
    this.state = call(this.initExe, [randKey()]);
    this.updates = 0;
  }
  step() {
    const out = call(this.stepExe, [...this.state, randKey()]);
    const metric = Number(out[this.metricIndex][0]);
    this.state = out.slice(0, this.metricIndex);
    this.updates++;
    return clamp01(metric);
  }
}

class MockBackend {
  constructor(modelId) {
    this.live = false;
    const m = MODELS.findIndex((x) => x.id === modelId);
    this.tau = 34 + m * 6;
    this.plateau = 0.7 + m * 0.05;
    this.reset(modelId);
  }
  reset(modelId = this.modelId) {
    this.modelId = modelId;
    this.rng = mulberry32(0x9e37 + MODELS.findIndex((x) => x.id === this.modelId) * 2654435761);
    this.updates = 0;
    this.smooth = 0.02 + 0.03 * this.rng();
  }
  step() {
    const t = this.updates++;
    const target = this.plateau * (1 - Math.exp(-t / this.tau));
    const noise = (this.rng() - 0.5) * 0.11 * Math.exp(-t / 90);
    this.smooth += 0.14 * (target - this.smooth) + noise;
    return clamp01(this.smooth);
  }
}

// ---------------------------------------------------------------------------
// plot: dependency-free canvas line chart, area-gradient on the active series,
// hover crosshair + tooltip, direct head labels. Reads theme colours live.
// ---------------------------------------------------------------------------

class Plot {
  constructor(canvas, root, labelOf) {
    this.canvas = canvas;
    this.root = root;
    this.labelOf = labelOf;
    this.ctx = canvas.getContext("2d");
    this.series = new Map(); // id -> { data: [] }
    this.active = null;
    this.hover = null; // x-index under the pointer
    this.running = false;
    this.pulse = 0;
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas);
    canvas.addEventListener("pointermove", (e) => this._onMove(e));
    canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      this.draw();
    });
  }
  _cssVar(name) {
    return getComputedStyle(this.root).getPropertyValue(name).trim() || "#888";
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
  ensure(id) {
    if (!this.series.has(id)) this.series.set(id, { data: [] });
    return this.series.get(id);
  }
  push(id, value) {
    this.ensure(id).data.push(value);
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
  setRunning(on) {
    this.running = on;
    if (on && !REDUCED) this._tick();
  }
  _tick() {
    if (!this.running) return;
    this.pulse = (this.pulse + 0.05) % 1;
    this.draw();
    requestAnimationFrame(() => this._tick());
  }
  _geom() {
    const padL = 38,
      padR = 14,
      padT = 16,
      padB = 24;
    return {
      x0: padL,
      x1: this.w - padR,
      y0: this.h - padB,
      y1: padT,
      sx: (i) => padL + (this.w - padR - padL) * (i / (MAX_UPDATES - 1)),
      sy: (v) => this.h - padB + (padT - (this.h - padB)) * clamp01(v),
    };
  }
  _onMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const g = this._geom();
    const px = e.clientX - rect.left;
    const t = (px - g.x0) / (g.x1 - g.x0);
    const maxLen = Math.max(0, ...[...this.series.values()].map((s) => s.data.length));
    if (maxLen < 2 || t < 0 || t > 1) {
      this.hover = null;
    } else {
      this.hover = Math.round(t * (MAX_UPDATES - 1));
    }
    this.draw();
  }
  draw() {
    const { ctx, w, h } = this;
    const g = this._geom();
    ctx.clearRect(0, 0, w, h);
    const ink = this._cssVar("--ink");
    const muted = this._cssVar("--muted");
    const line = this._cssVar("--line");
    const surface = this._cssVar("--surface");

    ctx.font = "600 10.5px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";

    // horizontal gridlines + y labels
    for (let i = 0; i <= 4; i++) {
      const v = i / 4;
      const y = g.sy(v);
      ctx.strokeStyle = rgba(line, i === 0 ? 1 : 0.5);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(g.x0, y);
      ctx.lineTo(g.x1, y);
      ctx.stroke();
      ctx.fillStyle = muted;
      ctx.textAlign = "right";
      ctx.fillText(Math.round(v * 100) + "%", g.x0 - 8, y);
    }
    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    ctx.fillText("Training updates", (g.x0 + g.x1) / 2, h - 8);

    const order = [...this.series.keys()].sort((a, b) => (a === this.active ? 1 : 0) - (b === this.active ? 1 : 0));

    // crosshair (drawn under the lines)
    if (this.hover != null) {
      const anyData = [...this.series.values()].some((s) => this.hover < s.data.length);
      if (anyData) {
        const x = g.sx(this.hover);
        ctx.strokeStyle = rgba(ink, 0.22);
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x, g.y1);
        ctx.lineTo(x, g.y0);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    for (const id of order) {
      const s = this.series.get(id);
      if (!s.data.length) continue;
      const isActive = id === this.active;
      const color = this._cssVar(MODELS.find((m) => m.id === id).varName);

      // area fill under the active line
      if (isActive) {
        const grad = ctx.createLinearGradient(0, g.y1, 0, g.y0);
        grad.addColorStop(0, rgba(color, 0.22));
        grad.addColorStop(1, rgba(color, 0));
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(g.sx(0), g.y0);
        s.data.forEach((v, i) => ctx.lineTo(g.sx(i), g.sy(v)));
        ctx.lineTo(g.sx(s.data.length - 1), g.y0);
        ctx.closePath();
        ctx.fill();
      }

      // the line
      ctx.strokeStyle = isActive ? color : rgba(color, 0.3);
      ctx.lineWidth = isActive ? 2.2 : 1.4;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      s.data.forEach((v, i) => (i ? ctx.lineTo(g.sx(i), g.sy(v)) : ctx.moveTo(g.sx(i), g.sy(v))));
      ctx.stroke();

      // head dot (+ pulse when running) and direct label on the active series
      const i = s.data.length - 1;
      const hx = g.sx(i),
        hy = g.sy(s.data[i]);
      if (isActive) {
        if (this.running && !REDUCED) {
          const r = 5 + 5 * Math.sin(this.pulse * Math.PI);
          ctx.fillStyle = rgba(color, 0.18 * (1 - this.pulse));
          ctx.beginPath();
          ctx.arc(hx, hy, r + 3, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = surface;
        ctx.beginPath();
        ctx.arc(hx, hy, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(hx, hy, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // hover tooltip: every series' value at the hovered update
    if (this.hover != null) {
      const rows = order
        .map((id) => ({ id, s: this.series.get(id) }))
        .filter(({ s }) => this.hover < s.data.length)
        .map(({ id, s }) => ({
          id,
          color: this._cssVar(MODELS.find((m) => m.id === id).varName),
          v: s.data[this.hover],
        }));
      if (rows.length) {
        const pad = 8,
          lh = 15,
          bw = 116,
          bh = pad * 2 + 14 + rows.length * lh;
        let bx = g.sx(this.hover) + 10;
        if (bx + bw > g.x1) bx = g.sx(this.hover) - bw - 10;
        const by = g.y1 + 4;
        ctx.fillStyle = rgba(surface, 0.96);
        ctx.strokeStyle = rgba(ink, 0.12);
        roundRect(ctx, bx, by, bw, bh, 8);
        ctx.fill();
        ctx.stroke();
        ctx.textAlign = "left";
        ctx.fillStyle = muted;
        ctx.font = "600 10px ui-monospace, monospace";
        ctx.fillText(`update ${this.hover}`, bx + pad, by + pad + 5);
        ctx.font = "600 11px ui-monospace, monospace";
        rows.forEach((r, k) => {
          const ry = by + pad + 14 + k * lh + 6;
          ctx.fillStyle = r.color;
          ctx.beginPath();
          ctx.arc(bx + pad + 3, ry, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = ink;
          ctx.fillText(this.labelOf(r.id), bx + pad + 12, ry);
          ctx.textAlign = "right";
          ctx.fillText(Math.round(r.v * 100) + "%", bx + bw - pad, ry);
          ctx.textAlign = "left";
        });
      }
    }
  }
}

function rgba(color, alpha) {
  color = color.trim();
  if (color.startsWith("#")) {
    let hex = color.slice(1);
    if (hex.length === 3)
      hex = hex
        .split("")
        .map((c) => c + c)
        .join("");
    const n = parseInt(hex, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
  }
  const m = color.match(/[\d.]+/g);
  if (m && m.length >= 3) return `rgba(${m[0]},${m[1]},${m[2]},${alpha})`;
  return color;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ---------------------------------------------------------------------------
// styles, native to al-folio via its --global-* variables
// ---------------------------------------------------------------------------

const CSS = `
#${MOUNT}{
  --surface: var(--global-card-bg-color, #fff);
  --ink: var(--global-text-color, #111);
  --muted: var(--global-text-color-light, #777);
  --line: var(--global-divider-color, rgba(0,0,0,.1));
  --accent: var(--global-theme-color, #2a78d6);
  --s1:#2a78d6; --s2:#008300; --s3:#e87ba4; --s4:#eda100;
  --track: color-mix(in srgb, var(--ink) 7%, transparent);
  --elev: 0 1px 2px rgba(0,0,0,.05), 0 10px 30px -18px rgba(0,0,0,.35);
  max-width: 40rem; margin: 2rem auto; color: var(--ink);
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
}
html[data-theme="dark"] #${MOUNT}{ --s1:#3987e5; --s2:#008300; --s3:#d55181; --s4:#c98500; }
html[data-theme="light"] #${MOUNT}{ --s1:#2a78d6; --s2:#008300; --s3:#e87ba4; --s4:#eda100; }
@media (prefers-color-scheme: dark){
  #${MOUNT}:not([data-forced]){ --s1:#3987e5; --s2:#008300; --s3:#d55181; --s4:#c98500; }
}
#${MOUNT} *{ box-sizing: border-box; }
#${MOUNT} .card{
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 16px; box-shadow: var(--elev); padding: 1.1rem 1.1rem .9rem;
}
#${MOUNT} .head{ display:flex; align-items:center; gap:.75rem; flex-wrap:wrap; margin-bottom:1rem; }

/* sliding segmented control */
#${MOUNT} .seg{ position:relative; display:inline-flex; background:var(--track);
  border-radius:11px; padding:3px; }
#${MOUNT} .seg .thumb{ position:absolute; top:3px; bottom:3px; left:0; border-radius:9px;
  background:var(--surface); box-shadow:0 1px 2px rgba(0,0,0,.18), 0 0 0 1px var(--line);
  transition:transform .3s cubic-bezier(.32,.72,0,1), width .3s cubic-bezier(.32,.72,0,1); }
#${MOUNT} .seg button{ position:relative; z-index:1; display:inline-flex; align-items:center;
  gap:.42rem; font:inherit; font-weight:600; font-size:.82rem; letter-spacing:-.01em;
  padding:.4rem .8rem; border:0; background:transparent; color:var(--muted); cursor:pointer;
  border-radius:9px; transition:color .2s; }
#${MOUNT} .seg button .dot{ width:.5rem; height:.5rem; border-radius:50%; background:currentColor; flex:none; }
#${MOUNT} .seg button[aria-pressed="true"]{ color:var(--ink); }
#${MOUNT} .seg button:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }

#${MOUNT} .spacer{ flex:1 1 auto; }
#${MOUNT} .actions{ display:flex; align-items:center; gap:.5rem; }
#${MOUNT} button.icon{ display:inline-grid; place-items:center; width:2rem; height:2rem;
  border:1px solid var(--line); border-radius:10px; background:transparent; color:var(--muted);
  cursor:pointer; transition:background .15s, color .15s, border-color .15s; }
#${MOUNT} button.icon:hover{ background:color-mix(in srgb, var(--ink) 6%, transparent); color:var(--ink); }
#${MOUNT} button.train{ display:inline-flex; align-items:center; gap:.45rem; font:inherit;
  font-weight:650; font-size:.84rem; letter-spacing:-.01em; padding:.44rem .95rem;
  border:0; border-radius:11px; background:var(--accent); color:#fff; cursor:pointer;
  box-shadow:0 1px 2px rgba(0,0,0,.12); transition:transform .12s ease, filter .15s, box-shadow .15s; }
#${MOUNT} button.train:hover{ transform:translateY(-1px); filter:brightness(1.05);
  box-shadow:0 4px 14px -4px color-mix(in srgb, var(--accent) 60%, transparent); }
#${MOUNT} button.train:active{ transform:translateY(0); }
#${MOUNT} button.train.paused{ background:transparent; color:var(--ink);
  box-shadow:inset 0 0 0 1px var(--line); }
#${MOUNT} button:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }
#${MOUNT} svg{ display:block; }

/* hero readout row sits ABOVE the chart, never over it */
#${MOUNT} .readout{ display:flex; align-items:flex-end; justify-content:space-between;
  gap:1rem; margin-bottom:.5rem; }
#${MOUNT} .readout .big{ font-size:1.85rem; font-weight:700; line-height:1; letter-spacing:-.02em; }
#${MOUNT} .readout .sub{ font-size:.72rem; color:var(--muted); margin-top:.28rem; }
#${MOUNT} .stage{ position:relative; }
#${MOUNT} canvas{ display:block; width:100%; height:190px; cursor:crosshair; }
#${MOUNT} .badge{ display:inline-flex; align-items:center; gap:.32rem; font-size:.62rem; font-weight:700;
  letter-spacing:.04em; text-transform:uppercase; padding:.2rem .5rem; border-radius:999px;
  border:1px solid var(--line); color:var(--muted); white-space:nowrap; }
#${MOUNT} .badge .led{ width:.44rem; height:.44rem; border-radius:50%; background:currentColor; }
#${MOUNT} .badge.live{ color:var(--s2); }
#${MOUNT} .badge.live .led{ animation: pomdpBlink 1.6s ease-in-out infinite; }
#${MOUNT} .badge.preview{ color:#c98500; }
@keyframes pomdpBlink{ 0%,100%{opacity:1} 50%{opacity:.35} }

#${MOUNT} .msg{ min-height:1.1rem; margin-top:.5rem; font-size:.74rem; color:var(--muted); }
#${MOUNT} .msg.err{ color:#c98500; }
#${MOUNT} .msg.busy{ color:var(--accent); }
#${MOUNT} input[type=file]{ display:none; }
#${MOUNT} .cap{ margin-top:.7rem; padding-top:.7rem; border-top:1px solid var(--line);
  font-size:.73rem; color:var(--muted); line-height:1.55; }
#${MOUNT} .cap a{ color:var(--accent); text-decoration:none; }
#${MOUNT} .cap a:hover{ text-decoration:underline; }
@media (prefers-reduced-motion: reduce){
  #${MOUNT} .seg .thumb, #${MOUNT} button.train{ transition:none; }
  #${MOUNT} .badge.live .led{ animation:none; }
}
`;

const ICON_PLAY = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.5v9l7-4.5z" fill="currentColor"/></svg>';
const ICON_PAUSE =
  '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="2.5" y="1.5" width="2.6" height="9" rx="1" fill="currentColor"/><rect x="6.9" y="1.5" width="2.6" height="9" rx="1" fill="currentColor"/></svg>';
const ICON_RESET =
  '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M11.5 7a4.5 4.5 0 1 1-1.32-3.18"/><path d="M11.2 1.6v2.6H8.6"/></svg>';
const ICON_UPLOAD =
  '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 9V2.6"/><path d="M4.4 5 7 2.4 9.6 5"/><path d="M2.5 9.5v1a1.5 1.5 0 0 0 1.5 1.5h6a1.5 1.5 0 0 0 1.5-1.5v-1"/></svg>';

function mount(root) {
  const style = el("style");
  style.textContent = CSS;
  root.appendChild(style);

  const card = el("div", "card");

  const head = el("div", "head");
  const seg = el("div", "seg");
  const thumb = el("div", "thumb");
  seg.appendChild(thumb);
  const modelBtns = MODELS.map((m) => {
    const b = el("button");
    b.type = "button";
    b.setAttribute("aria-pressed", "false");
    b.title = m.note;
    b.dataset.id = m.id;
    b.innerHTML = `<span class="dot" style="color:var(${m.varName})"></span>${m.label}`;
    if (m.custom) b.style.display = "none"; // revealed after an upload succeeds
    seg.appendChild(b);
    return b;
  });
  const customBtn = modelBtns.find((b) => b.dataset.id === "custom");
  const spacer = el("div", "spacer");
  const actions = el("div", "actions");
  const uploadBtn = el("button", "icon");
  uploadBtn.type = "button";
  uploadBtn.setAttribute("aria-label", "Upload your own policy");
  uploadBtn.title = "Upload your own policy (init.mlir + train_step.mlir)";
  uploadBtn.innerHTML = ICON_UPLOAD;
  const fileInput = el("input");
  fileInput.type = "file";
  fileInput.accept = ".mlir,.json,application/json";
  fileInput.multiple = true;
  const resetBtn = el("button", "icon");
  resetBtn.type = "button";
  resetBtn.setAttribute("aria-label", "Reset this model");
  resetBtn.innerHTML = ICON_RESET;
  const trainBtn = el("button", "train");
  trainBtn.type = "button";
  trainBtn.innerHTML = `${ICON_PLAY}<span class="txt">Train</span>`;
  actions.append(uploadBtn, resetBtn, trainBtn);
  head.append(seg, spacer, actions);

  const readout = el("div", "readout");
  const hero = el("div");
  hero.innerHTML = '<div class="big">&nbsp;</div><div class="sub"></div>';
  const badge = el("div", "badge");
  readout.append(hero, badge);
  const stage = el("div", "stage");
  const canvas = el("canvas");
  stage.appendChild(canvas);

  const msg = el("div", "msg");

  const cap = el("div", "cap");
  cap.innerHTML =
    "The same PPO update, compiled to WebAssembly by " +
    '<a href="https://github.com/noahfarr/whlo">whlo</a>. Only the torso changes ' +
    "between the buttons, and the algorithm never knows which network it is training. " +
    "The upload button takes your own <code>init.mlir</code> and <code>train_step.mlir</code>, " +
    "exported from any network that follows the same <code>(state, key)</code> interface, and " +
    "whlo compiles and trains it right here.";

  card.append(head, readout, stage, msg, cap);
  root.append(card, fileInput);
  return { seg, thumb, modelBtns, customBtn, uploadBtn, fileInput, resetBtn, trainBtn, canvas, readout, badge, msg };
}

// ---------------------------------------------------------------------------
// controller
// ---------------------------------------------------------------------------

async function main() {
  const root = document.getElementById(MOUNT);
  if (!root) return;
  const ui = mount(root);
  const labelOf = (id) => MODELS.find((m) => m.id === id).label;
  const plot = new Plot(ui.canvas, root, labelOf);

  const state = {
    modelId: MODELS[0].id,
    running: false,
    backends: new Map(),
    metrics: new Map(),
    shown: new Map(), // animated hero value per model
  };

  const bigEl = ui.readout.querySelector(".big");
  const subEl = ui.readout.querySelector(".sub");

  function positionThumb() {
    const btn = ui.modelBtns.find((b) => b.dataset.id === state.modelId);
    if (!btn) return;
    ui.thumb.style.width = btn.offsetWidth + "px";
    ui.thumb.style.transform = `translateX(${btn.offsetLeft - 3}px)`;
  }

  function paintSelector() {
    ui.modelBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === state.modelId)));
    positionThumb();
    plot.setActive(state.modelId);
    const running = state.running;
    ui.trainBtn.classList.toggle("paused", running);
    ui.trainBtn.querySelector(".txt").textContent = running ? "Pause" : "Train";
    ui.trainBtn.firstChild.outerHTML = running ? ICON_PAUSE : ICON_PLAY;
  }

  function setBadge(live) {
    ui.badge.className = "badge " + (live ? "live" : "preview");
    ui.badge.innerHTML = `<span class="led"></span>${live ? "live in-browser" : "preview"}`;
  }

  function setMsg(text, kind = "") {
    ui.msg.className = "msg" + (kind ? " " + kind : "");
    ui.msg.textContent = text || "";
  }

  function renderHud(target) {
    const b = state.backends.get(state.modelId);
    const upd = b ? b.updates : 0;
    subEl.textContent = `${labelOf(state.modelId)} · ${METRIC_LABEL} · Update ${upd}`;
    if (target == null) {
      bigEl.innerHTML = "&nbsp;";
      return;
    }
    // count-up toward target
    const cur = state.shown.get(state.modelId) ?? target;
    if (REDUCED || Math.abs(cur - target) < 0.004) {
      state.shown.set(state.modelId, target);
      bigEl.textContent = Math.round(target * 100) + "%";
    } else {
      const next = cur + (target - cur) * 0.3;
      state.shown.set(state.modelId, next);
      bigEl.textContent = Math.round(next * 100) + "%";
      requestAnimationFrame(() => renderHud(state.metrics.get(state.modelId)));
    }
  }

  async function backendFor(modelId) {
    if (state.backends.has(modelId)) return state.backends.get(modelId);
    let backend;
    try {
      backend = await WhloBackend.load(modelId);
    } catch (error) {
      console.warn(`[pomdp] ${modelId}: whlo backend unavailable, using illustrative preview`, error);
      backend = new MockBackend(modelId);
    }
    state.backends.set(modelId, backend);
    plot.ensure(modelId);
    return backend;
  }

  async function loop() {
    const backend = await backendFor(state.modelId);
    setBadge(backend.live);
    plot.setRunning(true);
    const perFrame = backend.live ? 1 : 2;
    while (state.running && backend.updates < MAX_UPDATES) {
      let metric;
      for (let i = 0; i < perFrame && backend.updates < MAX_UPDATES; i++) {
        metric = backend.step();
        plot.push(state.modelId, metric);
      }
      state.metrics.set(state.modelId, metric);
      renderHud(metric);
      await raf();
    }
    plot.setRunning(false);
    if (backend.updates >= MAX_UPDATES) stop();
  }

  function start() {
    if (state.running) return;
    state.running = true;
    paintSelector();
    loop();
  }
  function stop() {
    state.running = false;
    plot.setRunning(false);
    paintSelector();
  }

  async function selectModel(id) {
    stop();
    state.modelId = id;
    paintSelector();
    const backend = await backendFor(id);
    setBadge(backend.live);
    renderHud(state.metrics.get(id) ?? null);
  }

  // Compile an uploaded init.mlir + train_step.mlir with whlo, validate both the
  // init and one train step run against the (state, key) interface, then reveal the
  // "Yours" tab. Any layout/interface mismatch fails here, not mid-training.
  async function onUpload(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    setMsg("Compiling your policy with whlo…", "busy");
    let parsed;
    try {
      parsed = await Promise.all(files.map(async (f) => ({ name: f.name.toLowerCase(), text: await f.text() })));
    } catch (_) {
      setMsg("Could not read those files.", "err");
      return;
    }
    const mlir = parsed.filter((p) => p.name.endsWith(".mlir"));
    const init = mlir.find((p) => /init/.test(p.name));
    const step = mlir.find((p) => /train|step/.test(p.name)) || mlir.find((p) => p !== init);
    const layoutFile = parsed.find((p) => p.name.endsWith(".json"));
    if (!init || !step) {
      setMsg("Select both an init.mlir and a train_step.mlir.", "err");
      return;
    }
    let layout = null;
    try {
      if (layoutFile) layout = JSON.parse(layoutFile.text);
    } catch (_) {}
    try {
      const backend = await WhloBackend.fromSources(init.text, step.text, layout);
      backend.step(); // exercise train_step once so a bad interface throws now
      backend.reset(); // discard the trial rollout, start the tab clean
      state.backends.set("custom", backend);
      state.metrics.delete("custom");
      state.shown.delete("custom");
      plot.ensure("custom");
      plot.clear("custom");
      ui.customBtn.style.display = "";
      ui.customBtn.title = files.map((f) => f.name).join(", ");
      setMsg("");
      selectModel("custom");
    } catch (err) {
      console.error("[pomdp] upload compile/run failed:", err);
      setMsg("whlo could not compile that policy. It must export the (state, key) interface.", "err");
    }
  }

  ui.trainBtn.addEventListener("click", () => (state.running ? stop() : start()));
  ui.resetBtn.addEventListener("click", async () => {
    stop();
    const backend = await backendFor(state.modelId);
    backend.reset(state.modelId);
    state.metrics.delete(state.modelId);
    state.shown.delete(state.modelId);
    plot.clear(state.modelId);
    renderHud(null);
  });
  ui.modelBtns.forEach((b) => b.addEventListener("click", () => selectModel(b.dataset.id)));
  ui.uploadBtn.addEventListener("click", () => ui.fileInput.click());
  ui.fileInput.addEventListener("change", (e) => {
    onUpload(e.target.files);
    e.target.value = ""; // allow re-uploading the same filename
  });
  window.addEventListener("resize", positionThumb);

  paintSelector();
  const first = await backendFor(state.modelId);
  setBadge(first.live);
  renderHud(null);
  // thumb needs a layout pass before it can measure button geometry
  requestAnimationFrame(positionThumb);
}

main();
