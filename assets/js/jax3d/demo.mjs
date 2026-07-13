// jax3d car + track editor, in your browser. The physics `step` (exported from
// ~/Jax3D, compiled to WebAssembly by whlo) advances the full rigid-body state;
// render3d.mjs draws it. In Build mode you place fixated boxes (a pool exported
// with the scene) to make a track; in Drive mode the car drives what you built.

import { createRenderer } from "./render3d.mjs";

const WHLO = "./whlo/src/index.mjs";
const ASSET = (f) => `/assets/jax3d/${f}`;
const TA = { float32: Float32Array, int32: Int32Array, uint32: Uint32Array, bool: Uint8Array };
const toTA = (s) => (s.dtype === "bool" ? Uint8Array.from(s.data.map(Number)) : TA[s.dtype].from(s.data));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const DEG = Math.PI / 180;

function call(exe, inputs) {
  const named = {};
  exe.inputs.forEach((slot, i) => (named[slot.name] = inputs[i]));
  const out = exe.run(named);
  return exe.outputs.map((slot) => out[slot.name]);
}

// --- vec/quat helpers (quat is [x,y,z,w]) ---
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => {
  const l = Math.hypot(...a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const qAxis = (ax, ang) => {
  const s = Math.sin(ang / 2);
  return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(ang / 2)];
};
const qMul = (a, b) => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];

// piece types: half-extents, base tilt about y, colour
const PIECES = {
  ramp: { he: [1.5, 1.4, 0.05], tilt: 18 * DEG, color: [0.56, 0.58, 0.62] },
  platform: { he: [1.3, 1.3, 0.15], tilt: 0, color: [0.5, 0.52, 0.57] },
  wall: { he: [1.4, 0.13, 0.55], tilt: 0, color: [0.74, 0.46, 0.43] },
};
const restZ = (p) => PIECES[p.type].he[0] * Math.sin(PIECES[p.type].tilt) + PIECES[p.type].he[2] * Math.cos(PIECES[p.type].tilt);
const pieceQuat = (p) => qMul(qAxis([0, 0, 1], p.yaw), qAxis([0, 1, 0], -PIECES[p.type].tilt));

const CSS = `
#jax3d-app {
  --accent: #3b82f6; --accent-soft: rgba(59,130,246,.14); --line: rgba(127,127,127,.3);
  --btn-h: 2.125rem; --radius: 8px; --font: .82rem;
  /* spacing scale (~golden ratio: 4 · 6 · 10 · 16) */
  --sp-1: 4px; --sp-2: 6px; --sp-3: 10px; --sp-4: 16px;
  max-width: 48rem; margin: 1.5rem auto;
}
/* the site's global button styles (uppercase text, drop shadow, margin) are meant for
   Bootstrap/MDB .btn elements — reset them so our controls aren't fighting that theme */
#jax3d-app button { box-sizing: border-box; margin: 0; text-transform: none; letter-spacing: normal; box-shadow: none; }
#jax3d-app .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--sp-4); margin-bottom: var(--sp-3); }

/* mode toggle: Build / Drive labels flank a sliding switch (like a settings toggle) */
#jax3d-app .toggle { display: inline-flex; align-items: center; gap: .6rem; }
#jax3d-app .toggle .lbl { font: inherit; font-size: var(--font); font-weight: 600; border: 0; background: transparent; padding: 0; color: inherit; cursor: pointer; opacity: .5; transition: opacity .2s ease, color .2s ease; }
#jax3d-app .toggle .lbl.on { opacity: 1; color: var(--accent); }
#jax3d-app .toggle .track { position: relative; width: 2.75rem; height: 1.5rem; padding: 0; border: 1px solid var(--line); border-radius: 999px; background: rgba(127,127,127,.16); cursor: pointer; transition: background .2s ease, border-color .2s ease; }
#jax3d-app .toggle[data-mode="drive"] .track { background: var(--accent-soft); border-color: var(--accent); }
#jax3d-app .toggle .knob { position: absolute; top: .125rem; left: .125rem; width: 1.25rem; height: 1.25rem; border-radius: 50%; background: var(--accent); box-shadow: 0 1px 3px rgba(0,0,0,.3); transition: transform .22s cubic-bezier(.32,.72,0,1); }
#jax3d-app .toggle[data-mode="drive"] .knob { transform: translateX(1.25rem); }

/* tool groups: tight, related controls sit close together */
#jax3d-app .tools { display: flex; gap: var(--sp-3); align-items: center; flex-wrap: wrap; }
#jax3d-app .tools.hidden { display: none; }

/* joined segmented group (piece palette) — buttons share edges, no gaps */
#jax3d-app .group { display: inline-flex; border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
#jax3d-app .group button { height: var(--btn-h); width: var(--btn-h); padding: 0; border: 0; border-left: 1px solid var(--line); background: transparent; color: inherit; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; transition: background .15s ease, color .15s ease; }
#jax3d-app .group button svg { width: 1.15rem; height: 1.15rem; }
#jax3d-app .group button:first-child { border-left: 0; }
#jax3d-app .group button:hover { background: rgba(127,127,127,.1); }
#jax3d-app .group button.on { background: var(--accent-soft); color: var(--accent); }

/* standalone button (Clear, Reset) — same height/font as everything else */
#jax3d-app .btn { font: inherit; font-size: var(--font); font-weight: 600; height: var(--btn-h); padding: 0 1rem; border: 1px solid var(--line); border-radius: var(--radius); background: transparent; color: inherit; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; gap: .35rem; transition: background .15s ease, border-color .15s ease, color .15s ease; }
#jax3d-app .btn:hover { background: rgba(127,127,127,.1); border-color: rgba(127,127,127,.5); }
#jax3d-app .btn:active { background: rgba(127,127,127,.16); }
#jax3d-app .btn.danger:hover { border-color: #ef4444; color: #ef4444; background: rgba(239,68,68,.09); }

/* stage + canvas */
#jax3d-app .stage { position: relative; width: 100%; aspect-ratio: 16 / 10; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.25), 0 0 0 1px rgba(127,127,127,.25); background: #171a1f; }
#jax3d-app canvas { display: block; width: 100%; height: 100%; outline: none; touch-action: none; }
#jax3d-app .overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #cbd2da; font-weight: 600; background: rgba(20,22,26,.6); transition: opacity .3s; }

/* floating inspector for the selected piece */
#jax3d-app .inspector { position: absolute; left: .7rem; top: .7rem; display: flex; align-items: center; gap: .55rem; padding: .4rem .5rem; border-radius: 10px; background: rgba(22,25,31,.82); -webkit-backdrop-filter: blur(7px); backdrop-filter: blur(7px); box-shadow: 0 3px 14px rgba(0,0,0,.4), inset 0 0 0 1px rgba(255,255,255,.07); color: #e6e9ee; font-size: .8rem; }
#jax3d-app .inspector.hidden { display: none; }
#jax3d-app .inspector .lbl { opacity: .66; font-weight: 600; letter-spacing: .01em; }
#jax3d-app .inspector .val { min-width: 2.3rem; text-align: right; font-variant-numeric: tabular-nums; opacity: .95; }
#jax3d-app .inspector .sep { width: 1px; height: 1.3rem; background: rgba(255,255,255,.15); }
#jax3d-app .inspector .ibtn { width: 1.75rem; height: 1.75rem; padding: 0; display: inline-flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,.16); border-radius: 7px; background: rgba(255,255,255,.06); color: inherit; cursor: pointer; transition: background .12s ease, border-color .12s ease, color .12s ease; }
#jax3d-app .inspector .ibtn:hover { background: rgba(255,255,255,.15); }
#jax3d-app .inspector .ibtn svg { width: .95rem; height: .95rem; }
#jax3d-app .inspector .ibtn.del:hover { border-color: #f87171; color: #f87171; background: rgba(239,68,68,.16); }
#jax3d-app .inspector input[type=range] { -webkit-appearance: none; appearance: none; width: 6rem; height: 4px; border-radius: 999px; background: rgba(255,255,255,.22); outline: none; cursor: pointer; }
#jax3d-app .inspector input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 14px; height: 14px; border-radius: 50%; background: var(--accent); box-shadow: 0 1px 3px rgba(0,0,0,.4); }
#jax3d-app .inspector input[type=range]::-moz-range-thumb { width: 14px; height: 14px; border: 0; border-radius: 50%; background: var(--accent); }

#jax3d-app .hint { margin-top: .6rem; font-size: .8rem; opacity: .6; min-height: 1.2em; }
#jax3d-app kbd { font: inherit; font-size: .76rem; padding: .04rem .32rem; border: 1px solid rgba(127,127,127,.4); border-bottom-width: 2px; border-radius: 5px; }
`;

const el = (t, c) => {
  const e = document.createElement(t);
  if (c) e.className = c;
  return e;
};

// inline icons (feather-style, stroke = currentColor)
const svg = (p) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
// isometric "box" icons — same projection as the pieces they place, so the
// palette reads like tiny 3D previews rather than abstract glyphs
const ICON = {
  ramp: svg('<path d="M4 20 20 20 20 6Z"/><path d="M20 20 20 6 22 3 22 17Z"/>'),
  platform: svg('<path d="M12 4 20 8 12 12 4 8Z"/><path d="M4 8 4 14 12 18 12 12Z"/><path d="M12 12 12 18 20 14 20 8Z"/>'),
  wall: svg('<path d="M12 2 20 6 12 10 4 6Z"/><path d="M4 6 4 19 12 23 12 10Z"/><path d="M12 10 12 23 20 19 20 6Z"/>'),
  rotL: svg('<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>'),
  rotR: svg('<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>'),
  trash: svg('<path d="M4 7h16"/><path d="M9 7V5h6v2"/><path d="M6 7l1 13h10l1-13"/>'),
};

async function main() {
  const root = document.getElementById("jax3d-app");
  if (!root) return;
  const style = el("style");
  style.textContent = CSS;
  root.appendChild(style);

  // --- UI ---
  const toolbar = el("div", "toolbar");

  // primary mode toggle: Build / Drive labels flank a switch you click to slide between them
  const modeToggle = el("div", "toggle");
  modeToggle.dataset.mode = "build";
  const buildLbl = el("button", "lbl on");
  buildLbl.type = "button";
  buildLbl.textContent = "Build";
  const track = el("button", "track");
  track.type = "button";
  track.setAttribute("aria-label", "Switch between build and drive mode");
  const knob = el("div", "knob");
  track.appendChild(knob);
  const driveLbl = el("button", "lbl");
  driveLbl.type = "button";
  driveLbl.textContent = "Drive";
  modeToggle.append(buildLbl, track, driveLbl);

  // build mode: piece palette (one joined segmented group) + clear
  const buildTools = el("div", "tools");
  const paletteGroup = el("div", "group");
  const palette = {};
  for (const t of ["ramp", "platform", "wall"]) {
    const b = el("button");
    b.innerHTML = ICON[t];
    b.dataset.t = t;
    b.title = t[0].toUpperCase() + t.slice(1);
    b.setAttribute("aria-label", b.title);
    palette[t] = b;
    paletteGroup.appendChild(b);
  }
  const clearBtn = el("button", "btn danger");
  clearBtn.textContent = "Clear";
  buildTools.append(paletteGroup, clearBtn);

  // drive mode: reset
  const driveTools = el("div", "tools hidden");
  const resetBtn = el("button", "btn");
  resetBtn.textContent = "Reset car";
  driveTools.appendChild(resetBtn);

  toolbar.append(modeToggle, buildTools, driveTools);

  const stage = el("div", "stage");
  const canvas = el("canvas");
  canvas.tabIndex = 0;
  const overlay = el("div", "overlay");
  overlay.textContent = "Compiling the physics…";

  // floating inspector for the selected piece (rotate / height / delete)
  const inspector = el("div", "inspector hidden");
  const rotLBtn = el("button", "ibtn");
  rotLBtn.innerHTML = ICON.rotL;
  rotLBtn.title = "Rotate left (Q)";
  const rotRBtn = el("button", "ibtn");
  rotRBtn.innerHTML = ICON.rotR;
  rotRBtn.title = "Rotate right (E)";
  const heightLbl = el("span", "lbl");
  heightLbl.textContent = "Height";
  const heightSlider = el("input");
  heightSlider.type = "range";
  heightSlider.min = "0";
  heightSlider.max = "4";
  heightSlider.step = "0.1";
  heightSlider.value = "0";
  heightSlider.title = "Raise / lower (↑ / ↓)";
  const heightVal = el("span", "val");
  heightVal.textContent = "0.0";
  const delBtn = el("button", "ibtn del");
  delBtn.innerHTML = ICON.trash;
  delBtn.title = "Delete (Del)";
  inspector.append(rotLBtn, rotRBtn, el("span", "sep"), heightLbl, heightSlider, heightVal, el("span", "sep"), delBtn);

  stage.append(canvas, overlay, inspector);
  const hint = el("div", "hint");
  root.append(toolbar, stage, hint);

  let renderer;
  try {
    renderer = createRenderer(canvas);
  } catch (e) {
    overlay.textContent = "WebGL isn't available in this browser.";
    console.error(e);
    return;
  }

  let step, init, scene;
  try {
    const { compile, initCompiler } = await import(WHLO);
    await initCompiler();
    const [mlir, i, sc] = await Promise.all([
      fetch(ASSET("step.mlir")).then((r) => r.text()),
      fetch(ASSET("init_state.json")).then((r) => r.json()),
      fetch(ASSET("scene.json")).then((r) => r.json()),
    ]);
    step = await compile(mlir);
    init = i;
    scene = sc;
  } catch (e) {
    overlay.textContent = "Couldn't load the physics model.";
    console.error(e);
    return;
  }

  const L = scene.leaf,
    D = scene.drive,
    CI = scene.chassis_index;

  // --- track / state model ---
  // fresh state with every editable box cleared (just floor + car)
  const freshState = () => {
    const s = init.map(toTA);
    for (const b of scene.editable_boxes) s[L.box_active][b] = 0;
    return s;
  };
  const pieces = new Map(); // box index -> {type, x, y, yaw, elev}
  const free = () => scene.editable_boxes.find((b) => !pieces.has(b));
  const ELEV_MAX = 4; // metres a piece can be lifted off the ground

  let state = freshState();
  const syncPiece = (b) => {
    const p = pieces.get(b);
    state[L.box_active][b] = 1;
    state[L.box_position].set([p.x, p.y, restZ(p) + p.elev], b * 3);
    state[L.box_half_extents].set(PIECES[p.type].he, b * 3);
    state[L.box_quat].set(pieceQuat(p), b * 4);
  };
  const placePiece = (type, x, y) => {
    const b = free();
    if (b === undefined) return null;
    pieces.set(b, { type, x, y, yaw: 0, elev: 0 });
    syncPiece(b);
    return b;
  };
  const removePiece = (b) => {
    pieces.delete(b);
    state[L.box_active][b] = 0;
  };
  const resetCar = () => {
    const kept = new Map(pieces);
    state = freshState();
    for (const [b, p] of kept) {
      pieces.set(b, p);
      syncPiece(b);
    }
  };

  // a small starter so it isn't empty
  placePiece("ramp", -1.5, 0);
  placePiece("platform", 1.6, 0);

  // --- camera ---
  let { azimuth, elevation, distance } = scene.camera;
  let target = [...scene.camera.target];
  const eye = () => {
    const ce = Math.cos(elevation),
      se = Math.sin(elevation),
      ca = Math.cos(azimuth),
      sa = Math.sin(azimuth);
    return [target[0] + distance * ce * ca, target[1] + distance * ce * sa, target[2] + distance * se];
  };
  const fov = scene.camera.fov_deg;

  // ground raycast from a screen point
  function screenToGround(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    const ndcx = ((clientX - r.left) / r.width) * 2 - 1;
    const ndcy = -(((clientY - r.top) / r.height) * 2 - 1);
    const e = eye();
    const fwd = norm([target[0] - e[0], target[1] - e[1], target[2] - e[2]]);
    const right = norm(cross(fwd, [0, 0, 1]));
    const up = cross(right, fwd);
    const tanF = Math.tan((fov * DEG) / 2),
      aspect = r.width / r.height;
    const dir = norm([
      fwd[0] + ndcx * tanF * aspect * right[0] + ndcy * tanF * up[0],
      fwd[1] + ndcx * tanF * aspect * right[1] + ndcy * tanF * up[1],
      fwd[2] + ndcx * tanF * aspect * right[2] + ndcy * tanF * up[2],
    ]);
    if (Math.abs(dir[2]) < 1e-6) return null;
    const t = -e[2] / dir[2];
    if (t < 0) return null;
    return [e[0] + t * dir[0], e[1] + t * dir[1]];
  }
  const pickPiece = (gx, gy) => {
    let best = null,
      bestD = Infinity;
    for (const [b, p] of pieces) {
      const rad = Math.max(PIECES[p.type].he[0], PIECES[p.type].he[1]) + 0.3;
      const d = Math.hypot(p.x - gx, p.y - gy);
      if (d < rad && d < bestD) {
        bestD = d;
        best = b;
      }
    }
    return best;
  };

  // --- interaction ---
  let mode = "build";
  let armed = null; // palette type armed for placing
  let selected = null; // selected placed box
  let down = null; // pointer-down info

  // Selected-piece inspector (rotate / height / delete). Shown only while a
  // placed piece is selected in Build mode.
  const refreshInspector = () => {
    const show = mode === "build" && selected !== null && pieces.has(selected);
    inspector.classList.toggle("hidden", !show);
    if (show) {
      const p = pieces.get(selected);
      heightSlider.value = String(p.elev);
      heightVal.textContent = p.elev.toFixed(1);
    }
  };
  const select = (b) => {
    selected = b;
    refreshInspector();
  };
  const rotateSel = (dir) => {
    if (selected === null) return;
    pieces.get(selected).yaw += dir * 12 * DEG;
    syncPiece(selected);
  };
  const setElev = (v) => {
    if (selected === null) return;
    const p = pieces.get(selected);
    p.elev = clamp(v, 0, ELEV_MAX);
    syncPiece(selected);
    refreshInspector();
  };

  rotLBtn.onclick = () => rotateSel(1);
  rotRBtn.onclick = () => rotateSel(-1);
  delBtn.onclick = () => {
    if (selected !== null) {
      removePiece(selected);
      select(null);
    }
  };
  heightSlider.addEventListener("input", () => {
    if (selected === null) return;
    const p = pieces.get(selected);
    p.elev = Number(heightSlider.value);
    heightVal.textContent = p.elev.toFixed(1);
    syncPiece(selected);
  });

  const setMode = (m) => {
    mode = m;
    modeToggle.dataset.mode = m; // slides the toggle knob
    buildLbl.classList.toggle("on", m === "build");
    driveLbl.classList.toggle("on", m === "drive");
    buildTools.classList.toggle("hidden", m !== "build");
    driveTools.classList.toggle("hidden", m === "build");
    select(null);
    resetCar();
    hint.innerHTML =
      m === "build"
        ? "Pick a piece and click the ground to place it. Click a piece to select, drag to move; then <kbd>Q</kbd>/<kbd>E</kbd> rotate, <kbd>↑</kbd>/<kbd>↓</kbd> height, <kbd>Del</kbd> remove. Right-drag to orbit."
        : "<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to drive. Drag to orbit, scroll to zoom.";
  };
  buildLbl.onclick = () => setMode("build");
  driveLbl.onclick = () => setMode("drive");
  track.onclick = () => setMode(mode === "build" ? "drive" : "build");
  for (const t in palette)
    palette[t].onclick = () => {
      armed = armed === t ? null : t;
      select(null);
      for (const k in palette) palette[k].classList.toggle("on", k === armed);
    };
  clearBtn.onclick = () => {
    for (const b of [...pieces.keys()]) removePiece(b);
    select(null);
  };
  resetBtn.onclick = () => resetCar();

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    canvas.focus();
    down = { x: e.clientX, y: e.clientY, moved: false, orbit: e.button !== 0 || mode === "drive", dragBox: null };
    if (mode === "build" && e.button === 0) {
      const g = screenToGround(e.clientX, e.clientY);
      if (g) {
        const hit = pickPiece(g[0], g[1]);
        if (armed && hit === null) select(placePiece(armed, g[0], g[1]));
        else {
          select(hit);
          down.dragBox = hit;
        }
      }
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!down) return;
    const dx = e.clientX - down.x,
      dy = e.clientY - down.y;
    if (!down.moved && Math.hypot(dx, dy) > 3) down.moved = true;
    if (down.dragBox !== null && !down.orbit) {
      const g = screenToGround(e.clientX, e.clientY);
      if (g) {
        const p = pieces.get(down.dragBox);
        p.x = g[0];
        p.y = g[1];
        syncPiece(down.dragBox);
      }
    } else if (down.moved) {
      azimuth -= e.movementX * 0.01;
      elevation = clamp(elevation + e.movementY * 0.01, -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
    }
    down.x = e.clientX;
    down.y = e.clientY;
  });
  canvas.addEventListener("pointerup", (e) => {
    canvas.releasePointerCapture(e.pointerId);
    down = null;
  });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      distance = clamp(distance * (1 + Math.sign(e.deltaY) * 0.1), 3, 60);
    },
    { passive: false }
  );

  // keys: driving (WASD) + editing (Q/E rotate, Del remove)
  const keys = {};
  const KMAP = { w: "w", a: "a", s: "s", d: "d", arrowup: "w", arrowleft: "a", arrowdown: "s", arrowright: "d" };
  let hovering = false;
  stage.addEventListener("pointerenter", () => (hovering = true));
  stage.addEventListener("pointerleave", () => (hovering = false));
  const active = () => hovering || document.activeElement === canvas;
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return; // let the height slider handle its own keys
    if (!active()) return;
    const k = e.key.toLowerCase();
    if (mode === "drive" && KMAP[k]) {
      keys[KMAP[k]] = true;
      e.preventDefault();
      return;
    }
    if (mode === "build" && selected !== null) {
      if (k === "q" || k === "e") {
        rotateSel(k === "q" ? 1 : -1);
        e.preventDefault();
      } else if (k === "arrowup" || k === "arrowdown") {
        setElev(pieces.get(selected).elev + (k === "arrowup" ? 0.1 : -0.1));
        e.preventDefault();
      } else if (k === "delete" || k === "backspace") {
        removePiece(selected);
        select(null);
        e.preventDefault();
      }
    }
  });
  window.addEventListener("keyup", (e) => {
    const k = KMAP[e.key.toLowerCase()];
    if (k) keys[k] = false;
  });
  window.addEventListener("blur", () => {
    for (const k in keys) keys[k] = false;
  });

  // --- render / sim loop ---
  function drawItems() {
    const items = [];
    const bp = state[L.box_position],
      bq = state[L.box_quat],
      bh = state[L.box_half_extents],
      ba = state[L.box_active];
    for (let i = 0; i < scene.num_boxes; i++) {
      if (!ba[i]) continue;
      const piece = pieces.get(i);
      let color = i === scene.floor_index ? [0.14, 0.16, 0.19] : piece ? PIECES[piece.type].color : scene.box_colors[i] || [0.55, 0.57, 0.6];
      if (i === CI) color = scene.box_colors[CI] || [0.82, 0.24, 0.24];
      if (i === selected) color = [0.23, 0.55, 1.0];
      items.push({
        geom: "cube",
        pos: [bp[i * 3], bp[i * 3 + 1], bp[i * 3 + 2]],
        quat: [bq[i * 4], bq[i * 4 + 1], bq[i * 4 + 2], bq[i * 4 + 3]],
        scale: [bh[i * 3], bh[i * 3 + 1], bh[i * 3 + 2]],
        color,
      });
    }
    const sp = state[L.sphere_position],
      sq = state[L.sphere_quat],
      sr = state[L.sphere_radius],
      sa = state[L.sphere_active];
    for (let i = 0; i < scene.num_spheres; i++) {
      if (!sa[i]) continue;
      const r = sr[i];
      items.push({
        geom: "sphere",
        pos: [sp[i * 3], sp[i * 3 + 1], sp[i * 3 + 2]],
        quat: [sq[i * 4], sq[i * 4 + 1], sq[i * 4 + 2], sq[i * 4 + 3]],
        scale: [r, r, r],
        color: scene.sphere_color,
      });
    }
    return items;
  }

  overlay.style.opacity = "0";
  setTimeout(() => overlay.remove(), 400);
  setMode("build");

  // Self-running showcase for recording (gated so it never affects normal use).
  if (new URLSearchParams(location.search).has("auto")) {
    for (const b of [...pieces.keys()]) removePiece(b);
    const T = [
      [300, () => placePiece("ramp", -2.3, 0)],
      [850, () => placePiece("platform", 0.5, 0)],
      [1400, () => placePiece("ramp", 3.3, 0)],
      [1950, () => placePiece("wall", 0.5, 2.1)],
      [2450, () => placePiece("wall", 0.5, -2.1)],
      [3050, () => select([...pieces.keys()][2])],
      [
        3450,
        () => {
          const b = [...pieces.keys()][2];
          pieces.get(b).yaw += 22 * DEG;
          syncPiece(b);
        },
      ],
      [4100, () => setMode("drive")],
      [4600, () => (keys.w = true)],
      [6600, () => (keys.a = true)],
      [7600, () => (keys.a = false)],
      [8800, () => (keys.d = true)],
      [9800, () => (keys.d = false)],
      [11200, () => (keys.w = false)],
    ];
    for (const [t, fn] of T) setTimeout(fn, t);
    window.__auto = 1;
  }

  function frame() {
    if (mode === "drive") {
      const forward = ((keys.w ? 1 : 0) - (keys.s ? 1 : 0)) * (D.throttle ?? 1);
      const turn = (keys.a ? 1 : 0) - (keys.d ? 1 : 0);
      const left = forward - D.turn_strength * turn,
        right = forward + D.turn_strength * turn;
      const actions = new Float32Array(scene.num_joints);
      for (const i of D.left_wheels) actions[i] = left;
      for (const i of D.right_wheels) actions[i] = right;
      const targetYaw = turn * D.max_yaw_rate;
      for (let s = 0; s < D.steps_per_frame; s++) {
        state = call(step, [...state, actions]);
        const av = state[L.box_angular_velocity],
          j = CI * 3 + 2;
        av[j] = av[j] + (targetYaw - av[j]) * D.yaw_smoothing;
      }
      const bp = state[L.box_position];
      const cp = [bp[CI * 3], bp[CI * 3 + 1], bp[CI * 3 + 2]];
      for (let i = 0; i < 3; i++) target[i] += (cp[i] - target[i]) * 0.2;
    }
    renderer.render(drawItems(), { eye: eye(), target, fovDeg: fov });
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
