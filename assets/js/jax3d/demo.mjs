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
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const qAxis = (ax, ang) => { const s = Math.sin(ang / 2); return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(ang / 2)]; };
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
#jax3d-app { --accent: #3b82f6; max-width: 48rem; margin: 1.5rem auto; }
#jax3d-app .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin-bottom: .7rem; }
#jax3d-app .seg { display: inline-flex; border: 1px solid rgba(127,127,127,.32); border-radius: 999px; overflow: hidden; }
#jax3d-app .seg button { font: inherit; font-weight: 600; padding: .35rem .9rem; border: 0; background: transparent; color: inherit; cursor: pointer; }
#jax3d-app .seg button.on { background: var(--accent); color: #fff; }
#jax3d-app .tools { display: flex; gap: .4rem; align-items: center; }
#jax3d-app .tools.hidden { display: none; }
#jax3d-app .chip { font: inherit; font-size: .85rem; padding: .32rem .7rem; border: 1px solid rgba(127,127,127,.32); border-radius: 8px; background: transparent; color: inherit; cursor: pointer; }
#jax3d-app .chip.on { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); color: var(--accent); }
#jax3d-app .stage { position: relative; width: 100%; aspect-ratio: 16 / 10; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.25), 0 0 0 1px rgba(127,127,127,.25); background: #171a1f; }
#jax3d-app canvas { display: block; width: 100%; height: 100%; outline: none; touch-action: none; }
#jax3d-app .overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #cbd2da; font-weight: 600; background: rgba(20,22,26,.6); transition: opacity .3s; }
#jax3d-app .hint { margin-top: .6rem; font-size: .8rem; opacity: .6; min-height: 1.2em; }
#jax3d-app kbd { font: inherit; font-size: .76rem; padding: .04rem .32rem; border: 1px solid rgba(127,127,127,.4); border-bottom-width: 2px; border-radius: 5px; }
`;

const el = (t, c) => { const e = document.createElement(t); if (c) e.className = c; return e; };

async function main() {
  const root = document.getElementById("jax3d-app");
  if (!root) return;
  const style = el("style"); style.textContent = CSS; root.appendChild(style);

  // --- UI ---
  const toolbar = el("div", "toolbar");
  const seg = el("div", "seg");
  const buildBtn = el("button"); buildBtn.textContent = "Build";
  const driveBtn = el("button"); driveBtn.textContent = "Drive";
  seg.append(buildBtn, driveBtn);
  const buildTools = el("div", "tools");
  const palette = {};
  for (const t of ["ramp", "platform", "wall"]) {
    const b = el("button", "chip"); b.textContent = t[0].toUpperCase() + t.slice(1); b.dataset.t = t;
    palette[t] = b; buildTools.appendChild(b);
  }
  const clearBtn = el("button", "chip"); clearBtn.textContent = "Clear";
  buildTools.appendChild(clearBtn);
  const driveTools = el("div", "tools hidden");
  const resetBtn = el("button", "chip"); resetBtn.textContent = "Reset car";
  driveTools.appendChild(resetBtn);
  toolbar.append(seg, buildTools, driveTools);

  const stage = el("div", "stage");
  const canvas = el("canvas"); canvas.tabIndex = 0;
  const overlay = el("div", "overlay"); overlay.textContent = "Compiling the physics…";
  stage.append(canvas, overlay);
  const hint = el("div", "hint");
  root.append(toolbar, stage, hint);

  let renderer;
  try { renderer = createRenderer(canvas); }
  catch (e) { overlay.textContent = "WebGL isn't available in this browser."; console.error(e); return; }

  let step, init, scene;
  try {
    const { compile, initCompiler } = await import(WHLO);
    await initCompiler();
    const [mlir, i, sc] = await Promise.all([
      fetch(ASSET("step.mlir")).then((r) => r.text()),
      fetch(ASSET("init_state.json")).then((r) => r.json()),
      fetch(ASSET("scene.json")).then((r) => r.json()),
    ]);
    step = await compile(mlir); init = i; scene = sc;
  } catch (e) { overlay.textContent = "Couldn't load the physics model."; console.error(e); return; }

  const L = scene.leaf, D = scene.drive, CI = scene.chassis_index;

  // --- track / state model ---
  // fresh state with every editable box cleared (just floor + car)
  const freshState = () => {
    const s = init.map(toTA);
    for (const b of scene.editable_boxes) s[L.box_active][b] = 0;
    return s;
  };
  const pieces = new Map(); // box index -> {type, x, y, yaw}
  const free = () => scene.editable_boxes.find((b) => !pieces.has(b));

  let state = freshState();
  const syncPiece = (b) => {
    const p = pieces.get(b);
    state[L.box_active][b] = 1;
    state[L.box_position].set([p.x, p.y, restZ(p)], b * 3);
    state[L.box_half_extents].set(PIECES[p.type].he, b * 3);
    state[L.box_quat].set(pieceQuat(p), b * 4);
  };
  const placePiece = (type, x, y) => {
    const b = free(); if (b === undefined) return null;
    pieces.set(b, { type, x, y, yaw: 0 }); syncPiece(b); return b;
  };
  const removePiece = (b) => { pieces.delete(b); state[L.box_active][b] = 0; };
  const resetCar = () => { const kept = new Map(pieces); state = freshState(); for (const [b, p] of kept) { pieces.set(b, p); syncPiece(b); } };

  // a small starter so it isn't empty
  placePiece("ramp", -1.5, 0); placePiece("platform", 1.6, 0);

  // --- camera ---
  let { azimuth, elevation, distance } = scene.camera;
  let target = [...scene.camera.target];
  const eye = () => {
    const ce = Math.cos(elevation), se = Math.sin(elevation), ca = Math.cos(azimuth), sa = Math.sin(azimuth);
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
    const tanF = Math.tan((fov * DEG) / 2), aspect = r.width / r.height;
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
    let best = null, bestD = Infinity;
    for (const [b, p] of pieces) {
      const rad = Math.max(PIECES[p.type].he[0], PIECES[p.type].he[1]) + 0.3;
      const d = Math.hypot(p.x - gx, p.y - gy);
      if (d < rad && d < bestD) { bestD = d; best = b; }
    }
    return best;
  };

  // --- interaction ---
  let mode = "build";
  let armed = null;     // palette type armed for placing
  let selected = null;  // selected placed box
  let down = null;      // pointer-down info

  const setMode = (m) => {
    mode = m;
    buildBtn.classList.toggle("on", m === "build");
    driveBtn.classList.toggle("on", m === "drive");
    buildTools.classList.toggle("hidden", m !== "build");
    driveTools.classList.toggle("hidden", m === "build");
    selected = null;
    resetCar();
    hint.innerHTML = m === "build"
      ? "Pick a piece and click the ground to place it. Click a piece to select, drag to move, <kbd>Q</kbd>/<kbd>E</kbd> to rotate, <kbd>Del</kbd> to remove. Right-drag to orbit."
      : "<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to drive. Drag to orbit, scroll to zoom.";
  };
  buildBtn.onclick = () => setMode("build");
  driveBtn.onclick = () => setMode("drive");
  for (const t in palette) palette[t].onclick = () => {
    armed = armed === t ? null : t; selected = null;
    for (const k in palette) palette[k].classList.toggle("on", k === armed);
  };
  clearBtn.onclick = () => { for (const b of [...pieces.keys()]) removePiece(b); selected = null; };
  resetBtn.onclick = () => resetCar();

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId); canvas.focus();
    down = { x: e.clientX, y: e.clientY, moved: false, orbit: e.button !== 0 || mode === "drive", dragBox: null };
    if (mode === "build" && e.button === 0) {
      const g = screenToGround(e.clientX, e.clientY);
      if (g) {
        const hit = pickPiece(g[0], g[1]);
        if (armed && hit === null) { const b = placePiece(armed, g[0], g[1]); selected = b; }
        else { selected = hit; down.dragBox = hit; }
      }
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    if (!down.moved && Math.hypot(dx, dy) > 3) down.moved = true;
    if (down.dragBox !== null && !down.orbit) {
      const g = screenToGround(e.clientX, e.clientY);
      if (g) { const p = pieces.get(down.dragBox); p.x = g[0]; p.y = g[1]; syncPiece(down.dragBox); }
    } else if (down.moved) {
      azimuth -= e.movementX * 0.01;
      elevation = clamp(elevation + e.movementY * 0.01, -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
    }
    down.x = e.clientX; down.y = e.clientY;
  });
  canvas.addEventListener("pointerup", (e) => { canvas.releasePointerCapture(e.pointerId); down = null; });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); distance = clamp(distance * (1 + Math.sign(e.deltaY) * 0.1), 3, 60); }, { passive: false });

  // keys: driving (WASD) + editing (Q/E rotate, Del remove)
  const keys = {};
  const KMAP = { w: "w", a: "a", s: "s", d: "d", arrowup: "w", arrowleft: "a", arrowdown: "s", arrowright: "d" };
  let hovering = false;
  stage.addEventListener("pointerenter", () => (hovering = true));
  stage.addEventListener("pointerleave", () => (hovering = false));
  const active = () => hovering || document.activeElement === canvas;
  window.addEventListener("keydown", (e) => {
    if (!active()) return;
    const k = e.key.toLowerCase();
    if (mode === "drive" && KMAP[k]) { keys[KMAP[k]] = true; e.preventDefault(); return; }
    if (mode === "build" && selected !== null) {
      if (k === "q" || k === "e") { const p = pieces.get(selected); p.yaw += (k === "q" ? 1 : -1) * 12 * DEG; syncPiece(selected); e.preventDefault(); }
      if (k === "delete" || k === "backspace") { removePiece(selected); selected = null; e.preventDefault(); }
    }
  });
  window.addEventListener("keyup", (e) => { const k = KMAP[e.key.toLowerCase()]; if (k) keys[k] = false; });
  window.addEventListener("blur", () => { for (const k in keys) keys[k] = false; });

  // --- render / sim loop ---
  function drawItems() {
    const items = [];
    const bp = state[L.box_position], bq = state[L.box_quat], bh = state[L.box_half_extents], ba = state[L.box_active];
    for (let i = 0; i < scene.num_boxes; i++) {
      if (!ba[i]) continue;
      const piece = pieces.get(i);
      let color = i === scene.floor_index ? [0.14, 0.16, 0.19] : piece ? PIECES[piece.type].color : scene.box_colors[i] || [0.55, 0.57, 0.6];
      if (i === CI) color = scene.box_colors[CI] || [0.82, 0.24, 0.24];
      if (i === selected) color = [0.23, 0.55, 1.0];
      items.push({ geom: "cube", pos: [bp[i * 3], bp[i * 3 + 1], bp[i * 3 + 2]],
        quat: [bq[i * 4], bq[i * 4 + 1], bq[i * 4 + 2], bq[i * 4 + 3]],
        scale: [bh[i * 3], bh[i * 3 + 1], bh[i * 3 + 2]], color });
    }
    const sp = state[L.sphere_position], sq = state[L.sphere_quat], sr = state[L.sphere_radius], sa = state[L.sphere_active];
    for (let i = 0; i < scene.num_spheres; i++) {
      if (!sa[i]) continue;
      const r = sr[i];
      items.push({ geom: "sphere", pos: [sp[i * 3], sp[i * 3 + 1], sp[i * 3 + 2]],
        quat: [sq[i * 4], sq[i * 4 + 1], sq[i * 4 + 2], sq[i * 4 + 3]], scale: [r, r, r], color: scene.sphere_color });
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
      [3050, () => { selected = [...pieces.keys()][2]; }],
      [3450, () => { const b = [...pieces.keys()][2]; const p = pieces.get(b); p.yaw += 22 * DEG; syncPiece(b); }],
      [4100, () => { selected = null; setMode("drive"); }],
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
      const forward = (keys.w ? 1 : 0) - (keys.s ? 1 : 0);
      const turn = (keys.a ? 1 : 0) - (keys.d ? 1 : 0);
      const left = forward - D.turn_strength * turn, right = forward + D.turn_strength * turn;
      const actions = new Float32Array(scene.num_joints);
      for (const i of D.left_wheels) actions[i] = left;
      for (const i of D.right_wheels) actions[i] = right;
      const targetYaw = turn * D.max_yaw_rate;
      for (let s = 0; s < D.steps_per_frame; s++) {
        state = call(step, [...state, actions]);
        const av = state[L.box_angular_velocity], j = CI * 3 + 2;
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
