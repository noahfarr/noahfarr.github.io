// jax3d car, driven in your browser. The physics `step` (exported from ~/Jax3D
// with jax.export, compiled to WebAssembly by whlo) advances the full rigid-body
// state; render3d.mjs draws it. No server, no JAX at runtime.
//
// State is a flat list of 136 typed arrays in jax3d's pytree order; scene.json
// says which leaves hold the box/sphere transforms and the drive constants.

import { createRenderer } from "./render3d.mjs";

const WHLO = "./whlo/src/index.mjs";
const ASSET = (f) => `/assets/jax3d/${f}`;

const TA = { float32: Float32Array, int32: Int32Array, uint32: Uint32Array, bool: Uint8Array };
const toTA = (spec) => (spec.dtype === "bool" ? Uint8Array.from(spec.data.map(Number)) : TA[spec.dtype].from(spec.data));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function call(exe, inputs) {
  const named = {};
  exe.inputs.forEach((slot, i) => (named[slot.name] = inputs[i]));
  const out = exe.run(named);
  return exe.outputs.map((slot) => out[slot.name]);
}

const CSS = `
#jax3d-app { --accent: #3b82f6; max-width: 46rem; margin: 1.5rem auto; }
#jax3d-app .stage { position: relative; width: 100%; aspect-ratio: 16 / 10; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.25), 0 0 0 1px rgba(127,127,127,.25); background: #171a1f; }
#jax3d-app canvas { display: block; width: 100%; height: 100%; cursor: grab; outline: none; }
#jax3d-app canvas:active { cursor: grabbing; }
#jax3d-app .overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #cbd2da; font-weight: 600; background: rgba(20,22,26,.6); transition: opacity .3s; }
#jax3d-app .overlay.hidden { opacity: 0; pointer-events: none; }
#jax3d-app .controls { margin-top: .9rem; display: flex; align-items: center; gap: 1rem; }
#jax3d-app button.reset { font: inherit; font-weight: 500; padding: .45rem 1.1rem; cursor: pointer; border: 1px solid rgba(127,127,127,.32); border-radius: 999px; background: transparent; color: inherit; transition: background .15s ease, border-color .15s ease; }
#jax3d-app button.reset:hover { background: rgba(127,127,127,.1); border-color: rgba(127,127,127,.5); }
#jax3d-app .hint { font-size: .82rem; opacity: .6; }
#jax3d-app kbd { font: inherit; font-size: .78rem; padding: .05rem .35rem; border: 1px solid rgba(127,127,127,.4); border-bottom-width: 2px; border-radius: 5px; }
`;

function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }

async function main() {
  const root = document.getElementById("jax3d-app");
  if (!root) return;
  const style = el("style"); style.textContent = CSS; root.appendChild(style);

  const stage = el("div", "stage");
  const canvas = el("canvas"); canvas.tabIndex = 0;
  const overlay = el("div", "overlay"); overlay.textContent = "Compiling the physics…";
  stage.append(canvas, overlay);
  const controls = el("div", "controls");
  const resetBtn = el("button", "reset"); resetBtn.textContent = "Reset";
  const hint = el("div", "hint");
  hint.innerHTML = 'Click the scene, then <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to drive. Drag to orbit, scroll to zoom.';
  controls.append(resetBtn, hint);
  root.append(stage, controls);

  let renderer;
  try {
    renderer = createRenderer(canvas);
  } catch (e) {
    overlay.textContent = "WebGL isn't available in this browser.";
    console.error("[jax3d]", e);
    return;
  }

  let step, initState, scene;
  try {
    const { compile, initCompiler } = await import(WHLO);
    await initCompiler();
    const [mlir, init, sc] = await Promise.all([
      fetch(ASSET("step.mlir")).then((r) => r.text()),
      fetch(ASSET("init_state.json")).then((r) => r.json()),
      fetch(ASSET("scene.json")).then((r) => r.json()),
    ]);
    step = await compile(mlir);
    initState = init; scene = sc;
  } catch (e) {
    overlay.textContent = "Couldn't load the physics model.";
    console.error("[jax3d]", e);
    return;
  }

  const L = scene.leaf;
  const D = scene.drive;
  const actions = new Float32Array(scene.num_joints);
  let state = initState.map(toTA);

  // camera (orbit around a target that follows the chassis)
  let { azimuth, elevation, distance } = scene.camera;
  let target = [...scene.camera.target];
  const eye = () => {
    const ce = Math.cos(elevation), se = Math.sin(elevation), ca = Math.cos(azimuth), sa = Math.sin(azimuth);
    return [target[0] + distance * ce * ca, target[1] + distance * ce * sa, target[2] + distance * se];
  };

  function drawItems() {
    const items = [];
    const bp = state[L.box_position], bq = state[L.box_quat], bh = state[L.box_half_extents], ba = state[L.box_active];
    for (let i = 0; i < scene.num_boxes; i++) {
      if (!ba[i]) continue;
      items.push({ geom: "cube", pos: [bp[i * 3], bp[i * 3 + 1], bp[i * 3 + 2]],
        quat: [bq[i * 4], bq[i * 4 + 1], bq[i * 4 + 2], bq[i * 4 + 3]],
        scale: [bh[i * 3], bh[i * 3 + 1], bh[i * 3 + 2]], color: scene.box_colors[i] });
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

  const keys = {};
  canvas.addEventListener("keydown", (e) => { const k = e.key.toLowerCase(); if ("wasd".includes(k)) { keys[k] = true; e.preventDefault(); } });
  canvas.addEventListener("keyup", (e) => { keys[e.key.toLowerCase()] = false; });
  let dragging = false;
  canvas.addEventListener("pointerdown", (e) => { dragging = true; canvas.setPointerCapture(e.pointerId); canvas.focus(); });
  canvas.addEventListener("pointerup", (e) => { dragging = false; canvas.releasePointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    azimuth -= e.movementX * 0.01;
    elevation = clamp(elevation + e.movementY * 0.01, -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
  });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); distance = clamp(distance * (1 + Math.sign(e.deltaY) * 0.1), 1, 60); }, { passive: false });

  resetBtn.addEventListener("click", () => {
    state = initState.map(toTA);
    ({ azimuth, elevation, distance } = scene.camera);
    target = [...scene.camera.target];
  });

  overlay.classList.add("hidden");
  setTimeout(() => overlay.remove(), 400);
  function frame() {
    const forward = (keys.w ? 1 : 0) - (keys.s ? 1 : 0);
    const turn = (keys.a ? 1 : 0) - (keys.d ? 1 : 0);
    const left = forward - D.turn_strength * turn, right = forward + D.turn_strength * turn;
    for (const i of D.left_wheels) actions[i] = left;
    for (const i of D.right_wheels) actions[i] = right;
    const targetYaw = turn * D.max_yaw_rate;

    for (let s = 0; s < D.steps_per_frame; s++) {
      state = call(step, [...state, actions]);
      const av = state[L.box_angular_velocity], j = scene.chassis_index * 3 + 2;
      av[j] = av[j] + (targetYaw - av[j]) * D.yaw_smoothing; // steering assist, as in car.py
    }

    const bp = state[L.box_position], ci = scene.chassis_index;
    const cp = [bp[ci * 3], bp[ci * 3 + 1], bp[ci * 3 + 2]];
    for (let i = 0; i < 3; i++) target[i] += (cp[i] - target[i]) * 0.2; // smooth follow
    renderer.render(drawItems(), { eye: eye(), target, fovDeg: scene.camera.fov_deg });
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
