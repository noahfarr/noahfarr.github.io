// Tic-tac-toe vs. MCTS, running entirely in your browser.
//
// Three StableHLO modules (exported from ~/mcts with jax.export, see
// examples/export_web.py) are compiled to WebAssembly by whlo and driven from
// here. No server, no JAX, no pgx at runtime: the game rules and the search are
// both inside the .wasm.
//
//   init.mlir       key            -> State
//   step.mlir       State, action  -> State
//   pick_move.mlir  key, State     -> action, root_children[9], visits[N+1], parents[N+1]
//
// The board is drawn with the pixel-accurate pgx renderer (render.mjs); a
// transparent 3x3 overlay on top handles clicks and shows the MCTS visit
// heatmap. The search tree MCTS builds is drawn beside the board from the
// per-node visits/parents the search returns.
//
// State is a flat list of 10 typed arrays in pgx's pytree order (indices below,
// verified against assets/mcts/layout.json).

import { renderSVG } from "./render.mjs";
import { treeHTML } from "./tree.mjs";

// whlo is imported dynamically inside main() so a not-yet-built package surfaces
// as a friendly message rather than an uncatchable module-resolution error.
const WHLO = "./whlo/src/index.mjs";
const MLIR = (name) => `/assets/mcts/${name}.mlir`;

// State leaf indices.
const TERMINATED = 3;
const LEGAL = 5;
const COLOR = 7;
const BOARD = 8;
const WINNER = 9;

const HUMAN = 0; // human plays color 0 (X) and moves first
const AI = 1; // MCTS plays color 1 (O)

// AI-turn animation timing (ms): tree grows in over GROW_MS (+ a node's own
// NODE_MS to settle), then the best line reveals over REVEAL_MS.
const GROW_MS = 1000;
const NODE_MS = 340;
const REVEAL_MS = 550;

// ---------------------------------------------------------------------------
// whlo plumbing
// ---------------------------------------------------------------------------

function call(exe, inputs) {
  const named = {};
  exe.inputs.forEach((slot, i) => {
    named[slot.name] = inputs[i];
  });
  const out = exe.run(named);
  return exe.outputs.map((slot) => out[slot.name]);
}

const i32 = (n) => new Int32Array([n]);

function randKey() {
  const k = new Uint32Array(2);
  crypto.getRandomValues(k);
  return k;
}

const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

// The pgx tic-tac-toe SVG is 240x240 with the 3x3 play area inset 30px (12.5%),
// so the click/heat overlay is inset 12.5% on every side and split into 3x3.
const CSS = `
#mcts-app { --accent: #3b82f6; max-width: 50rem; margin: 1.5rem auto; font-variant-numeric: tabular-nums; }
#mcts-app .status { display: inline-flex; align-items: center; gap: .5rem; min-height: 1.7em; margin-bottom: .85rem; font-weight: 600; color: inherit; }
#mcts-app .status .dot { width: .55rem; height: .55rem; border-radius: 50%; background: currentColor; flex: none; }
#mcts-app .status.turn { color: var(--accent); }
#mcts-app .status.thinking { color: #9aa2ab; }
#mcts-app .status.thinking .dot { animation: mctsPulse 1s ease-in-out infinite; }
#mcts-app .status.win { color: #22c55e; }
#mcts-app .status.lose { color: #ef4444; }
#mcts-app .status.draw { color: #9aa2ab; }
@keyframes mctsPulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: .3; transform: scale(.6); } }
#mcts-app .game { display: flex; flex-wrap: wrap; justify-content: center; gap: 1.2rem; align-items: flex-start; }
/* align-self keeps the board square even if the tree panel grows taller, so the
   percentage-inset click overlay stays aligned with the drawn board. */
#mcts-app .board { position: relative; width: min(23rem, 90vw); aspect-ratio: 1; flex: 0 0 auto; align-self: flex-start; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.18), 0 0 0 1px rgba(0,0,0,.06); }
#mcts-app .board svg { display: block; width: 100%; height: auto; }
#mcts-app .overlay { position: absolute; inset: 12.5%; display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); }
#mcts-app .cell { position: relative; border: 0; margin: 0; padding: 0; background: transparent; cursor: default; }
#mcts-app .cell.playable { cursor: pointer; }
/* hover preview: an actual cross, matching pgx's X (10%-90% diagonals, 5% stroke) */
#mcts-app .cell .preview { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; transition: opacity .12s; pointer-events: none; }
#mcts-app .cell .preview line { stroke: var(--accent); stroke-width: 5; }
#mcts-app .cell.playable:hover .preview { opacity: .5; }
#mcts-app .cell .heat { position: absolute; right: 6%; bottom: 4%; font: 600 .72rem/1 ui-monospace, monospace; color: var(--accent); opacity: .85; }
#mcts-app .tree { width: min(23rem, 90vw); aspect-ratio: 1; flex: 0 0 auto; align-self: flex-start; display: flex; }
#mcts-app .tree .panel { flex: 1; min-width: 0; box-sizing: border-box; border: 1px solid rgba(127,127,127,.28); border-radius: 12px; padding: .5rem .6rem; display: flex; flex-direction: column; justify-content: center; }
#mcts-app .tree svg { display: block; width: 100%; height: auto; color: inherit; }
/* grow-in: each node/edge fades+pops in at its birth time (animation-delay set
   per element by tree.mjs, in the order the search created the nodes). */
@keyframes mctsNodeIn { from { opacity: 0; transform: scale(0); } to { opacity: var(--o); transform: scale(1); } }
@keyframes mctsEdgeIn { from { opacity: 0; } to { opacity: var(--o); } }
#mcts-app .tree .edge { --o: .16; stroke: currentColor; stroke-width: .6; opacity: var(--o); animation: mctsEdgeIn .3s ease-out backwards; }
#mcts-app .tree .node { --o: .4; fill: currentColor; opacity: var(--o); transform-box: fill-box; transform-origin: center; animation: mctsNodeIn .34s ease-out backwards; }
/* best-line reveal: once the tree has grown, the .done class lights up the
   principal variation. */
#mcts-app .tree svg.done .edge.pv { stroke: var(--accent); stroke-width: 1.4; opacity: .9; transition: stroke .45s ease, stroke-width .45s ease, opacity .45s ease; }
#mcts-app .tree svg.done .node.pv { fill: var(--accent); opacity: 1; transition: fill .45s ease, opacity .45s ease; }
/* skeleton placeholder before the first move: dim, static, gently pulsing */
#mcts-app .tree .panel.skeleton .node, #mcts-app .tree .panel.skeleton .edge { animation: none; }
#mcts-app .tree .panel.skeleton svg { animation: mctsSkeleton 2.6s ease-in-out infinite; }
@keyframes mctsSkeleton { 0%, 100% { opacity: .12; } 50% { opacity: .26; } }
#mcts-app .tree .caption { margin-top: .4rem; font-size: .72rem; opacity: .6; text-align: center; }
#mcts-app .tree .placeholder { font-size: .8rem; opacity: .5; text-align: center; padding: 1rem .5rem; }
#mcts-app .controls { margin-top: 1rem; display: flex; align-items: center; gap: .8rem; }
#mcts-app button.newgame { font: inherit; font-weight: 500; padding: .45rem 1.1rem; cursor: pointer; border: 1px solid rgba(127,127,127,.32); border-radius: 999px; background: transparent; color: inherit; transition: background .15s ease, border-color .15s ease; }
#mcts-app button.newgame:hover { background: rgba(127,127,127,.1); border-color: rgba(127,127,127,.5); }
#mcts-app button.newgame:active { background: rgba(127,127,127,.16); }
#mcts-app .hint { margin-top: .75rem; font-size: .78rem; opacity: .5; }
`;

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

// A small static tree, drawn dim and gently pulsing as a skeleton placeholder
// until the first real search replaces it.
const SK_PARENTS = Int32Array.from([-1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 7, 8, 8]);
const SK_VISITS = Float32Array.from([64, 24, 20, 16, 10, 8, 9, 7, 8, 6, 4, 3, 3, 3, 3]);
const SKELETON_HTML = treeHTML(SK_PARENTS, SK_VISITS, {
  grow: 0,
  caption: "search tree",
});

function showSkeleton(panel) {
  panel.classList.add("skeleton");
  panel.innerHTML = SKELETON_HTML;
}

function mountUI(root) {
  const style = document.createElement("style");
  style.textContent = CSS;
  root.appendChild(style);

  const status = el("div", "status");
  status.append(el("span", "dot"), el("span", "label"));

  const game = el("div", "game");
  const board = el("div", "board");
  const svgHolder = el("div", "svg-holder");
  const overlay = el("div", "overlay");
  const cells = [];
  for (let i = 0; i < 9; i++) {
    const c = el("button", "cell");
    c.dataset.i = String(i);
    c.innerHTML =
      '<svg class="preview" viewBox="0 0 100 100" aria-hidden="true">' +
      '<line x1="10" y1="10" x2="90" y2="90" /><line x1="10" y1="90" x2="90" y2="10" /></svg>';
    c.appendChild(el("span", "heat"));
    overlay.appendChild(c);
    cells.push(c);
  }
  board.append(svgHolder, overlay);

  const tree = el("div", "tree");
  const treePanel = el("div", "panel");
  showSkeleton(treePanel);
  tree.appendChild(treePanel);

  game.append(board, tree);

  const controls = el("div", "controls");
  const newBtn = el("button", "newgame");
  newBtn.textContent = "New game";
  controls.appendChild(newBtn);

  const note = el("div", "hint");
  note.textContent = "You play × and move first.";

  root.append(status, game, controls, note);
  return { status, svgHolder, cells, newBtn, note, treePanel };
}

// ---------------------------------------------------------------------------
// Game
// ---------------------------------------------------------------------------

async function main() {
  const root = document.getElementById("mcts-app");
  if (!root) return;
  const ui = mountUI(root);

  const statusLabel = ui.status.querySelector(".label");
  const setStatus = (text, kind = "") => {
    statusLabel.textContent = text;
    ui.status.className = "status" + (kind ? " " + kind : "");
  };

  let mods;
  setStatus("Compiling the model…", "thinking");
  try {
    const { compile, initCompiler } = await import(WHLO);
    await initCompiler();
    const [initSrc, stepSrc, pickSrc] = await Promise.all(
      ["init", "step", "pick_move"].map((n) =>
        fetch(MLIR(n)).then((r) => {
          if (!r.ok) throw new Error(`${MLIR(n)} -> ${r.status}`);
          return r.text();
        })
      )
    );
    const [init, step, pick] = await Promise.all([initSrc, stepSrc, pickSrc].map((t) => compile(t)));
    mods = { init, step, pick };
  } catch (err) {
    setStatus("Couldn't load the model", "lose");
    console.error("[mcts] load/compile failed:", err);
    ui.note.textContent = "Missing assets/mcts/*.mlir or assets/js/mcts/whlo/. Run examples/export_web.py and copy whlo's js/ (see the page source).";
    return;
  }

  let state;
  let busy = false;

  const isTerminal = () => Boolean(state[TERMINATED][0]);
  const colorToMove = () => state[COLOR][0];

  function render(heat = null) {
    const board = state[BOARD];
    const legal = state[LEGAL];
    const humanTurn = !isTerminal() && colorToMove() === HUMAN;
    const max = heat ? Math.max(1e-9, ...heat) : 0;

    // pgx-accurate board. pgx emits no viewBox, so add one (identity, from the
    // native width/height) and drop the fixed size — otherwise width:100% would
    // stretch the SVG box while its content stayed at native px in the corner,
    // and the click/heat overlay would not line up.
    ui.svgHolder.innerHTML = renderSVG({ env_id: "tic_tac_toe", board: Array.from(board) });
    const svg = ui.svgHolder.querySelector("svg");
    if (svg) {
      svg.setAttribute("viewBox", `0 0 ${svg.getAttribute("width")} ${svg.getAttribute("height")}`);
      svg.removeAttribute("width");
      svg.removeAttribute("height");
    }

    // interactive overlay
    ui.cells.forEach((cell, i) => {
      const empty = board[i] === -1;
      const playable = humanTurn && legal[i] === 1 && empty;
      cell.classList.toggle("playable", playable);
      cell.disabled = !playable;

      const show = heat && empty && heat[i] > 0;
      cell.style.backgroundColor = show ? `rgba(59,130,246,${(0.08 + 0.5 * (heat[i] / max)).toFixed(3)})` : "";
      cell.querySelector(".heat").textContent = show ? String(Math.round(heat[i])) : "";
    });
  }

  function announceIfOver() {
    if (!isTerminal()) return false;
    const w = state[WINNER][0];
    if (w === HUMAN) setStatus("You win", "win");
    else if (w === AI) setStatus("MCTS wins", "lose");
    else setStatus("Draw", "draw");
    return true;
  }

  async function aiTurn() {
    busy = true;
    setStatus("MCTS is searching…", "thinking");
    await raf();
    await raf(); // let the status paint before the blocking search

    const [actionArr, rootChildren, visits, parents] = call(mods.pick, [randKey(), ...state]);
    const heat = new Float32Array(9);
    for (let a = 0; a < 9; a++) {
      const c = rootChildren[a];
      heat[a] = c >= 0 ? visits[c] : 0;
    }
    render(heat); // show the search's visit distribution on the board
    ui.treePanel.classList.remove("skeleton");
    ui.treePanel.innerHTML = treeHTML(parents, visits, { grow: GROW_MS / 1000 }); // grow the tree beside it
    const treeSvg = ui.treePanel.querySelector("svg");
    // reveal the best line once the tree has finished growing in
    setTimeout(() => treeSvg && treeSvg.classList.add("done"), GROW_MS + NODE_MS);
    await sleep(GROW_MS + NODE_MS + REVEAL_MS);

    state = call(mods.step, [...state, actionArr]);
    render();
    busy = false;
    if (!announceIfOver()) setStatus("Your move", "turn");
  }

  function onCellClick(ev) {
    if (busy || isTerminal()) return;
    const cell = ev.currentTarget;
    if (cell.disabled) return;
    const i = Number(cell.dataset.i);
    if (state[LEGAL][i] !== 1 || colorToMove() !== HUMAN) return;

    state = call(mods.step, [...state, i32(i)]);
    render();
    if (announceIfOver()) return;
    aiTurn();
  }

  function newGame() {
    if (busy) return;
    state = call(mods.init, [randKey()]);
    render();
    showSkeleton(ui.treePanel);
    setStatus("Your move", "turn");
  }

  ui.cells.forEach((c) => c.addEventListener("click", onCellClick));
  ui.newBtn.addEventListener("click", newGame);
  newGame();
}

main();
