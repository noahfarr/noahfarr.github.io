# tools/regen — regenerate the browser-demo assets

The MCTS (`/projects/mcts/`) and jax3d (`/projects/jax3d/`) demos ship
pre-generated StableHLO modules and scene data under `assets/`. This directory
regenerates them from the source libraries, pulled from GitHub as **pinned**
dependencies (see `pyproject.toml`), so the site can rebuild its own assets
without cloning those repos.

The `.mlir` → WebAssembly compile happens **in the browser** via whlo at
runtime, so regeneration only produces the `.mlir` / scene JSON: no Rust and no
whlo are needed here.

## Setup

```bash
cd tools/regen
uv sync            # installs the pinned mcts + jax3d from GitHub
```

## Regenerate

```bash
# MCTS: assets/mcts/{init,step,pick_move}.mlir + layout.json
JAX_PLATFORMS=cpu uv run python export_mcts.py --env-id tic_tac_toe --num-simulations 400

# jax3d: assets/jax3d/{step.mlir,scene.json,init_state.json,layout.json}
JAX_PLATFORMS=cpu uv run python export_jax3d.py --num-boxes 12 --num-fixated 11

# pgx renderer: regenerates assets/js/mcts/chess_pieces.mjs and dumps reference SVGs
uv run python gen_reference.py
# then pixel-diff the deployed renderer against pgx (needs chromium + imagemagick `compare`)
node verify_render.mjs
```

`verify_render.mjs` imports `../../assets/js/mcts/render.mjs` directly, so the
harness checks the exact renderer the site ships (its single source of truth).

## Notes

- **whlo `pkg/` is vendored** (`assets/js/*/whlo/pkg`). Rebuild it from `~/whlo`
  (`cd js && npm run build`) only when whlo itself changes, then copy `pkg/` in.
- **Bump the commit pins** in `pyproject.toml` when you update a library.
- mcts pins `jax[cuda12]` on Linux, so `uv sync` pulls CUDA jax (a large
  download); it still runs on CPU with `JAX_PLATFORMS=cpu`.
- If mcts and jax3d dependency resolution ever conflicts, split them into two
  separate uv environments.

## Attribution

`assets/js/mcts/render.mjs` ports the SVG board rendering from **pgx**
(https://github.com/sotetsuk/pgx, Apache-2.0); `assets/js/mcts/chess_pieces.mjs`
is generated from pgx's bundled chess piece set (Cburnett).
