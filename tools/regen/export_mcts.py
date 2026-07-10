"""Export the pieces needed to run MCTS in the browser via whlo.

Emits three StableHLO modules that whlo (https://github.com/noahfarr/whlo)
compiles to WebAssembly and runs client-side:

    init.mlir       key            -> State          (fresh game)
    step.mlir       State, action  -> State          (apply one move)
    pick_move.mlir  key, State     -> action, visits (run MCTS, pick a move)

Everything is single-instance (no vmap): one board, one search tree. That keeps
the emitted gather/scatter single-instance and the shapes fully static, which is
exactly whlo's sweet spot. `env.step` is traced and inlined, so the game rules
ship baked into the modules; no pgx at runtime.

`num_simulations` (tree capacity) is baked in at export time because whlo has no
dynamic shapes. Re-run with a different --num-simulations to change AI strength.

Usage:
    uv run python examples/export_web.py \
        --out ~/noahfarr.github.io/assets/mcts --num-simulations 400
"""

from __future__ import annotations

import argparse
import json
import pathlib

import jax
import jax.numpy as jnp
import pgx
from jax import export

from mcts import instantiate, search, action_selection
from mcts.tree import Tree


def build_fns(env: pgx.Env, num_simulations: int):
    def init(key):
        return env.init(key)

    def step(state, action):
        return env.step(state, action)

    def pick_move(key, state):
        tree = instantiate(env, state, num_simulations)
        tree, action = search(
            key, tree, env.step, num_simulations, jnp.int32(0), jnp.int32(1)
        )
        # Return enough of the tree to both drive the per-action heatmap and draw
        # the whole search tree in the browser:
        #   root_children: node id reached by each root action (-1 if unexpanded)
        #   visits:  per-node visit count (node size / heatmap)
        #   parents: per-node parent id (-1 => node not in the tree); gives edges
        root_children = tree.children[Tree.ROOT_INDEX]  # (num_actions,)
        return (
            action.astype(jnp.int32),
            root_children.astype(jnp.int32),
            tree.visits.astype(jnp.float32),  # (num_simulations + 1,)
            tree.parents.astype(jnp.int32),   # (num_simulations + 1,)
        )

    return init, step, pick_move


def leaf_layout(avals):
    """Positional dtype/shape list, matching how whlo names arg0..argN."""
    return [{"dtype": str(a.dtype), "shape": list(a.shape)} for a in avals]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-id", default="tic_tac_toe")
    ap.add_argument("--num-simulations", type=int, default=400)
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2] / "assets" / "mcts",
        help="output directory for the .mlir modules",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    env = pgx.make(args.env_id)
    init, step, pick_move = build_fns(env, args.num_simulations)

    key = jax.random.PRNGKey(0)
    state = env.init(key)
    action = jnp.int32(0)

    # keep_unused=True is essential: jax.jit otherwise DCEs inputs a function
    # doesn't read (env.step ignores observation/rewards), which would prune them
    # from that module's signature and break the fixed positional State layout the
    # JS driver relies on. With it, every module keeps the full pytree order.
    exports = {
        "init": export.export(jax.jit(init, keep_unused=True))(key),
        "step": export.export(jax.jit(step, keep_unused=True))(state, action),
        "pick_move": export.export(jax.jit(pick_move, keep_unused=True))(key, state),
    }

    layout = {}
    for name, exp in exports.items():
        path = args.out / f"{name}.mlir"
        path.write_text(exp.mlir_module())
        layout[name] = {
            "inputs": leaf_layout(exp.in_avals),
            "outputs": leaf_layout(exp.out_avals),
        }
        n = len(exp.mlir_module())
        print(f"wrote {path}  ({n:,} bytes)")

    # The JS driver maps its State to arg0..argN positionally; this file documents
    # that order (and doubles as a sanity check against the JS constants).
    (args.out / "layout.json").write_text(json.dumps(layout, indent=2))
    print(f"wrote {args.out / 'layout.json'}")

    print(
        f"\nenv={args.env_id}  num_actions={env.num_actions}  "
        f"num_players={env.num_players}  num_simulations={args.num_simulations}"
    )
    print("State leaf order (inputs to step / outputs of init):")
    for i, leaf in enumerate(layout["step"]["inputs"][:-1]):
        print(f"  [{i}] {leaf['dtype']}{leaf['shape']}")


if __name__ == "__main__":
    main()
