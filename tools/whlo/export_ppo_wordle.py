"""Export one RecurrentPPO training step on Wordle to StableHLO for whlo.

This is the grad-through-scan smoke test for the blog post: it takes the repo's
real RecurrentPPO.update_step -- rollout scan, reverse-scan GAE, threefry action
sampling, and value_and_grad *through the GRU's nn.scan* (BPTT) inside nested
epoch/minibatch scans, then an Adam update -- and exports the whole thing as a
single StableHLO module, exactly the way tools/whlo/export_mcts.py exports MCTS.

The network is a small GRU torso behind the standard (obs, action, carry) torso
interface, so nothing about update_step knows a recurrent net is involved. A
transformer/LLM torso would slot into the same call with its own (KV-cache) risks;
the recurrent cell is the thing that exercises grad-through-scan, which is the
construct most likely to trip a from-scratch StableHLO compiler.

Everything is static-shape (whlo has no dynamic shapes): num_envs, num_steps,
num_minibatches, update_epochs, the word count, and the GRU width are all baked in.

This script lives in the site repo but imports the reinforcement-learning workspace
packages (algorithms/networks/environments/policies/utils), so run it against that
project's environment:

    cd ~/reinforcement-learning
    uv run python ~/noahfarr.github.io/tools/whlo/export_ppo_wordle.py --out /tmp/ppo_wordle

Then compile the emitted module with whlo:

    cd ~/whlo
    cargo run -q -p whlo-cli -- compile /tmp/ppo_wordle/train_step.mlir -o /tmp/ppo_wordle/train_step.wasm
"""

from __future__ import annotations

import argparse
import json
import pathlib

# Neutralise the host-callback logger before importing the algorithm: lox.log is
# called inside update_minibatch and would otherwise emit an io_callback into the
# StableHLO, which whlo (rightly) will not compile. We only want the math.
import lox

lox.log = lambda *args, **kwargs: None

import jax

# jax.export defaults to emitting the Shardy (`sdy`) dialect; whlo parses the
# plain StableHLO subset, so turn the partitioner off to keep sdy.mesh/sdy.sharding
# out of the module.
jax.config.update("jax_use_shardy_partitioner", False)

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from jax import export

import environments
import policies
from environments.wrappers import Vectorize
from algorithms.recurrent_ppo import RecurrentPPO, RecurrentPPOConfig
from networks import ActorCritic, FeatureExtractor, Network, Projection, RNN, Stack


def build_algorithm(args):
    env, env_params = environments.make(
        namespace="wordle",
        env_id="Wordle-v0",
        kwargs={"num_words": args.num_words},
    )
    num_actions = env.action_space(env_params).num_categories
    env = Vectorize(env, num_envs=args.num_envs)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [lambda obs: obs.astype(jnp.float32), nn.Dense(args.hidden), nn.relu]
            ),
            # Wordle feedback is positional, so the policy needs its own past
            # actions alongside the observation. Same as the real recipe.
            action_extractor=lambda action: jax.nn.one_hot(action, num_actions),
        ),
        torso=Stack(
            (
                Projection(args.hidden, activation=nn.relu),
                RNN(cell=nn.GRUCell(features=args.hidden)),
            )
        ),
        head=ActorCritic(actor=nn.Dense(num_actions), critic=nn.Dense(1)),
    )

    cfg = RecurrentPPOConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    algorithm = RecurrentPPO(
        cfg=cfg,
        environment=env,
        environment_params=env_params,
        network=network,
        policy=policies.categorical,
        optimizer=optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(2.5e-4),
        ),
    )
    return algorithm, num_actions


def leaf_layout(avals):
    return [{"dtype": str(a.dtype), "shape": list(a.shape)} for a in avals]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-words", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--num-minibatches", type=int, default=2)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/ppo_wordle"))
    parser.add_argument("--dump-inputs", action="store_true",
                        help="also write init-state leaves + a jax reference output as .npy for a whlo run/diff")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    algorithm, num_actions = build_algorithm(args)

    # Seed with PRNGKey (raw uint32[2]), like export_mcts.py: a typed jax.random.key
    # input makes jax stamp a @Sharding custom_call on the key arg, which whlo won't parse.
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)
    state = algorithm.init(init_key)

    # The exported entry point: one full PPO update, returning just the new state
    # pytree (drop the () aux so every output leaf is an array).
    def train_step(state, key):
        new_state, _ = algorithm.update_step(state, key)
        return new_state

    jitted = jax.jit(train_step, keep_unused=True)

    # Sanity: it must run under plain JAX before we trust the export.
    _ = jitted(state, step_key)
    print("jax update_step ran clean under jit")

    exported = export.export(jitted)(state, step_key)
    mlir = exported.mlir_module()
    path = args.out / "train_step.mlir"
    path.write_text(mlir)
    print(f"wrote {path}  ({len(mlir):,} bytes)")

    layout = {
        "inputs": leaf_layout(exported.in_avals),
        "outputs": leaf_layout(exported.out_avals),
    }
    (args.out / "layout.json").write_text(json.dumps(layout, indent=2))
    print(f"wrote {args.out / 'layout.json'}  "
          f"({len(layout['inputs'])} inputs, {len(layout['outputs'])} outputs)")

    # A quick op census so we can see what whlo has to swallow.
    import re
    ops = re.findall(r"stablehlo\.([a-z_]+)", mlir)
    from collections import Counter
    census = Counter(ops)
    print("stablehlo op census (top 25):")
    for op, n in census.most_common(25):
        print(f"  {n:5d}  stablehlo.{op}")
    for interesting in ("while", "scatter", "gather", "reduce", "dynamic_slice",
                        "dynamic_update_slice", "rng_bit_generator", "sort"):
        if interesting in census:
            print(f"  [region/dynamic op present] stablehlo.{interesting} x{census[interesting]}")

    if args.dump_inputs:
        flat_in, _ = jax.tree.flatten((state, step_key))
        for i, leaf in enumerate(flat_in):
            np.save(args.out / f"in_{i:03d}.npy", np.asarray(leaf))
        out = jitted(state, step_key)
        flat_out, _ = jax.tree.flatten(out)
        for i, leaf in enumerate(flat_out):
            np.save(args.out / f"ref_{i:03d}.npy", np.asarray(leaf))
        print(f"dumped {len(flat_in)} input leaves and {len(flat_out)} reference outputs")


if __name__ == "__main__":
    main()
