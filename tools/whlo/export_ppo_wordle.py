"""Export a *normal* recurrent PPO Wordle loop to StableHLO for whlo, so the blog
demo trains for real in the browser.

This is a plain recurrent PPO, shaped like recurrent_ppo_minatar, NOT the Qwen
recurrent_ppo_wordle recipe. There are no tokens, no MCP, no Chunked torso, no LoRA
and no tokenizer. The policy just plays Wordle directly:

    action  = a single Discrete choice over the guess pool (pick a word)
    obs     = {"observation": last-guess feedback, "action_mask": legal words}
    torso   = GRU / MinGRU / attention (the only thing the three buttons swap)

Because the action is a whole word, every guess is a legal word, so the task is
learnable from the plain sparse +1 win reward alone (~0.34 random -> ~0.97 solved),
with no reward shaping and no exploration wall. The action_mask is carried in the
obs dict and zeroed into the actor logits; here it is all-true (every pool word is
a legal guess), so it is ready for hard-mode masking but currently a no-op.

WordChoice and the masked network are PRIVATE to this export -- they are not in the
reinforcement-learning repo, which keeps its native 5-letter action for the LLM path.

Two modules are emitted per torso, exactly like tools/whlo/export_mcts.py:

    init.mlir        key         -> state
    train_step.mlir  state, key  -> state, metric   (metric = solve rate this update)

Run against the reinforcement-learning workspace:

    cd ~/reinforcement-learning
    uv run python ~/noahfarr.github.io/tools/whlo/export_ppo_wordle.py --torso gru \
        --out ~/noahfarr.github.io/assets/pomdp/gru
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

import jax

jax.config.update("jax_use_shardy_partitioner", False)

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from jax import export
from utils import Timestep

import environments
import policies
from environments.spaces import Space
from environments.wrappers import SameStepAutoReset, Vectorize
from environments.wrappers.wrapper import Wrapper
from algorithms.recurrent_ppo import RecurrentPPO, RecurrentPPOConfig
from networks import (
    MinGRUCell,
    Projection,
    RNN,
    SSM,
    SelfAttention,
    Stack,
    causal_attention_mask,
)

WORD_LENGTH = 5
WORDS = [
    "crane", "slate", "audio", "house", "plant", "brick", "storm", "cloud",
    "grape", "flint", "mound", "spade", "chair", "lemon", "frost", "glaze",
]
LETTERS = np.array([[ord(c) - ord("a") for c in word] for word in WORDS], np.int32)


class WordChoice(Wrapper):
    """Turn the letter env into a single Discrete choice over `words`.

    action index -> look up the five letters -> step the underlying letter env.
    obs becomes {"observation": feedback, "action_mask": all-true over words}.
    """

    def __init__(self, env, words):
        super().__init__(env)
        self._words = jnp.asarray(words, jnp.int32)
        self._n = len(words)

    def _wrap(self, timestep, action):
        mask = jnp.ones((self._n,), bool)
        obs = {"observation": timestep.obs, "action_mask": mask}
        return timestep.replace(obs=obs, action=action)

    def init(self, key, params=None):
        state, timestep = self._env.init(key, params)
        return state, self._wrap(timestep, jnp.zeros((), jnp.int32))

    def step(self, key, state, action, params=None):
        state, timestep = self._env.step(key, state, self._words[action], params)
        return state, self._wrap(timestep, action)

    def observation_space(self, params=None):
        return {
            "observation": self._env.observation_space(params),
            "action_mask": Space(shape=(self._n,), dtype=bool, low=0, high=1),
        }

    def action_space(self, params=None):
        return Space(shape=(), dtype=jnp.int32, low=0, high=self._n - 1)


class MaskedActorCriticNetwork(nn.Module):
    """obs (+ previous word) -> torso -> masked actor logits and a value."""

    torso: nn.Module
    num_actions: int
    hidden: int

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, carry=None, **kwargs):
        features = jnp.concatenate(
            [
                nn.relu(nn.Dense(self.hidden, name="obs")(obs["observation"])),
                jax.nn.one_hot(action, self.num_actions),
            ],
            axis=-1,
        )
        carry, x = self.torso(carry, features, done)
        logits = nn.Dense(self.num_actions, name="actor", use_bias=False)(x)
        logits = jnp.where(obs["action_mask"], logits, jnp.finfo(logits.dtype).min)
        value = nn.Dense(1, name="critic")(x)
        return carry, (logits, value)

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.torso.initialize_carry(key, input_shape)


def build_torso(args):
    if args.torso == "gru":
        recurrent = RNN(
            cell=nn.GRUCell(
                features=args.hidden,
                recurrent_kernel_init=nn.initializers.glorot_normal(),
            )
        )
    elif args.torso == "min_gru":
        recurrent = SSM(cell=MinGRUCell(features=args.hidden))
    elif args.torso == "attention":
        recurrent = SelfAttention(
            features=args.hidden,
            num_heads=args.num_heads,
            context_length=args.num_steps,
            attention_mask=causal_attention_mask,
        )
    else:
        raise ValueError(args.torso)
    return Stack((Projection(args.hidden, activation=nn.relu), recurrent))


def build_algorithm(args):
    words = pathlib.Path(tempfile.mkdtemp()) / "words.txt"
    words.write_text("\n".join(WORDS))
    # The env is sparse by default now (shaping constants are 0): word choice makes
    # every guess legal, so the plain +1 win reward is enough and shaping is unneeded.
    env, env_params = environments.make(
        namespace="wordle",
        env_id="Wordle-v0",
        kwargs={"answer_pool_path": words, "guess_pool_path": words},
    )
    env = WordChoice(env, LETTERS)
    env = SameStepAutoReset(env)
    env = Vectorize(env, num_envs=args.num_envs)

    network = MaskedActorCriticNetwork(
        torso=build_torso(args),
        num_actions=len(WORDS),
        hidden=args.hidden,
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
        optimizer=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(2.5e-4)),
    )
    return algorithm, args.eval_steps


def leaf_layout(avals):
    return [{"dtype": str(a.dtype), "shape": list(a.shape)} for a in avals]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso", choices=["gru", "min_gru", "attention"], default="gru")
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=24)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/ppo_wordle"))
    # A local CPU run is only a sanity check; XLA's CPU backend has a shape-dependent
    # crash in the min_gru associative_scan gradient that jax.export (StableHLO, no CPU
    # codegen) and whlo do not hit. --skip-check exports without that CPU execution.
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    algorithm, eval_steps = build_algorithm(args)
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)
    state = algorithm.init(init_key)

    def solve_rate(state, key):
        def one(state, key):
            state, transition = algorithm.rollout(state, key, temperature=1.0)
            timestep = transition.second
            zeros = jnp.zeros_like(timestep.reward)
            solved = timestep.info.get("matched_secret", zeros)
            return state, (solved, timestep.done.astype(jnp.float32))

        _, (solved, done) = jax.lax.scan(one, state, jax.random.split(key, eval_steps))
        return solved.sum() / jnp.maximum(done.sum(), 1.0)

    def train_step(state, key):
        update_key, eval_key = jax.random.split(key)
        new_state, _ = algorithm.update_step(state, update_key)
        return new_state, solve_rate(new_state, eval_key)

    initialize = jax.jit(algorithm.init, keep_unused=True)
    step = jax.jit(train_step, keep_unused=True)

    if not args.skip_check:
        _, metric = step(state, step_key)
        print(f"jax ran clean: metric (solve rate) = {float(metric):.4f}")

    exported = {
        "init": export.export(initialize)(init_key),
        "train_step": export.export(step)(state, step_key),
    }
    layout = {}
    for name, exp in exported.items():
        (args.out / f"{name}.mlir").write_text(exp.mlir_module())
        layout[name] = {"inputs": leaf_layout(exp.in_avals), "outputs": leaf_layout(exp.out_avals)}
        print(f"wrote {args.out / f'{name}.mlir'}  ({len(exp.mlir_module()):,} bytes)")

    # train_step returns (state..., metric): the metric is the final output leaf.
    layout["metric_index"] = len(layout["train_step"]["outputs"]) - 1
    (args.out / "layout.json").write_text(json.dumps(layout, indent=2))
    print(f"wrote {args.out / 'layout.json'}  (metric_index {layout['metric_index']})")


if __name__ == "__main__":
    main()
