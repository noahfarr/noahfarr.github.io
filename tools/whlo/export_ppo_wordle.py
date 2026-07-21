"""Export the token-level RecurrentPPO Wordle loop to StableHLO for whlo, so the
blog demo trains for real in the browser.

This mirrors baselines/recipes/recurrent_ppo_wordle.py: the policy outputs tokens,
and MCP converts token -> tool call -> MultiDiscrete guess. The recipe's to_action
is a host pure_callback (tokenizer.decode + json.loads), which cannot cross into
whlo, so this uses the jittable letter-codec instead: the tokens between the
<tool_call>/</tool_call> ids are the five letters, gathered straight into the
env's MultiDiscrete action. Same conversion, pure JAX.

Two modules are emitted per torso, exactly like tools/whlo/export_mcts.py:

    init.mlir        key         -> state
    train_step.mlir  state, key  -> state, metric   (metric = mean reward this update)

Everything is static-shape: num_envs, num_steps, the vocab, the word list and the
torso width are all baked in. Run against the reinforcement-learning workspace:

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

import environments
import policies
from environments.wrappers import MCP, SameStepAutoReset, Vectorize
from algorithms.recurrent_ppo import RecurrentPPO, RecurrentPPOConfig
from networks import (
    ActorCritic,
    Chunked,
    MinGRUCell,
    Network,
    Projection,
    RNN,
    SSM,
    SelfAttention,
    Stack,
    causal_attention_mask,
)

WORD_LENGTH = 5
START, END, PAD = 26, 27, 28
ABSENT, PRESENT, CORRECT, ILLEGAL = 29, 30, 31, 32
VOCAB_SIZE = 33
WORDS = [
    "crane", "slate", "audio", "house", "plant", "brick", "storm", "cloud",
    "grape", "flint", "mound", "spade", "chair", "lemon", "frost", "glaze",
]


def to_action(arguments, cursor):
    return arguments[:WORD_LENGTH]


def to_tokens(obs):
    cells = obs.reshape(WORD_LENGTH, 3)
    return jnp.where(cells.sum(-1) > 0, ABSENT + cells.argmax(-1), ILLEGAL).astype(jnp.int32)


class TokenFeatureExtractor(nn.Module):
    vocab_size: int
    features: int

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None):
        embedding = nn.Embed(self.vocab_size, self.features, name="token_embedding")
        return jnp.concatenate([embedding(action)[..., None, :], embedding(obs)], axis=-2)


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
            context_length=args.num_steps * (1 + WORD_LENGTH),
            attention_mask=causal_attention_mask,
        )
    else:
        raise ValueError(args.torso)
    return Chunked(Stack((Projection(args.hidden, activation=nn.relu), recurrent)))


def build_algorithm(args):
    words = pathlib.Path(tempfile.mkdtemp()) / "words.txt"
    words.write_text("\n".join(WORDS))
    env, env_params = environments.make(
        namespace="wordle",
        env_id="Wordle-v0",
        kwargs={"answer_pool_path": words, "guess_pool_path": words},
    )
    env = MCP(
        env,
        to_action=to_action,
        to_tokens=to_tokens,
        start=START,
        end=END,
        pad=PAD,
        vocab_size=VOCAB_SIZE,
        capacity=WORD_LENGTH,
        observation_shape=(WORD_LENGTH,),
        action_shape=(),
    )
    env = SameStepAutoReset(env)
    env = Vectorize(env, num_envs=args.num_envs)

    network = Network(
        feature_extractor=TokenFeatureExtractor(VOCAB_SIZE, args.hidden),
        torso=build_torso(args),
        head=ActorCritic(actor=nn.Dense(VOCAB_SIZE, use_bias=False), critic=nn.Dense(1)),
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
    return algorithm, args.num_steps


def leaf_layout(avals):
    return [{"dtype": str(a.dtype), "shape": list(a.shape)} for a in avals]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso", choices=["gru", "min_gru", "attention"], default="gru")
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--num-minibatches", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/ppo_wordle"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    algorithm, eval_steps = build_algorithm(args)
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)
    state = algorithm.init(init_key)

    def rollout_reward(state, key):
        def one(state, key):
            state, transition = algorithm.rollout(state, key, temperature=1.0)
            return state, transition.second.reward

        _, rewards = jax.lax.scan(one, state, jax.random.split(key, eval_steps))
        return rewards.mean()

    def train_step(state, key):
        update_key, eval_key = jax.random.split(key)
        new_state, _ = algorithm.update_step(state, update_key)
        return new_state, rollout_reward(new_state, eval_key)

    initialize = jax.jit(algorithm.init, keep_unused=True)
    step = jax.jit(train_step, keep_unused=True)

    new_state, metric = step(state, step_key)
    print(f"jax ran clean: metric (mean reward) = {float(metric):.4f}")

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
