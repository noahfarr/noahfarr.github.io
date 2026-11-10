---
layout: post
title: Abstractions for scaling reinforcement learning in JAX
date: 2026-11-10 10:00:00 +0100
description: the abstractions behind boonta, a JAX library in which one algorithm interface trains RL agents from scratch, pre-trains and post-trains LLMs, and fine-tunes vision-language-action models
tags: reinforcement-learning jax llms partial-observability software
categories: research
featured: true
---

In [Behavioral Language Modeling](/blog/2026/behavioral-language-modeling/) I argued that
a language model acting as an agent is a recurrent policy in a POMDP, and that a
recurrent RL algorithm can train it whether a GRU or a language model processes the
history. This post is about the library where that holds in code. It is called [boonta](https://github.com/noahfarr/boonta), after
the podrace Anakin wins in _The Phantom Menace_, and it is built around one idea. The
algorithm, the environment and the training loop should never meet. If they never meet,
then a GRU learning CartPole from scratch and a pretrained Qwen3 playing Wordle through
tool calls are the same program with different parts plugged in, and everything the RL
community has built for partially observable problems applies to the LLM for free.

The same holds along the whole training pipeline and across modalities. Next-token
pre-training is behaviour cloning, so it runs through the same algorithm interface as RL
post-training, with the same network and checkpoints. And a vision-language-action model
such as π0 is just another policy, one that reads camera frames and emits robot actions.

Most of what follows is about where the boundaries sit and why. The LLM shows up near the
end, and by then there is almost nothing left to say about it, which is the point.

<figure style="margin:1.6rem auto; text-align:center;">
<img class="repo-img-light" src="{{ '/assets/img/boonta/structure-light.webp' | relative_url }}" alt="The hangar, holding the config and the recipe, stands on boonta. The recipe builds a podracer, one of Anakin, Sebulba or Quadinaros, which holds a curriculum wrapping the algorithm (network, optimizer and more), the environment (vectorize around auto-reset around the game) and, for offline training, the dataset. Beside the track a grandstand watches the race. Artisans along its wall craft checkpoints, metrics and videos, and loggers broadcast them from the tower on its roof." style="width:100%; max-width:760px; margin:0 auto;">
<img class="repo-img-dark" src="{{ '/assets/img/boonta/structure-dark.webp' | relative_url }}" alt="The same diagram of boonta's structure in dark colours." style="width:100%; max-width:760px; margin:0 auto;">
<figcaption class="caption">The whole library on one page. The hangar turns a config into a recipe, the recipe builds a podracer, and the podracer races an algorithm against an environment while the grandstand watches.</figcaption>
</figure>

## Three things that never meet

boonta has three kinds of object, and each one owns exactly one thing.

1. **The algorithm** decides what to do and learns from what happened. It never touches
   the environment or the devices.
2. **The environment** is the game. It never knows who is playing it.
3. **The podracer** is the training loop. It owns the devices and the sharding, and calls
   the other two. It never knows what either of them is doing.

The only place they meet is a **recipe**, one plain file per algorithm and suite that
builds an environment, a network, an optimizer and an algorithm and hands them to a
podracer. There is no shared helper layer between recipes. Each one reads top to bottom
on its own, and copying the closest one is how you start a new one.

This sounds like ordinary modularity, and most RL libraries claim it. The test is what
each interface is allowed to see, and the rest of the post is mostly about those
signatures.

## The algorithm sees three calls

An algorithm is three pure functions, and all three run inside `jax.jit`.

| Method | Called | Returns |
| --- | --- | --- |
| `init(key, timestep)` | once, with the first timestep | the algorithm state |
| `step(state, key, timestep, temperature)` | every environment step | `(state, action, aux)` |
| `update(state, key, transitions)` | once per batch | the new state |

The network and the optimizer are passed in, never built inside, so the same `PPO` runs a
CNN on MinAtar and an MLP on CartPole. And there is no policy object. The network's head
returns a distribution, so the network _is_ the policy and `step` simply samples from what
it returns. Temperature goes to the head and nowhere else, which is how one flag makes
evaluation greedy for a categorical policy, ε-greedy for a Q-network and tighter for a
Gaussian.

The decision I care most about is where the recurrent state lives. It does not live in the
algorithm. A recurrent carry is rollout state, exactly like the environment state, so the
**podracer** owns it. A recurrent algorithm's `step` takes the carry in and hands it back
out, the podracer records the carry that entered every step, and `update` replays the
rollout from the carry it started on. Two things follow. A checkpoint holds only what was
learned, never a half-finished episode's memory. And the algorithm cannot tell what kind
of memory it is threading through time, a GRU's hidden state, a state-space model's scan
or a transformer's KV cache. That last fact is what the whole LLM story rests on.

## The environment sees nothing but actions

Every environment, whether it is pure JAX, a Gymnasium simulator or C++ called through
FFI, has the same small interface. `init` and `step` return a `Timestep` carrying the
observation, the action that led to it, the reward and two flags, `terminated` for a real
end and `truncated` for a cut. The distinction matters because a value target bootstraps
through a truncation but not through a termination, and conflating the two is one of the
quietest bugs in RL.

Two conventions do a lot of work. Episodes reset on the same step they end, so the final
reward and the first observation of the next episode arrive together and the loop never
pauses. And the first timestep of every episode sets `terminated`, which is exactly the
signal a recurrent network needs to wipe its memory. Every recurrent block resets to a
blank carry wherever that flag is set, so episodes can never leak into each other, however
the rollout is chunked.

Wrappers are environments around environments, and a podracer always receives a batched,
auto-resetting stack. A game with free parameters, such as a Kinetix level, also exposes
`update(state, key, theta=...)`, which means "play this theta from its start". That one method is how curricula reach into the game without the game knowing
a curriculum exists.

## Three podracers

The training loop is modelled on DeepMind's Podracer architectures
{% cite hessel2021podracer --file references %}, and the three podracers are named after
the drivers in the race.

🏎️ **Anakin** is for environments written in JAX. Acting, stepping and learning compile
into a single program, the environments are split across every device on a `data` mesh
axis, and the parameters are replicated. Training is one `jax.jit` call that scans over
updates, each of which scans over environment steps and then calls `update`. With
`num_processes` set it spans every device of every machine, so going from one GPU to a
SLURM allocation is a command-line flag.

🔧 **Sebulba** is for environments that are slow or not JAX at all. Actor threads step their
own environments on some devices while a learner trains on the rest. The lag between them
is fixed at exactly one update, as in Cleanba {% cite huang2023cleanba --file references %},
so rollout _k_ is always collected with the parameters of update _k_−2 and a run is
reproducible at a fixed seed regardless of thread timing.

📼 **Quadinaros** is for offline RL and behaviour cloning. It samples batches from a fixed
dataset inside a jitted scan and only touches the environment to evaluate.

Sharding is declared, not coded. A field of an algorithm's state marked
`metadata={"axis": "data"}`, such as a replay buffer with one row per environment, is
split across devices along with the environments. Everything unmarked is replicated. So
an algorithm never mentions a device, and a podracer never needs to know what a replay
buffer is.

## Pit and lap

Curricula are where I most wanted the boundaries to hold, because curricula want to touch
everything. They read how the agent is doing, rewrite which levels it plays, and in a
league even reset the learner. In most codebases that means a curriculum-specific training
loop.

In boonta every podracer takes two hooks instead. A **pit** learns from each rollout after
the algorithm's update and then rewrites the environment, once per update. A **lap** fires
after every single environment step, for state the game must see continuously. Both have
their own state, which the podracer carries and shards like anything else.

A curriculum is then just a function that takes an algorithm and an environment and returns
an algorithm, an environment, a pit and a lap. Prioritized Level Replay
{% cite jiang2021prioritized --file references %} is a pit that scores levels by how much
the agent learned from them and replays the most useful ones. Success-rate-targeted
sampling is a pit that keeps a fixed pool of tasks and draws them by how close each one's
success rate sits to a target. A league is a pit that seats opponents from a pool of
earlier policies, all the way up to PSRO {% cite lanctot2017unified --file references %}.
None of them needed a change to an algorithm, an environment or a podracer, and all of them
run on every podracer. This is also the part of the library I care most about for
generalization. Which tasks an agent sees, in what order, and how diverse they are is the
lever on whether it learns a skill or memorises a training set, and it should be as easy to
swap as an optimizer.

## The network is three slots

A `Network` has a feature extractor, an optional torso and a head.

```python
network = Network(
    feature_extractor=FeatureExtractor(observation_extractor=cnn),
    torso=RNN(GRUCell(256)),  # optional, takes and returns a carry
    head=ActorCritic(actor=Categorical(nn.Dense(num_actions)), critic=nn.Dense(1)),
)
```

Every torso is a `Block` whose call takes `(carry, x, done)` and returns `(carry, x)`.
Cells, torsos and stacks are separate config groups, so a GRU in a sequential scan, a
MinGRU in a parallel scan, a self-attention block with a KV cache, a highway stack or a
Llama-style stack are all one command-line override apart. Mixed precision follows one
rule, bf16 for anything that enters a matrix multiply and fp32 for anything accumulated,
and distribution parameters and values are always cast back to fp32 before they reach a
loss.

The interface is narrow enough that a pretrained transformer is just another `Block`.
Qwen3 and Gemma are implemented as stacks of attention and GLU blocks whose carry is the
KV cache, and their HuggingFace weights load straight into them. LoRA wraps any block,
keeps the base weights frozen and optionally quantized, and trains only the adapters.

## An LLM, plugged in

Here is the Wordle recipe, trimmed to the parts that matter. The game itself is a plain
JAX environment that takes a five-letter guess and returns per-letter feedback. Two
wrappers turn it into a language game.

```python
env = environments.make(**cfg.environment)               # wordle, in JAX
env = MCP(env, to_action=parse_tool_call, to_tokens=feedback_tokens, ...)
env = Prompt(env, tokenizer.encode(SYSTEM_PROMPT).ids, pad)
env = instantiate(cfg.environment.auto_reset, env)

torso = Chunked(LoRA(block=qwen3, params=quantize(pretrained), rank=8, alpha=16.0))
embed = nn.Embed(vocab_size, hidden, embedding_init=init_fn(embeddings))

network = Network(
    feature_extractor=TokenFeatureExtractor(embed=embed),
    torso=torso,
    head=ActorCritic(actor=Categorical(Unembed(embed=embed)), critic=nn.Dense(1)),
)
algorithm = RecurrentPPO(cfg=..., network=network, optimizer=...)
```

The `MCP` wrapper is where tokens become actions. From the algorithm's side the action
space is the vocabulary, and every environment step is one sampled token. The wrapper
watches the stream, starts collecting when the model opens a `<tool_call>`, and when it
closes the call it parses the JSON, steps the real game once with the guess, and returns
the feedback as observation tokens. On every other step nothing comes back from the world,
and the observation is padding. The `Prompt` wrapper feeds the system prompt in as the
first observation of every episode.

So the POMDP from the BLM post is not a metaphor here, it is the data layout. Observations
and actions arrive on separate, step-aligned channels. The policy's tokens are the action
channel, the prompt and the game's feedback are the observation channel, and the reward
lands when the game says so. There is no loss mask anywhere in the code, because there is
nothing to mask. The KV cache is the carry, the podracer threads it through time exactly
as it would a GRU's hidden state, and the critic reads the same torso and so conditions on
the whole history. The action embedding and the actor's readout are both tied to the
model's own token embeddings, so "tokens are actions" is literal all the way down to the
parameters.

Then the payoff. Swap `RecurrentPPO` for `RecurrentGRPO` and nothing else changes. Swap
Qwen3 for Gemma and nothing else changes. Set `num_processes=8` and the same run spans two
machines.

## Pre-training is behaviour cloning

The same picture covers pre-training, seen from the other side. A corpus is a record of
someone else's actions, and next-token prediction is behaviour cloning on that record.
In boonta that is not an analogy either. It is `RecurrentBC` on Quadinaros, which samples
token sequences from a dataset instead of rolling out an environment, and trains the same
`Network` with the same torso, the same sharding and the same checkpoint format as the RL
run that follows it.

So a model can go from pre-training to RL post-training without changing libraries,
data layouts or model code. The pre-training run and the RL run differ in the algorithm
object and the podracer, and in nothing else. It is also the shape of the comparison in
the [BLM post](/blog/2026/behavioral-language-modeling/). Cloning every token and cloning
only the agent's actions are the same algorithm with a different choice of which tokens
count as actions.

## Vision-language-action models

Nothing above is specific to text. A vision-language-action model is a policy whose
observations include camera frames and whose actions are robot commands, and it fits the
same three slots. boonta implements π0, with a SigLIP vision encoder, a Gemma language
backbone and a separate action expert, and loads its published weights into that
structure. On the environment side, LIBERO and Isaac Lab Arena wrap robot manipulation
suites behind the usual `init` and `step`.

That means a VLA gets the same treatment as the LLM. It can be fine-tuned by behaviour
cloning on demonstrations through Quadinaros, then trained further with RL in simulation
on Anakin or Sebulba, with curricula over tasks through the same pit. The algorithms do not know they are driving a robot, just as they did
not know they were writing tool calls.

## How it compares

boonta stands on a lot of good work. PureJaxRL
{% cite lu2022discovered --file references %} showed that compiling the whole loop into
one program is the fastest way to do RL on accelerators. Stoix and Mava brought that to
many algorithms and to cooperative multi-agent RL, PufferLib to very fast native
environments, Cleanba to reproducible actor-learner training, and TorchRL to PyTorch.
What boonta adds is mostly a matter of scope under one interface. It trains online,
offline and through curricula with the same algorithm objects, runs on all three
podracers, and treats pretrained LLMs and VLAs as just another torso. As far as I know
it is the only JAX library that trains LLMs as agents this way.

## What is next

The machinery is in place, and the next step is to use it for the question I actually care
about, whether RL post-training teaches a model a skill or only fits its training tasks.
boonta makes the ingredients of that question cheap to vary, the task distribution, the
curriculum, the memory, the algorithm, and it makes held-out evaluation the default rather
than an afterthought. If you want to try it, the
[quickstart](https://github.com/noahfarr/boonta/blob/main/quickstart.ipynb) trains PPO on
Breakout one part at a time. And if you think one of the boundaries is in the wrong place,
I would like to hear about it, so [get in touch](/).

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
