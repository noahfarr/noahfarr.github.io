---
layout: post
title: remembering without a replay buffer
date: 2026-07-05 09:00:00 +0200
description: an accessible writeup of our paper on streaming RL under partial observability with real-time recurrent learning
tags: reinforcement-learning streaming partial-observability recurrent
categories: research
featured: true
---

This is an accessible writeup of a paper I wrote with Aryaman Reddi, Carlo D'Eramo, and
Jan Peters, *Streaming Reinforcement Learning under Partial Observability with Real-Time
Recurrent Learning* {% cite farr2026streaming --file references %}, which won a best paper award at the RL
in Big Worlds workshop at RLC. In one sentence: we teach a streaming RL agent to
remember, without a replay buffer and without ever unrolling the network through time.

## Streaming RL in one paragraph

Most deep RL keeps a big replay buffer and learns from random minibatches sampled out of
it. Streaming RL throws that out. The agent sees each transition once, updates immediately
with a batch size of one, and never stores or replays anything. It is much closer to how a
natural agent actually learns: online, incrementally, from a single stream of experience.
For a long time this simply did not work with deep networks, they were too unstable, but a
recent line of work on streaming deep RL got it stable and competitive. The catch is that
all of it assumed *full observability*: the current observation tells you everything you
need to know.

## The catch: memory

Real environments are partially observable. The current observation does not tell you the
whole state, so to act well the agent has to remember. That means a recurrent network
carrying a hidden state, and a recurrent network only learns if the gradient can travel
back through time: to discover that something you saw twenty steps ago mattered, credit has
to reach twenty steps back.

There are two standard ways to do that, and under streaming both break.

1. **Backprop through time (TBPTT).** Unroll the network over a window of past steps and
   backpropagate through the unrolled graph. But streaming has no buffer and a batch size of
   one, so there is no stored window to unroll. The truncation collapses to a single step,
   TBPTT(1). The agent can only assign credit to the step immediately before, and anything
   older is invisible. Long-range memory is off the table.
2. **Real-time recurrent learning (RTRL).** Instead of looking backward, carry the gradient
   *forward*. At every step you maintain how the current hidden state depends on the
   parameters, and update that online as new observations arrive. It is exact and needs no
   buffer, which fits streaming perfectly. The problem is cost: for an ordinary recurrent
   network this bookkeeping scales quadratically or worse in the hidden size, which is
   hopeless for anything deep.

So one method fits the streaming constraints but cannot see the past, and the other sees the
past exactly but is too expensive to run.

## The idea: make the recurrence diagonal

The way out is to change the architecture so that RTRL becomes cheap. If the recurrence is
*diagonal*, each hidden unit feeds back only into itself, then the way the hidden state
depends on any one parameter stays local and decoupled from the others. You no longer need a
full sensitivity matrix; a single running trace per parameter is enough to keep the exact
gradient up to date. That collapses RTRL from prohibitive to **linear in the number of
parameters**, in both time and memory.

This is exactly what recurrent trace units (RTUs) provide {% cite elelimy2024real --file references %}. The
hidden state follows a diagonal recurrence,

$$
\boldsymbol{h}_t = \boldsymbol{\Lambda}\, \boldsymbol{h}_{t-1} + \boldsymbol{W}\, \boldsymbol{x}_t,
$$

where $$\boldsymbol{\Lambda}$$ is diagonal and complex-valued, with entries
$$\lambda_k = r_k(\cos\Omega_k + i\sin\Omega_k)$$. Each hidden unit rotates and decays its own
past, with magnitude $$r_k$$ and angle $$\Omega_k$$. Because $$\boldsymbol{\Lambda}$$ is
diagonal, every unit has a single recurrent connection to itself, and that is what makes
exact RTRL linear in the number of parameters rather than prohibitive.

<figure style="margin:1.4rem auto; text-align:center;">
<img src="{{ '/assets/img/streaming/rtrl_cost.svg' | relative_url }}" alt="Per-step wall-clock of exact RTRL versus hidden size for FFN, GRU, LSTM, and RTU; GRU and LSTM grow steeply while the diagonal RTU stays flat." style="width:100%; max-width:460px; background:#fff; padding:12px; border-radius:8px; box-sizing:border-box;">
<figcaption class="caption">The cost of exact RTRL as the hidden size grows. For a GRU or LSTM it blows up; the diagonal RTU stays flat, on par with a plain feedforward net, cheap enough to run online.</figcaption>
</figure>

If you have used eligibility traces in RL, this is the same idea pushed into the recurrent
hidden state. A trace is a running summary of the past that you keep current online, so
credit can reach back arbitrarily far without ever storing the past. RTUs let the hidden
state itself act as that kind of trace.

The result is the best of both options: RTRL's exact, buffer-free, forward-in-time credit
assignment, at a cost you can actually pay.

## Dropping it into streaming algorithms

The clean part is that none of the surrounding streaming machinery has to change. RTUs slot
in as the network inside the existing streaming algorithms, for both discrete-action and
continuous-control agents. The update stays batch-size-one, buffer-free, and online. You
swap a feedforward or truncated-recurrent network for an RTU and let RTRL carry the memory.

## Does it actually help?

Three settings, from a pure memory stress test to real control.

**MemoryChain.** A diagnostic where the agent has to carry information across a chain of
length 2 up to 128 before it can be rewarded. It is nothing but memory. The streaming
TBPTT(1) baselines, whether feedforward, GRU, or even an RTU trained with one-step
truncation, all collapse as the chain grows, because one-step gradients cannot reach the
cue. Exact RTRL sustains performance across the whole range. The lesson is sharp: it is not
the architecture that saves you, it is the exact online credit assignment. An RTU with
TBPTT(1) fails right alongside the others.

<figure style="margin:1.4rem auto; text-align:center;">
<img src="{{ '/assets/img/streaming/memory_chain.svg' | relative_url }}" alt="IQM episode returns versus chain length; RTU-RTRL sustains high returns far longer than FFN, GRU-TBPTT(1), and RTU-TBPTT(1), which all collapse early." style="width:100%; max-width:480px; background:#fff; padding:12px; border-radius:8px; box-sizing:border-box;">
<figcaption class="caption">MemoryChain. The RTU with exact RTRL, in purple, holds up as the chain grows, while every one-step-truncated baseline collapses, including an RTU trained with one-step backprop. Same architecture, different credit assignment.</figcaption>
</figure>

**POPGym.** On five partially observable tasks, the streaming agents are competitive with
batched PPO, which gets a replay buffer and batched updates that ours never use.

<figure style="margin:1.4rem auto; text-align:center;">
<img src="{{ '/assets/img/streaming/popgym.svg' | relative_url }}" alt="IQM episode returns over frames on five POPGym tasks, comparing FFN, batched PPO-RTU, and streaming Stream AC and QRC with RTUs." style="width:100%; max-width:800px; background:#fff; padding:12px; border-radius:8px; box-sizing:border-box;">
<figcaption class="caption">POPGym. Across five partially observable tasks, the streaming agents with RTUs, Stream AC(&lambda;) and QRC(&lambda;), keep up with batched PPO, which uses a replay buffer and batched updates they do not.</figcaption>
</figure>

**Masked MuJoCo.** On partially observable continuous control, where either the velocities or
the positions are hidden and the agent has to reconstruct them from history, the streaming
agent recovers a substantial fraction of batched performance, again with no buffer and no
batches.

<figure style="margin:1.4rem auto; text-align:center;">
<img src="{{ '/assets/img/streaming/brax.svg' | relative_url }}" alt="IQM episode returns over frames on four MuJoCo tasks under position-only (P) and velocity-only (V) observability, comparing batched PPO-RTU with streaming Stream AC-RTU." style="width:100%; max-width:800px; background:#fff; padding:12px; border-radius:8px; box-sizing:border-box;">
<figcaption class="caption">Masked MuJoCo. Position-only observability on the top row, velocity-only on the bottom, across four tasks. The streaming Stream AC(&lambda;)-RTU recovers much of batched PPO's performance without any replay or batched updates.</figcaption>
</figure>

## Why I like this result

Streaming RL is appealing because it is honest about the constraints a real, online,
embodied learner faces: one stream of experience, no storing the past, update as you go. The
piece that was missing was memory. This closes that gap. An agent can now learn long-range
temporal dependencies in a single stream, without replay, by carrying an exact gradient
forward in time instead of trying to look backward through a past it never kept.

## Acknowledgements

Thanks to my coauthors Aryaman Reddi, Carlo D'Eramo, and Jan Peters. The paper and code are
linked from the references below; if you work on streaming RL, recurrent credit assignment,
or partial observability and want to compare notes, [get in touch](/).

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
