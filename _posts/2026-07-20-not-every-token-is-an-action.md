---
layout: post
title: not every token is an action
date: 2026-07-20 10:00:00 +0200
description: an argument for treating LLM RL as a POMDP instead of an MDP, and what falls out for free once you do
tags: reinforcement-learning llms partial-observability agents
categories: research
featured: true
---

This one is an argument, not a result. RL on LLMs gets described as sequential
decision-making on a Markov decision process, and for a single-turn rollout that's a
defensible simplification. The moment there's a tool call or a second turn, it stops being
defensible, because the thing generating what happens next is no longer sitting inside the
token stream the model can see. That's a POMDP. And once it's written down properly, a
handful of things that usually get justified as engineering folklore — masking the loss on
tool outputs, scoping the KL penalty, scoping the importance ratio — turn out to be forced
consequences of one distinction: which tokens the policy chose, and which tokens it
didn't.

## Start with a bandit

The simplest honest description of vanilla RLHF is a contextual bandit. A prompt is drawn,
the whole response is a single action pulled from an enormous arm space, and one scalar
reward comes back:

$$
c \sim \mathcal{D}, \qquad a \sim \pi_\theta(\cdot \mid c), \qquad r = R(c, a).
$$

This is exactly how DPO was originally derived, as a bandit problem where the entire
response is treated as one arm {% cite rafailov2024from --file references %}. It's a
clean formulation, and it's also throwing away everything about what happens *inside* the
response. There is no sequential structure here at all — one reward covers every token in
the completion equally, whether the token mattered or not.

## Refine it into a token-level MDP

The standard fix is to stop treating the response as one arm and decompose it into a
sequence of token-level actions. State is the prefix, action is the next token, and the
state transition is just concatenation:

$$
s_0 = c, \qquad s_{t+1} = s_t \oplus a_t, \qquad a_t \sim \pi_\theta(\cdot \mid s_t), \qquad
r_t = \begin{cases} R(s_T) & t = T \\ 0 & \text{otherwise.} \end{cases}
$$

This is how RLHF is actually implemented under PPO, and it's the view {% cite
rafailov2024from --file references %} uses to show DPO's implicit reward is secretly a
Q-function once you stop pretending the response is atomic. It's Markovian by
construction, because $$s_t$$ *is* the entire history — the prefix holds everything that's
ever happened, so there's nothing hidden left to condition on.

## Two ways this already breaks

1. **It collapses back into a bandit.** Give the token MDP a reward that only lands at the
   last token, and don't give it a real per-token value function, and you've bought a
   sequential state you aren't using. GRPO is the clean example: sample a group of $$G$$
   completions per prompt, score each one, and normalize within the group to get a single
   scalar advantage per completion {% cite shao2024deepseekmath --file references %}:
   $$
   A_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}, \qquad
   \hat A_{i,t} = A_i \ \text{ for every token } t \text{ in completion } i.
   $$
   Every token in a completion gets the same advantage. There's no critic, no per-token
   credit assignment, nothing that the token-level state buys you over just picking a whole
   response as one arm. This isn't a criticism — it's a pragmatic, stable thing to do — but
   it's worth being honest that the "MDP" here is a data structure, not an algorithm.

2. **It has no home for a token you didn't choose.** $$s_{t+1} = s_t \oplus a_t$$ assumes
   every token past the prompt was sampled from $$\pi_\theta$$. That's true in a single
   forward pass with no tools. It stops being true the instant something else can write
   into the transcript — a tool result, another turn from a user or environment. Those
   tokens are not draws from $$\pi_\theta(\cdot \mid s_t)$$; they're just data that showed
   up. The token MDP has no formal slot for "something entered my state that I have no
   distribution over."

The first failure is a choice of algorithm, and you can fix it by using a better one. The
second isn't — it's a hole in the formalism, and no amount of clever credit assignment
patches it. That one needs an actual observation channel.

## The POMDP

A POMDP makes the split explicit. There's a true state $$s_t$$ the agent never sees
directly — the actual contents of a file, the bug that's failing a test, a user's actual
intent. The agent gets an observation instead, and it acts:

$$
s_{t+1} \sim T(\cdot \mid s_t, a_t), \qquad o_{t+1} \sim \Omega(\cdot \mid s_{t+1}, a_t),
\qquad h_t = (o_1, a_1, \dots, o_t), \qquad a_t \sim \pi_\theta(\cdot \mid h_t).
$$

Because $$s_t$$ is hidden, the policy can't condition on it, and it can't condition on
$$o_t$$ alone either — a single observation underdetermines the state. What it needs is the
whole history $$h_t$$, or a sufficient statistic of it (a belief state), since that's the
thing that's actually Markovian here {% cite kaelbling1998planning --file references %}.
Formally the history is a sequence of $$(o, a, r)$$ triples, but the policy only ever reads
$$(o, a)$$ — reward is what the learning algorithm consumes to build a training signal, not
something fed back in as an input token. I'll drop $$r$$ from $$h_t$$ from here on.

This is, concretely, what the context window already is. Every prompt token, every tool
result, every prior turn — it's all just $$h_t$$, and attention over the whole thing stands
in for conditioning on history. Nobody built a separate belief-state module; the field
backed into the right formalism by brute-force concatenation.

<figure style="margin: 1.8rem 0;">
<style>
.rollout-row{display:flex;flex-wrap:wrap;gap:6px;align-items:stretch;font-family:monospace;font-size:0.74rem;}
.tok{padding:7px 10px;border-radius:6px;display:flex;align-items:center;justify-content:center;
     text-align:center;line-height:1.3;flex:1 1 auto;min-width:88px;}
.tok.obs{background:rgba(128,128,128,0.16);opacity:.7;}
.tok.act{background:rgba(70,130,240,0.16);border:1px solid rgba(70,130,240,0.55);font-weight:600;}
.tok.rew{background:rgba(240,160,50,0.2);border:1px solid rgba(240,160,50,0.6);font-weight:600;}
.tok sub{opacity:.65;font-weight:400;}
@media (max-width:640px){.rollout-row{font-size:.66rem;}.tok{min-width:70px;}}
</style>
<div class="rollout-row">
  <div class="tok obs">prompt<br><sub>o₀</sub></div>
  <div class="tok act">"check the<br>test output"<br><sub>a₁</sub></div>
  <div class="tok act">bash: pytest<br><sub>a₂</sub></div>
  <div class="tok obs">stdout:<br>3 failed<br><sub>o₁</sub></div>
  <div class="tok act">"off-by-one<br>in parse()"<br><sub>a₃</sub></div>
  <div class="tok act">edit parse.py<br><sub>a₄</sub></div>
  <div class="tok obs">tool:<br>file saved<br><sub>o₂</sub></div>
  <div class="tok act">bash: pytest<br><sub>a₅</sub></div>
  <div class="tok obs">stdout:<br>3 passed<br><sub>o₃</sub></div>
  <div class="tok rew">reward<br><sub>r</sub></div>
</div>
<figcaption class="caption" style="margin-top:8px;">One rollout of a tool-using agent. Blue spans are actions the policy actually sampled and should get gradient; grey spans are observations injected from outside and shouldn't; reward lands once, at the end, and is consumed by the learning algorithm rather than read back in as a token.</figcaption>
</figure>

## What falls out for free

Once $$o_t$$ and $$a_t$$ are formally distinct, a few things that get presented as
implementation tricks turn out to be forced moves.

1. **Loss masking.** $$\pi_\theta(o_{t+1} \mid h_t)$$ isn't a quantity that means anything —
   the policy didn't choose $$o_{t+1}$$, so there's no action distribution to
   differentiate. Masking the loss on prompt, tool-output, and other-turn tokens isn't a
   hack bolted onto training; it's declining to compute a gradient for something that was
   never defined in the first place. Agent-R1 builds exactly this as an explicit "action
   mask" that "clearly delineate[s] the tokens generated by the LLM agent (its actions)
   from the environmental feedback or the initial prompt," and ablates it directly: with
   both the loss mask and the advantage mask disabled, their average score drops from
   0.372 to 0.302 under PPO, an 18.7% fall, and from 0.388 to 0.372 under GRPO with just
   the loss mask off {% cite cheng2025agentr1 --file references %}. That's not a stability
   trick paying off, it's the cost of training on a signal that was ill-defined to begin
   with.
2. **KL scoping.** The penalty against a reference policy is supposed to measure how far
   the policy's own choices have drifted. A reference model's log-probability on an
   injected observation token isn't measuring drift in a choice, because it was never a
   choice. Scope the KL term to action tokens for the same reason you scope the loss.
3. **Ratio scoping.** The PPO/GRPO importance ratio $$\pi_\theta(a_t \mid h_t) /
   \pi_{\theta_\text{old}}(a_t \mid h_t)$$ corrects for distribution shift in what the
   policy would have sampled between updates. There's no such shift to correct on a token
   that was never sampled from any policy in the first place — old or new.

All three are the same one-line fact, applied at three different places in the objective:
the loss, the KL term, the clipping ratio. None of them are independent tricks you'd need
to separately rediscover; they're one distinction wearing three hats.

There's a fourth consequence a level up from the loss rather than inside it: a value or
advantage estimate at an action token needs to condition on the *whole* history up to that
point, not just the most recent observation. Two rollouts that differ in what happened ten
tool calls ago but happen to share the same last observation are not the same state — and
if your value function only looks at the latest turn, it will alias them, for exactly the
reason a POMDP says it should.

None of this changes what careful agentic RL implementations already do — good ones
already mask, already scope, already condition value estimates on the full transcript.
What changes is whether that's folklore independently rediscovered per codebase, or one
fact about the problem, applied consistently. If you think I've got a piece of this wrong,
or you're running into a place where the split doesn't hold cleanly, I'd like to hear
about it — [get in touch](/).

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
