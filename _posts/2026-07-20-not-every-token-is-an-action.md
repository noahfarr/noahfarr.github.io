---
layout: post
title: not every token is an action
date: 2026-07-20 10:00:00 +0200
description: an argument for treating LLM RL as a POMDP rather than an MDP, and what falls out for free once you do
tags: reinforcement-learning llms partial-observability agents
categories: research
featured: true
---

This post is an argument rather than a result. RL on LLMs is usually described as a
Markov decision process over tokens, and I think the honest frame is a partially
observable one: output tokens are actions, prompts and tool results are observations,
and the context window is the history a POMDP policy conditions on. In one sentence: the
moment anything other than the policy can write into the transcript, you are in a POMDP,
and once you admit that, a family of standard "tricks", like masking tool outputs out of
the loss, stops being folklore and becomes forced.

The way there is a ladder of three formalisms. RLHF started as a bandit. It got refined
into a token-level MDP. That frame cracks in two distinct ways, and underneath the
cracks is the POMDP that was there the whole time. At the end I'll put the claim to a
practical test: if the frame is right, an LLM should drop into a completely standard RL
implementation with no adaptations at all. It does.

## Where RLHF starts: a bandit

The simplest honest description of single-turn RLHF is a contextual bandit. A prompt
comes in, the whole response goes out as one action from an enormous arm space, and one
scalar reward comes back:

$$
c \sim \mathcal{D}, \qquad y \sim \pi_\theta(\cdot \mid c), \qquad r = R(c, y).
$$

This is not a strawman; it is exactly how DPO was originally derived, with the entire
response treated as a single arm {% cite rafailov2024from --file references %}. And it
is fine, as far as it goes. But it flattens everything that happens inside the response:
one reward covers every token equally, whether that token mattered or not.

## One step finer: a token-level MDP

The standard refinement is to break the arm into a sequence. The state is the prefix,
the action is the next token, and the transition is concatenation:

$$
s_0 = c, \qquad a_t \sim \pi_\theta(\cdot \mid s_t), \qquad s_{t+1} = s_t \oplus a_t,
$$

with reward landing once, at the last token. This is the MDP that PPO-style RLHF
actually implements, and it is the lens under which DPO's implicit reward turns out to
be a Q-function {% cite rafailov2024from --file references %}. Note that it is Markovian
by construction: the state _is_ the entire history, so there is nothing hidden for the
policy to miss.

## The catch: two cracks

The token MDP breaks in two distinct ways, and they are worth separating, because only
one of them can be fixed with a better algorithm.

1. **It quietly collapses back into the bandit.** The sequential state only earns its
   keep if something assigns credit at the token level. Take GRPO: sample a group of
   $$G$$ responses per prompt, score each one, normalise within the group, and hand
   every token in a response the same advantage {% cite shao2024deepseekmath --file references %}:

   $$
   A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)},
   \qquad \hat{A}_{i,t} = A_i \ \text{ for every token } t.
   $$

   No critic, no per-token credit. Squint and the "token MDP" is doing exactly what the
   bandit did, judging whole arms. That is not a criticism, it is a stable and pragmatic
   choice, but it does mean the MDP is functioning as a data structure rather than as a
   decision process.

2. **It has no slot for a token the policy did not choose.** The transition
   $$s_{t+1} = s_t \oplus a_t$$ assumes every token past the prompt was sampled from
   $$\pi_\theta$$. In a single uninterrupted completion that is true. The moment a tool
   result lands in the transcript, or a user replies, it is false: those tokens were
   never drawn from any policy. The transition would need to read
   $$s_{t+1} = s_t \oplus a_t \oplus o_{t+1}$$, with $$o_{t+1}$$ arriving from outside,
   and the token MDP simply has no name for $$o_{t+1}$$.

The first crack you can patch inside the frame, by training a critic. The second is a
hole in the formalism itself, and it is the door to the POMDP.

## The idea: it was a POMDP all along

A POMDP makes the missing distinction primitive. There is a true state the agent never
sees directly: the contents of files it has not opened, the bug behind a failing test,
the user's actual intent. Acting changes that state, and observations leak it:

$$
s_{t+1} \sim T(\cdot \mid s_t, a_t), \qquad o_{t+1} \sim \Omega(\cdot \mid s_{t+1}, a_t).
$$

Because the state is hidden, no single observation is enough to act on. The classical
answer is to condition on the whole history, or a sufficient statistic of it (a belief
state), because the history is the thing that is actually Markovian
{% cite kaelbling1998planning --file references %}:

$$
h_t = (o_0, a_1, o_1, \dots, o_t), \qquad a_{t+1} \sim \pi_\theta(\cdot \mid h_t).
$$

Strictly the history carries rewards too, $$(o, a, r)$$ triples rather than pairs.
Nothing below needs them, so I leave $$r$$ out to keep the notation light.

And this is, concretely, what an LLM agent already is. Every prompt token, every tool
result, every prior turn is $$h_t$$, and attention over the full context stands in for
conditioning on history. Nobody sat down and built a belief-state module; the field
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
<figcaption class="caption" style="margin-top:8px;">One rollout of a tool-using agent. Blue spans are actions the policy sampled and should get gradient; grey spans are observations injected from outside and should not; reward lands once, at the end, and goes to the learner rather than back into the context.</figcaption>
</figure>

## What falls out for free

Once observation and action are different objects, several things that usually get
presented as engineering folklore become consequences you can read off the formalism.
They are all the same one-line fact, worn three ways: $$\pi_\theta(o \mid h)$$ is not a
thing. The policy has no distribution over tokens it did not emit.

1. **Loss masking.** There is no policy gradient to compute on a tool-output token,
   because the policy never chose it. Masking prompt, tool-result, and other-turn tokens
   out of the loss is not a stabilisation trick; it is declining to differentiate a
   quantity that was never defined. Agent-R1 builds exactly this as an explicit action
   mask separating agent-generated tokens from environment feedback, and ablates it:
   with the masks disabled, their average score drops from 0.372 to 0.302 under PPO, and
   from 0.388 to 0.372 under GRPO {% cite cheng2025agentr1 --file references %}. Same
   architecture, same data, the only difference is whether observations are treated as
   actions.
2. **KL scoping.** The penalty against a reference policy measures how far the policy's
   choices have drifted. A reference model's log-probability on an injected observation
   is not measuring drift in a choice, because there was no choice. The KL term scopes
   to action tokens for the same reason the loss does.
3. **Ratio scoping.** The PPO-style importance ratio corrects for distribution shift in
   what the policy would have sampled between updates. On a token that was never sampled
   from any policy, old or new, there is no shift to correct.

And one consequence a level up from the loss: a value or advantage estimate at an action
token has to condition on the whole history, not just the latest observation. Two
rollouts that differ in what happened ten tool calls ago but share the same last tool
output are different histories, and a value function that only reads the last turn will
alias them, for exactly the reason a POMDP says it will.

## Dropping an LLM into a standard RL stack

Here is the practical test. I keep a JAX RL codebase around with the usual suspects,
DQN, PPO, SAC, GRPO, plus recurrent variants of each for partially observable
environments. The recurrent algorithms are written against exactly the interface the
POMDP prescribes: the network takes the sequence of observations and previous actions
plus a carry, and returns action logits; the policy turns logits into a distribution;
the algorithm does not know or care what the network is.

If LLM RL really is just POMDP RL, a language model should slot straight into that
codebase. So: letter-level Wordle, a clean little POMDP where one action is one letter,
the secret word is hidden state, and the observation is all-zero except on the step that
completes a guess, so interpreting the feedback requires remembering your own past
actions. Load a pretrained Qwen3-0.6B, wrap it in LoRA, and hand it to the same
`RecurrentPPO` class that would otherwise train a GRU:

```python
network = Network(
    feature_extractor=TokenFeatureExtractor(num_actions, hidden_size),
    torso=LoRA(block=qwen3, params=pretrained, rank=8, alpha=16.0),
    head=ActorCritic(actor=nn.Dense(num_actions), critic=nn.Dense(1)),
)

algorithm = RecurrentPPO(
    cfg=cfg,
    environment=wordle,
    environment_params=params,
    network=network,  # the LLM is just the torso
    policy=policies.categorical,
    optimizer=optimizer,
)
```

Not one line of algorithm code knows an LLM is involved. And the correspondence goes all
the way down. "Tokens are actions" is literal here: the action embedding and the actor's
readout are initialised from the LLM's own token embeddings for the letters a to z. The
carry the algorithm threads through time is a GRU's hidden state in one configuration
and the transformer's KV cache in this one; the algorithm cannot tell the difference,
and it should not, because both are playing the same role, a running sufficient
statistic of the history. The critic rides on the same torso, so the value estimate
conditions on the full history for free, which is exactly what the aliasing argument
above demanded. And swap `RecurrentPPO` for `RecurrentGRPO` and nothing else changes.

Notice also what happened to the masking question: it dissolved. This implementation is
POMDP-native, so observations arrive on their own input channel and the action sequence
contains only actions. There is nothing to mask out of the loss, because observations
were never mixed into the action stream in the first place. Loss masking is the
correction you need when $$h_t$$ is stored as one flat token buffer; write the POMDP
down structurally and the bug it corrects cannot even be expressed.

<figure style="margin: 1.8rem 0;">
<style>
.stream-wrap{overflow-x:auto;}
.stream-grid{display:grid;grid-template-columns:auto repeat(6,1fr);gap:6px;align-items:stretch;font-family:monospace;font-size:0.74rem;min-width:560px;}
.lane-label{display:flex;align-items:center;justify-content:flex-end;padding:0 8px;opacity:.6;font-style:italic;}
.tok.none{background:none;border:1px dashed rgba(128,128,128,0.4);opacity:.45;font-weight:400;}
@media (max-width:640px){.stream-grid{font-size:.62rem;}}
</style>
<div class="stream-wrap">
<div class="stream-grid">
  <div class="lane-label">o</div>
  <div class="tok obs">prompt</div>
  <div class="tok none">—</div>
  <div class="tok obs">stdout:<br>3 failed</div>
  <div class="tok none">—</div>
  <div class="tok obs">tool:<br>file saved</div>
  <div class="tok obs">stdout:<br>3 passed</div>
  <div class="lane-label">a</div>
  <div class="tok act">"check the<br>test output"</div>
  <div class="tok act">bash: pytest</div>
  <div class="tok act">"off-by-one<br>in parse()"</div>
  <div class="tok act">edit parse.py</div>
  <div class="tok act">bash: pytest</div>
  <div class="tok none">stop</div>
  <div class="lane-label">r</div>
  <div class="tok none">0</div>
  <div class="tok none">0</div>
  <div class="tok none">0</div>
  <div class="tok none">0</div>
  <div class="tok none">0</div>
  <div class="tok rew">+1</div>
</div>
</div>
<figcaption class="caption" style="margin-top:8px;">The same rollout, the way a POMDP-native implementation stores it: observation, action, and reward as separate, step-aligned channels, which is literally the network's input signature. Dashes are steps where nothing came back from the environment. The loss only ever touches the action lane, not because anything is masked, but because there is nothing else it could touch.</figcaption>
</figure>

## Why I like this framing

Nothing here changes what a careful implementation already does. Good agentic RL
codebases already mask the loss, scope the KL, and condition their value estimates on
the full transcript. What the POMDP buys is that these stop being independent tricks,
each rediscovered and re-justified per codebase, and become one fact about the problem,
applied consistently: not every token is an action, and only actions get gradients. It
also buys the other direction: the moment an LLM is just a sequence model behind a POMDP
interface, everything the RL community has built for partially observable problems
applies to it verbatim, no bespoke "LLM RL" framework required. If
you think a piece of this is wrong, or you have a setting where the split does not hold
cleanly, I would genuinely like to hear about it, so [get in touch](/).

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
