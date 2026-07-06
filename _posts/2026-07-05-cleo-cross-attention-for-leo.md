---
layout: post
title: letting every goal ask its own question
date: 2026-07-05 12:00:00 +0200
description: a small extension to Michael Matthews' LEO that fixes its fidelity with a cross-attention readout
tags: reinforcement-learning goal-conditioned
categories: research
featured: true
---

This post is meant to serve as an accessible writeup of a small extension I've been
building on top of [**LEO: Learning Everything all at Once**](https://www.mtmatthews.com/blog/leo/) {% cite matthews2026leo --file references %},
by Michael Matthews and collaborators at FLAIR. Michael already wrote a
[wonderful, accessible post](https://www.mtmatthews.com/blog/leo/) on LEO, and this is
very much a sequel to it. I'll recap enough to stand alone, but read his first.

I take LEO's all-goals value network and swap its readout for a cross-attention layer. I
show, or rather hope to show, that this recovers the value fidelity LEO gives up, without
needing a second network to do it. I've been calling it **CLEO**.

## Primer: LEO

Goal-conditioned RL learns a value function $$Q(s, g)$$ over states *and* goals. To wring
more signal out of each transition we relabel trajectories with goals the agent actually
reached (hindsight relabelling). The dream is to relabel every transition with *every*
goal, but for a goal set of size $$|G|$$ that's $$|G|$$ passes per transition. That clearly
does not scale. Can we get all-goals learning without the all-goals price tag?

<figure style="margin: 1.6rem auto; text-align: center;">
<img src="{{ '/assets/img/leo/maze.gif' | relative_url }}" alt="A goal-conditioned agent navigating a maze to commanded goals." style="width:100%; max-width:460px; height:auto; border-radius:10px;">
<figcaption class="caption">A goal-conditioned agent commanded to reach goals in a maze. <span style="opacity:.7">Animation &#169; Michael Matthews, from the <a href="https://www.mtmatthews.com/blog/leo/">LEO post</a>.</span></figcaption>
</figure>

LEO's answer is to change the shape of the network. Instead of the goal being an *input*,
it moves into the *output*, so the reparameterisation goes from

$$
Q(s, g) \rightarrow \mathbb{R}
\qquad\text{to}\qquad
Q(s) \rightarrow \mathbb{R}^{|G| \times |A|}.
$$

Feed in one state and you get back a whole table of values, one row per goal, one column
per action, in a single forward pass. Learning against all goals is then a *single
backward pass*, which with 512 goals works out around 250× cheaper than actually
relabelling against all of them. All the reach of all-goals learning, almost none of the
bill. Great!

<figure style="margin: 1.8rem 0;">
<style>
.explainer-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;align-items:start;}
.explainer-row video{width:100%;height:auto;border-radius:8px;display:block;}
@media (max-width:640px){.explainer-row{grid-template-columns:repeat(2,1fr);}}
</style>
<div class="explainer-row">
  <video src="{{ '/assets/img/leo/part2a.mp4' | relative_url }}" autoplay loop muted playsinline></video>
  <video src="{{ '/assets/img/leo/part2b.mp4' | relative_url }}" autoplay loop muted playsinline></video>
  <video src="{{ '/assets/img/leo/all_goals_uvfa.mp4' | relative_url }}" autoplay loop muted playsinline></video>
  <video src="{{ '/assets/img/leo/part2c.mp4' | relative_url }}" autoplay loop muted playsinline></video>
</div>
<figcaption class="caption" style="margin-top:8px; text-align:center;">The learning update under four schemes, ending in LEO's all-goals network (each panel is labelled in the animation). <span style="opacity:.7">Animations &#169; Michael Matthews, from the <a href="https://www.mtmatthews.com/blog/leo/">LEO post</a>.</span></figcaption>
</figure>

## Why does LEO lose fidelity?

On Craftax, LEO does something no baseline does: it gets *non-zero* success on the hard,
long-horizon goals (like descending to the first dungeon level). But when you look at the
per-goal rates, something strange shows up. On the *easy* goals, where every UVFA
baseline sails to nearly 100%, LEO plateaus around 80%. Why is this?

<figure style="margin: 1.6rem auto; text-align: center;">
<video src="{{ '/assets/img/leo/craftax_dlvl1.mp4' | relative_url }}" autoplay loop muted playsinline style="width:100%; max-width:260px; height:auto; border-radius:10px;"></video>
<figcaption class="caption">LEO commanded a hard goal in Craftax: reach dungeon level 1, which the baselines never solve. <span style="opacity:.7">Animation &#169; Michael Matthews, from the <a href="https://www.mtmatthews.com/blog/leo/">LEO project</a> (<a href="https://github.com/MichaelTMatthews/purejaxgcrl">purejaxgcrl</a>, MIT); converted to mp4.</span></figcaption>
</figure>

The answer, as Michael diagnoses it, is really two answers:

1. **The shared, unconditioned trunk is an information bottleneck.** At action time the
   trunk is never told which goal is in play, so its fixed-width representation has to be
   good enough for *all* of them simultaneously. Capacity that a single-goal network could
   spend on one goal is spread thin across the whole set.
2. **State and goal are combined by late fusion.** Every goal's values come off one shared
   embedding via a single linear layer, so state and goal never really interact until that
   last step. It is about the shallowest way the two could be fused.

Michael has a memorable way of putting this: LEO is a *"coarse-grained data sponge"*, one
that absorbs all the information it can from every transition but does not wring it back
out at high fidelity. His fix, Dual LEO, keeps the sponge and adds a conventional UVFA
network as a student that learns from what the sponge discovers. It works beautifully. But
it does mean carrying a second network, and I wondered whether we could fix the sponge
itself.

## Learning Everything all at Once with Cross-Attention

Here's the thing both limitations have in common: the goal shows up too late and too
weakly. So what if, instead of splitting one shared vector with a linear layer, we let
each goal *reach into* the state and ask its own question, pulling out exactly the
features it needs? I've been calling this **CLEO** (Cross-attention LEO).

Concretely: the trunk stops producing a single vector and instead produces a small set of
state *tokens*. Each goal's embedding becomes a *query*, and one cross-attention layer
lets every goal attend over the state tokens:

```python
# LEO: all goals share one vector, read out by a single linear layer
x = trunk(obs)
q = nn.Dense(num_goals * num_actions)(x)

# CLEO: the state becomes tokens; each goal queries them via cross-attention
tokens  = to_tokens(trunk(obs))
queries = embed(goal_embeddings)
context = CrossAttention(d_model, num_heads)(queries, tokens)
queries = queries + context
q       = nn.Dense(num_actions)(queries)
```

<figure style="margin: 2rem 0; text-align: center;">
{% include leo_cleo_transformer.svg %}
<figcaption class="caption">LEO and CLEO. LEO reads every goal off one shared vector through a single linear layer (late fusion). CLEO is a decoder: the state becomes a set of tokens serving as keys and values, and each goal is a query that attends to them (early fusion).</figcaption>
</figure>

From the outside the shapes are identical. One pass still gives us
$$\mathbb{R}^{|G| \times |A|}$$, so we keep the sponge's efficiency. But the single change
goes after *both* limitations at once:

1. It **widens the bottleneck**: the state is now a set of tokens acting as a
   goal-addressable memory, not one vector every goal has to fight over.
2. It **trades late fusion for early(er) fusion**: goal and state interact through
   attention *before* the readout, so each goal shapes its own representation instead of
   indexing a frozen one.

Note that this sits neatly between LEO (goal enters last, late fusion) and a full UVFA
(goal conditions the whole trunk, early fusion but no all-goals reuse): the expensive
trunk still runs *once*, and only the cheap per-goal attention is goal-specific. The hope
is to buy UVFA's fidelity at LEO's price. The name is a placeholder I've grown fond of,
but the mechanism is the point here, not the acronym.

## Does it actually help?

*Coming soon.*

## When would CLEO help?

CLEO inherits LEO's preconditions and adds one of its own. It might be worth trying if:

- your problem is a goal-conditioned MDP with a finite goal set small enough to be an
  output layer;
- querying the goal-conditioned reward is cheap (so relabelling is free-ish);
- and, the new part, you're paying for LEO's efficiency but feeling its fidelity ceiling,
  and you'd rather fix the network than bolt a second one alongside it.

## What next?

A few threads I want to pull:

- **Is CLEO a rival to Dual LEO, or a component of it?** Dual LEO fixes fidelity with a
  student; CLEO fixes it in the readout. A Dual setup with a CLEO *teacher* is the obvious
  next experiment. Do the gains stack, or overlap?
- **How far does the token/goal attention scale?** Where does the number of state tokens
  stop paying for itself as the goal count grows?
- **Does higher-fidelity value help curricula?** autotelix already has a learning-progress
  goal generator; sharper per-goal values might make automatic curricula sharper too.

## Acknowledgements

Enormous thanks to Michael Matthews. Both the LEO paper and that blog post are the kind
of clear, generous writeup that makes it easy for someone else to pick up a thread and
pull. Almost everything here is standing on that work. If you're working on
goal-conditioned RL and want to compare notes, [get in touch](/).

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
