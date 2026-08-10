---
layout: post
title: skipping the part you already know
date: 2026-08-08 18:00:00 +0200
description: what a modified restart distribution is actually optimising, why the answer involves the optimal policy, and the two different things you might be asking it for
tags: reinforcement-learning exploration sample-efficiency
categories: research
featured: true
---

In a long-horizon task, an agent spends most of every episode redoing things it already
knows how to do. In Craftax, once you can reliably collect wood, place a table, and craft a
pickaxe, you still walk through all of that from scratch on every reset before you reach the
part you are actually still learning. That looks like waste, and it suggests an obvious fix:
if you can reset the simulator to any state you have already seen, stop starting from the
beginning. Restart the agent near the edge of what it knows.

This is an old idea. Go-Explore builds an archive of visited states and returns to them
{% cite ecoffet2021goexplore --file references %}, Tavakoli and colleagues adapt the restart
distribution from an experience memory {% cite tavakoli2018restart --file references %}, and
Salimans and Chen solve Montezuma's Revenge by restarting along a single demonstration
{% cite salimans2018montezuma --file references %}. I wanted to know what these methods are
actually optimising, because once you write it down it tells you which of the many available
heuristics are principled and which are guesses. This post is about the formulation. The
measurements are still in progress and can wait for their own writeup.

> **Update (2026-08-10).** [The measurements are in](/blog/2026/the-part-you-already-know-measured/),
> and they overturned a recommendation this post makes: the gradient-magnitude target of the
> exploitation section is statistically indistinguishable from ignoring the score
> entirely, for a reason that is now a theorem, while the witnessed-gap target more than
> doubles the baseline and beats the oracle quantity it estimates. The formulation below
> stands; my advice about which factor to trust did not.

## What you are choosing

Write $$\rho$$ for the environment's real initial state distribution, the thing
`env.reset()` samples. You are graded on $$J_\rho$$: expected return when you start where
the task says you start. What the archive lets you change is the distribution you actually
_train_ from,

$$
\mu = (1-\lambda)\,\rho + \lambda\,\nu,
$$

where $$\nu$$ is the archive's distribution over stored states and $$\lambda$$ is how often
a reset draws from it. So $$\lambda$$ is a mixing weight, and $$\nu$$ is the real design
question: **which states should the archive put mass on?**

The tension is visible in the notation. You train on $$\mu$$ and you are graded on
$$\rho$$. Tavakoli and colleagues already flagged the consequence: changing the initial
distribution can change which policy is optimal, once you are searching inside a restricted
policy class rather than over all policies.

## Two questions, not one

Before deriving anything it is worth separating two things that "restart somewhere better"
can mean, because they are different objectives and they want different distributions.

Policy-gradient error decomposes, loosely, into an optimisation term and a statistical term
scaled by a concentrability coefficient. The restart distribution touches both.

**Exploration.** Can the optimal policy be learned at all? This is the statistical term, and
it asks whether $$\nu$$ reaches the states $$\pi^\star$$ relies on. It is the only one of the
two that can put mass somewhere you have never been.

**Exploitation.** Are the samples you collect worth their cost? This is the optimisation
term, and it asks whether $$\nu$$ avoids regions the policy has already mastered, where the
advantage is zero and the gradient carries nothing, and spends its budget where learning is
still live.

They are not the same question, and the asymmetry between them is the most useful thing in
this post: the exploration answer is cleaner to derive but needs something you cannot have,
while the exploitation answer is less elegant and entirely observable. I derive the clean
one first, which is a fact about exposition rather than about which matters more.

## The exploration answer

For the coverage question there is an exact answer, and it falls out of the concentrability
coefficient that governs policy-gradient sample complexity
{% cite kakade2002approximately --file references %}. Using
$$d^\pi_\mu \ge (1-\gamma)\mu$$ pointwise, the coefficient is bounded by

$$
C \;\le\; \frac{1}{(1-\gamma)\lambda}\sum_s \frac{d^{\pi^\star}_\rho(s)^2}{\nu(s)},
$$

where $$d^{\pi^\star}_\rho$$ is the discounted state distribution of the optimal policy.
Minimising that sum subject to $$\sum_s \nu(s) = 1$$ is one line of Lagrange, and the
minimiser is

$$
\nu^\star \propto d^{\pi^\star}_\rho .
$$

So the archive should cover the states the _optimal_ policy spends its time in. Not the
reachable set, not whatever is novel. Mass on states $$\pi^\star$$ never visits is wasted.

Two features of that objective matter more than the formula.

**It is asymmetric.** The penalty comes from $$\nu$$ being _small_ where
$$d^{\pi^\star}_\rho$$ is large, because you are dividing by $$\nu$$. A hole where the
optimal policy goes is catastrophic. Excess mass where it never goes only dilutes, and it
dilutes linearly. You can afford to be wrong by covering too much; you cannot afford to be
wrong by covering too little. So the case for broad coverage does not rest on novelty being
intrinsically valuable. It rests on holes being punished harder than waste.

**And we do not know it.** Knowing where the optimal policy spends its time is close to
knowing how to solve the task. This is not a technical gap, it is the whole problem
restated.

Which reframes every heuristic in this literature as a _prior about where $$\pi^\star$$
goes_:

| how you weight the archive        | what it assumes about $$\pi^\star$$            |
| :-------------------------------- | :--------------------------------------------- |
| by visitation frequency           | _"it goes where I already go"_                 |
| uniformly over discovered states  | _"I know nothing"_                             |
| by progress, evenly across stages | _"it advances steadily"_                       |
| toward the frontier               | _"it goes further than I have"_                |
| from a demonstration              | you are handed the answer                      |
{: .table .table-sm }

Read that way, it is not surprising that the demonstration-based methods are the ones that
solve the hardest exploration problems, and that the others tend to be evaluated on how much
they explore.

## The exploitation answer

The other question has no such closed form, and needs no oracle either.

The gradient contribution of a state is proportional to the size of the advantage there, so
a region the policy has already mastered costs budget and produces nothing. Skipping it is
the original motivation for this whole idea, and the target is

$$
\nu \;\propto\; d^{\pi}_\rho(s)\,\lVert g(s) \rVert ,
$$

the current policy's occupancy weighted by gradient magnitude. Every quantity in that is
observable. Your rollouts sample $$d^\pi_\rho$$; the advantages are already computed during
the update; and if you want something cheaper, the entropy of $$\pi(\cdot \mid s)$$ is a
serviceable stand-in, since a policy that has committed at a state is a policy with nothing
left to learn there. No oracle, no circularity.

It also comes with a ceiling you can measure before building anything. If a fraction $$m$$
of each episode is already mastered, the most this can win you is about $$1/(1-m)$$. Worth
computing first, because if $$m$$ is near zero there is no prefix to skip and nothing on
offer, and you will spend a long time discovering that the expensive way.

The catch is the mirror image of the exploration answer's. Gradient magnitude is undefined at
states you have never visited, so this objective can reweight the support you already have
but can never tell you the archive is missing something. Exploration expands; exploitation
allocates.

The two agree about the frontier and disagree about the prefix, and the disagreement is free:
$$\rho$$ already covers the prefix by definition, so the archive never needed mass there.

If you want _both_, the performance difference lemma gives the combined target directly.
Since

$$
J_\rho(\pi^\star) - J_\rho(\pi) = \frac{1}{1-\gamma}\,
\mathbb{E}_{s \sim d^{\pi^\star}_\rho}\,\mathbb{E}_{a \sim \pi^\star}\big[A^\pi(s,a)\big],
$$

the quantity you are driving to zero is an integral over $$d^{\pi^\star}_\rho$$ weighted by
how much better $$\pi^\star$$ is than you at each state. The variance-minimising proposal for
that integral is the product,

$$
\nu^\star \;\propto\; d^{\pi^\star}_\rho(s)\,
\big\lvert \mathbb{E}_{a \sim \pi^\star}[A^\pi(s,a)] \big\rvert ,
$$

which is zero on the mastered prefix because the second factor vanishes, zero off
$$\pi^\star$$'s path because the first does, and large exactly at the frontier. So "restart
near the edge of what you know" is something you can derive rather than assert.

It also makes something explicit that is easy to miss. The combined target contains
$$\pi^\star$$ **twice**. Of the three objectives, only the exploitation target is genuinely
oracle-free.

## How to handle a $$\pi^\star$$ you do not know

The standard answer in the theory is to assume it away: state that the concentrability
coefficient is bounded and proceed. That is fine until the coefficient is enormous, which is
exactly the case on the problems this mechanism is meant for, and then the bound says
nothing.

There are three honest alternatives.

**Be minimax.** If you do not know $$\pi^\star$$, minimise the worst case over all policies
rather than guessing. That is a well-posed problem, needs no oracle, and the object it
produces is the distribution achieving the _coverability_ coefficient. It also explains why
uniform-over-discovered-states is a reasonable default: it approximates the minimax solution,
which has to cover every policy's occupancy at once.

**Use sound bounds.** The advantage-gap factor admits a one-sided estimate. The best return
you have actually _witnessed_ from a state is a lower bound on its optimal value in
deterministic dynamics, so the gap you compute from it can only understate the true gap. You
never chase a phantom, because you act only on witnessed improvement.

**Or avoid it.** The exploitation target above involves no $$\pi^\star$$ at all. It is
the part of this that is unconditionally defensible, and it comes with a ceiling you can
measure in advance: if a fraction $$m$$ of each episode is already mastered, the most you can
possibly win is about $$1/(1-m)$$. Worth computing before building anything, because if
$$m$$ is near zero there is no prefix to skip and nothing on offer.

## What the archive actually lets you choose

Everything above treats $$\nu$$ as a free variable. It is not. $$\nu$$ is realised by an
archive of stored states, and the rule you use to decide what goes in it already fixes a
distribution, whether or not you meant to choose one. Before you get to weight the archive,
the archive has weighted itself.

So it is worth asking what measure you get by default, and how far you can move away from
it. This is mostly a question about the exploration channel; the exploitation channel needs
much less from the archive, which turns out to matter for how simple a usable version can
be.

If you reservoir-sample the stream of visited states, your archive is _visitation weighted_.
It approximates $$d^\pi_\rho$$, the current policy's own occupancy. That sounds useless and
is not. Restarting from a state drawn part-way through an episode and then continuing pushes
the composed occupancy later in the episode, since the time weighting goes from
$$\gamma^n$$ to $$(n+1)\gamma^n$$. So visitation weighting does skip the prefix, which is
what you wanted. What it cannot do is change the _support_. Every state it can restart from
is one the policy already reaches, so it discovers nothing.

The distinction matters because the deficit it fails to fix is the one that bites. In a
hard-exploration problem the mass at depth $$c$$ decays exponentially in $$c$$, and shifting
the time weighting by a polynomial factor does not touch that. Visitation weighting is a
no-op for exploration and useful for exploitation.

To get a free choice of $$\nu$$ you have to break that coupling, and the way to do it is to
key the archive by a discrete cell and keep one representative per cell. A state visited ten
thousand times then occupies the same single slot as one visited once, so visitation stops
determining where you restart and you can impose whatever measure you actually wanted. This
is the cell mechanism from Go-Explore, and on the coverage channel the difference between
having it and not is not a constant factor.

Even then the freedom is not complete, because uniform-over-cells is not
uniform-over-_progress_. If deep states are reachable from fewer predecessors than shallow
ones, a flat measure over cells hands out mass in proportion to how many cells each depth
happens to contain, and starves the frontier. Equalising across a progress variable fixes
it, and needs nothing beyond a scalar notion of progress.

There is a smaller trap below that. If a cell key is hashed into a power-of-two-sized table,
the modulo takes the _low_ bits, so a key without good low-bit entropy throws cells away.
Nothing errors. The archive holds a fraction of what you think it does, and every downstream
diagnostic looks plausible.

The code is a pair of small wrappers in my JAX RL library. If you work on restart
distributions, curricula, or exploration and want to compare notes,
[get in touch](/).

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
