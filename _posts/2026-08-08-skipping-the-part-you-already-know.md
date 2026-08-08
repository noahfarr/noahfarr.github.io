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
{% cite salimans2018montezuma --file references %}. I have been trying to work out what these
methods are actually optimising, because the answer turns out to be unusually clean, and it
tells you which of the many available heuristics are principled and which are guesses. This
post is about the formulation. The measurements are still in progress and can wait for their
own writeup.

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

## What $$\nu$$ should be

There is an exact answer, and it falls out of the concentrability coefficient that governs
policy-gradient sample complexity {% cite kakade2002approximately --file references %}. Using
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

Two things about this are worth holding onto.

**It is asymmetric.** The penalty comes from $$\nu$$ being _small_ where
$$d^{\pi^\star}_\rho$$ is large, because you are dividing by $$\nu$$. A hole where the
optimal policy goes is catastrophic. Excess mass where it never goes merely dilutes,
linearly. You can afford to be wrong by covering too much; you cannot afford to be wrong by
covering too little. That, rather than any appeal to novelty, is the real argument for broad
coverage.

**And we do not know it.** Knowing where the optimal policy spends its time is close to
knowing how to solve the task. This is not a technical gap, it is the whole problem
restated.

Which reframes every heuristic in this literature as a _prior about where $$\pi^\star$$
goes_:

| how you weight the archive        | what it assumes                              |
| --------------------------------- | -------------------------------------------- |
| by visitation frequency           | "the optimal policy goes where I already go" |
| uniformly over discovered states  | "I know nothing"                             |
| by progress, evenly across stages | "it advances steadily"                       |
| toward the frontier               | "it goes further than I have"                |
| from a demonstration              | you are handed the answer                    |

Read that way, it is not surprising that the demonstration-based methods are the ones that
solve the hardest exploration problems, and that the others tend to be evaluated on how much
they explore.

## Two different things you might be asking for

This is the distinction I found most clarifying, and it is easy to blur.

**Exploration.** You want the archive to reach where $$\pi^\star$$ goes, so that the optimal
policy is learnable at all. The target is $$d^{\pi^\star}_\rho$$, it needs an oracle you do
not have, and it is the only one of the two that can put mass somewhere you have never been.

**Sample efficiency.** You want the samples you collect to be _informative_. The gradient
contribution of a state is proportional to the size of the advantage there, so a region the
policy has already mastered costs budget and produces nothing. The target is

$$
\nu \;\propto\; d^{\pi}_\rho(s)\,\lVert g(s) \rVert ,
$$

the current policy's occupancy weighted by gradient magnitude. Every quantity in that is
observable: your rollouts sample $$d^\pi_\rho$$, and the advantages are already computed
during the update. No oracle, no circularity. The catch is that it can only reweight the
support you already have — $$\lVert g(s) \rVert$$ is not defined at states you have never
visited, so it can never tell you the archive is missing something.

They agree about the frontier and disagree about the prefix, and the disagreement is free:
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
$$\pi^\star$$'s path because the first does, and large exactly at the frontier. That is a
nice derivation of "restart near the edge of what you know" rather than an assertion of it.

It also makes something explicit that is easy to miss: the combined target contains
$$\pi^\star$$ **twice**. The sample-efficiency objective is the only one of the three that is
genuinely oracle-free.

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
which has to cover every policy's occupancy at once. Not a shrug, an approximation to
something specific.

**Use sound bounds.** The advantage-gap factor admits a one-sided estimate. The best return
you have actually _witnessed_ from a state is a lower bound on its optimal value in
deterministic dynamics, so the gap you compute from it can only understate the true gap. You
never chase a phantom; you act only on witnessed improvement.

**Or avoid it.** The sample-efficiency target above involves no $$\pi^\star$$ at all. It is
the part of this that is unconditionally defensible, and it comes with a ceiling you can
measure in advance: if a fraction $$m$$ of each episode is already mastered, the most you can
possibly win is about $$1/(1-m)$$. Worth computing before building anything, because if
$$m$$ is near zero there is no prefix to skip and nothing on offer.

## Cells, not visits

One implementation choice dominates all of the weighting subtleties: what counts as a
distinct state worth storing.

If you reservoir-sample the stream of visited states, your archive is _visitation weighted_,
a time-average of the occupancies you have induced so far. That object inherits their decay
in depth, which in a hard-exploration problem is exponential — so the archive concentrates
overwhelmingly on the shallow states you were already going to see. If instead you key the
archive by a discrete cell and keep one representative per cell, a state visited ten
thousand times occupies the same single slot as one visited once, and the weighting collapses
by construction. This is the cell mechanism from Go-Explore, and the difference between the
two is not a constant factor.

There is a smaller trap underneath. If a cell key is hashed into a power-of-two-sized table,
the modulo takes the _low_ bits, so a key without good low-bit entropy silently throws cells
away. Nothing errors; the archive just quietly holds a fraction of what you think it does,
and every downstream diagnostic looks plausible.

## What is actually new here

Very little of the above, and that is worth stating plainly. That $$\nu^\star \propto
d^{\pi^\star}_\rho$$ is a line of Lagrange applied to a twenty-year-old bound. Cell-keyed
archives are Go-Explore. Adaptive restart distributions from an experience memory are
Tavakoli and colleagues, including the caveat about restricted policy classes. Replacing
hand-designed cells with a learned latent representation is already done and works
{% cite gallouedec2023lge --file references %}.

What I have not found is anyone connecting the two halves — using the concentrability
coefficient as a _design tool_ for the archive, rather than as an assumption in a proof. The
practical line picks restart weightings by heuristic; the theoretical line treats the restart
distribution as given. Scoring candidate distributions against a reference occupancy, before
spending any training compute, sits in between, and it makes several of the choices above
decidable rather than a matter of taste.

The other gap is a criterion for the cells. Latent Go-Explore says its method combines with
_any_ strategy for learning a representation, and leaves open which one you should want. The
answer implied by the formulation is a value-preserving abstraction: cells should merge states
with equal optimal value, which is the classical condition for state aggregation, and the
within-cell spread of value is a measurable check on whether yours do.

## Where I would start

If I were beginning this again, I would start from the sample-efficiency framing rather than
the exploration one. It has an observable target, it needs no oracle, and it comes with a
ceiling you can measure from a single control run instead of discovering empirically after
the fact. Exploration then enters as a support-expansion prior on top, which is honest about
being a guess.

And I would compute the coefficient before running anything. It is the difference between
choosing $$\nu$$ by argument and choosing it by sweep.

The code is a pair of small wrappers in my JAX RL library. If you work on restart
distributions, curricula, or exploration and want to compare notes,
[get in touch](/).

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
