---
layout: post
title: skipping the part you already know
date: 2026-08-08 18:00:00 +0200
description: notes on modified restart distributions, what the theory says the archive should cover, and why covering more of it did not make the agent any better
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
{% cite salimans2018montezuma --file references %}. I spent a while trying to understand what
these methods are actually optimising, and then measuring whether it works. The theory turned
out to be clean. The measurements were mostly negative, in an instructive way.

## What you are choosing

Write $$\rho$$ for the environment's real initial state distribution, the thing
`env.reset()` samples. You are graded on $$J_\rho$$: expected return when you start where
the task says you start. What the archive lets you change is the distribution you actually
*train* from,

$$
\mu = (1-\lambda)\,\rho + \lambda\,\nu,
$$

where $$\nu$$ is the archive's distribution over stored states and $$\lambda$$ is how often
a reset draws from it. So $$\lambda$$ is a mixing weight and $$\nu$$ is the real design
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

So the archive should cover the states the *optimal* policy spends its time in. Not the
reachable set, not the frontier, not whatever is novel. Mass on states $$\pi^\star$$ never
visits is wasted.

Two things about this are worth holding onto.

**It is asymmetric.** The penalty comes from $$\nu$$ being *small* where
$$d^{\pi^\star}_\rho$$ is large, because you are dividing by $$\nu$$. A hole where the
optimal policy goes is catastrophic. Excess mass where it never goes merely dilutes,
linearly. You can afford to be wrong by covering too much; you cannot afford to be wrong by
covering too little. That, rather than any appeal to novelty, is the real argument for broad
coverage.

**And we do not know it.** Knowing where the optimal policy spends its time is close to
knowing how to solve the task. This is not a technical gap, it is the whole problem
restated.

Which reframes every heuristic in this literature as a *prior about where
$$\pi^\star$$ goes*:

| how you weight the archive | what it assumes |
| --- | --- |
| by visitation frequency | "the optimal policy goes where I already go" |
| uniformly over discovered states | "I know nothing" |
| by progress, evenly across stages | "it advances steadily" |
| toward the frontier | "it goes further than I have" |
| from a demonstration | you are handed the answer |

Read that way, it is not surprising that the demonstration-based method is the one that
solves Montezuma, and that the others are evaluated on how much they explore.

## The one choice that matters exponentially

Before any of the weighting subtleties, there is a cruder decision that dominates
everything: what counts as a distinct state worth storing.

If you reservoir-sample the stream of visited states, your archive is *visitation
weighted* — a time-average of the occupancies you have induced so far. That object inherits
their exponential decay in depth. If instead you key the archive by a discrete cell and keep
one representative per cell, a state visited ten thousand times occupies the same single
slot as one visited once. This is the cell mechanism from Go-Explore, and the gap between
the two is not a constant factor.

On DeepSea, a chain where the agent must take the same action $$N$$ times in a row to see
any reward, with a fixed uniform-random policy and half of resets drawn from the archive:

| depth $$N$$ | no archive | reservoir over visits | uniform over cells |
| --- | --- | --- | --- |
| 14 | 7.3e4 | 6.7e4 | **4.5e3** |
| 18 | 1.8e6 | 1.8e6 | **1.6e4** |
| 22 | 2.8e7 | over budget | **4.1e4** |
| 30 | over budget | over budget | **2.1e5** |

Median environment steps to first reach the goal, 21 seeds. Reservoir-over-visits is
indistinguishable from having no archive at all — still exponential in the horizon. Keying
by cell is polynomial. Both are "uniform sampling from an archive of visited states", and
they are separated exponentially.

There is a smaller lesson underneath. If a cell key is hashed into a
power-of-two-sized table, the modulo takes the *low* bits, so a key without good low-bit
entropy silently throws cells away. My first Craftax key lost 50.8% of distinct cells to
slot collisions; a proper avalanche brought that to 4.5%. Nothing errors, the archive just
quietly holds half of what you think it does.

## And then it did not work

With cells keyed properly, coverage improves robustly and significantly. On DeepSea, with
staggered resets in both arms so the comparison is not confounded by episode-phase
alignment {% cite bharthulwar2025staggered --file references %}, the archive reaches
substantially deeper:

| depth $$N$$ | frontier reached, no archive | with archive |
| --- | --- | --- |
| 16 | 11.50 ± 1.44 | 14.00 ± 0.41 |
| 20 | 14.40 ± 1.35 | 17.20 ± 1.00 |
| 24 | 14.70 ± 2.07 | 19.04 ± 1.57 |
| 28 | 15.80 ± 2.44 | 20.50 ± 1.25 |

Ten seeds per arm, significant at every depth, and the gap widens with the horizon. At depth
20 the effect is eight sigma.

Then I measured whether the agent got better. Annealing $$\lambda$$ to zero over training so
the final phase is an unbiased measurement of $$J_\rho$$, with forty seeds per arm:

| | solve rate | $$J_\rho$$ |
| --- | --- | --- |
| no archive | 1/40 | −0.0007 ± 0.0023 |
| archive | 2/40 | +0.0016 ± 0.0030 |

Fisher exact $$p = 0.50$$. Nothing. And at ten seeds I had 0/10 versus 5/20 and was briefly
convinced I had something.

One trap worth naming. A Mann-Whitney test on the returns comes out significant
($$p = 0.001$$), and it is significant for the wrong reason. DeepSea charges a small cost
for each step in the rewarding direction, so once the goal is never reached, return is
*minus* the amount of exploring you did. The test was ranking runs by how little they
explored. The medians give it away: both are still negative.

Craftax Classic said the same thing more gently. Ten seeds, no archive 10.92 ± 0.14
achievements, with archive 10.87 ± 0.15, $$p = 0.59$$ — and the archive arm behind at every
single epoch, catching up only as the annealing removed it. There, the mechanism was a pure
cost.

## Why, and the cheap thing I should have done first

The concentrability picture explains it quantitatively. DeepSea is small enough to compute
$$C$$ exactly by dynamic programming, because $$\pi^\star$$ is known in closed form. At
depth 20:

| $$\nu$$ | $$C(\mu)$$ | vs no archive |
| --- | --- | --- |
| no archive | 5.02e4 | 1.0x |
| visitation weighted | 4.58e4 | 1.1x |
| uniform over cells | 3.47e2 | 145x |
| stratified by progress | 1.79e2 | 280x |
| frontier tilt, $$\tau = 4$$ | **1.18e2** | **426x** |
| frontier tilt, greedy | 1.44e3 | 35x |
| oracle, $$\nu = d^{\pi^\star}_\rho$$ | 2.44e1 | 2055x |

The archive really does what the theory says: it reduces the coefficient by 426x. And
$$C \approx 10^2$$ is *still large*. Sample complexity goes like $$\sqrt{C\,\varepsilon}$$,
so being 426 times less hopeless is not the same as being tractable. What caps it is that
the archive covers 85.6% of the optimal policy's occupancy and misses the last few states —
which is exactly where all the reward lives.

The part that stings is that this table costs seconds to compute and I produced it after
about four hundred training runs. It also reproduces things those runs told me expensively:
that visitation weighting is worth nothing (1.1x), that the frontier tilt has an interior
optimum around $$\tau = 4$$, and that tilting greedily is worse than not tilting at all.
Scoring candidate restart distributions offline against a reference occupancy, before
spending any GPU time, is the obvious move and I did it last.

## What I take from this

I set out to build an exploration method and found that the quantity I was steering by and
the quantity that governs the bound point in different directions. Coverage went up by eight
sigma and $$J_\rho$$ did not move. That is worth saying out loud, because much of this
literature — including the strongest recent work on learned archive representations
{% cite gallouedec2023lge --file references %} — reports exploration metrics rather than
return.

There is also a framing I would start from if I did this again. "Skip the mastered prefix"
is a *sample efficiency* claim, not an exploration claim, and it has a target that needs no
oracle: put mass where the advantage is still nonzero, which is observable from the current
policy. It also comes with a ceiling you can measure in advance. If a fraction $$m$$ of each
episode is mastered, the most you can win is about $$1/(1-m)$$. In both settings I tested,
$$m$$ was near zero — the agent had not mastered anything yet — so there was no prefix to
skip and nothing the mechanism could have bought. I tested a good idea in the two regimes
where it has nothing to offer.

The code is a pair of small wrappers in my JAX RL library, and I am still working on this;
the first thing to fix is that the informativeness signal should come from the per-step
advantage, which is dense and already computed, rather than from the sparse witnessed
returns I used. If you work on restart distributions, curricula, or exploration and want to
compare notes, [get in touch](/).

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
