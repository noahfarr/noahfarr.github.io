---
layout: post
title: the part you already know, measured
date: 2026-08-10 22:00:00 +0200
description: the restart-distribution formulation meets a preregistered experiment, and the arm I recommended last time loses to the one I could prove things about
tags: reinforcement-learning exploration sample-efficiency
categories: research
featured: true
---

[Last time](/blog/2026/skipping-the-part-you-already-know/) I wrote down what a modified
restart distribution is optimising and promised the measurements their own writeup. This is
that writeup, and it opens with a correction: the arm I called the unconditionally defensible
one is the arm that lost.

## The taxonomy that replaced the two questions

The exploration-versus-exploitation split survived contact with the theory, but the useful
axis turned out to be a different one. Every practical score for "how much is a restart at
$$s$$ worth" falls into one of two classes.

**Visible scores** are the signals the current policy generates at $$s$$: absolute temporal
difference error, policy gradient magnitude, entropy. My last post recommended these because
every quantity in them is observable. That is true, and it is also the problem. The expected
advantage under your own policy is identically zero, so $$\lvert\delta\rvert$$ measures
critic error, not improvement. And the policy-gradient block at $$s$$ is bounded by
$$2\,\epsilon_s\, d^{\pi}_{\mu}(s)\, R_s/(1-\gamma)$$ with
$$\epsilon_s = 1 - \max_a \pi(a\mid s)$$, so it vanishes wherever the policy is _confidently
wrong_. A restart can repair the occupancy factor. It cannot repair the confidence factor. At
exactly the states restarts exist for, states where the agent is sure of a bad action, every
visible score reads zero.

**Latent scores** reference history instead of the policy's expectations. The best return you
have witnessed from a state is, under deterministic dynamics, a certified lower bound on its
optimal value, so the _witnessed gap_ $$\Delta(s) = (G^{\max}(s) - V^{\pi}(s))^{+}$$ never
overstates the improvement actually available. It sees exactly what the visible scores are
blind to, and it is blind exactly where they see, which is the regime where a greedy price is
the right price anyway.

## What the preregistered screen said

On a diagnostic environment built so the factors separate (a mastered corridor followed by a
combination lock), with 36 shared seeds, an analysis plan committed before any result was
retrieved, and paired McNemar tests:

- The unmodified baseline solves 16/36. Uniform coverage over the archive: 19/36, no
  separation.
- Temporal difference error: 17/36. Gradient magnitude: 22/36. Neither separates from
  uniform coverage. The gradient arm was implemented _after_ the vanishing result was proved,
  so its tie is a prediction, not a postdiction.
- The witnessed gap: **33/36**. Against the control that is 19 discordant seeds to 2.
- A 10% uniform floor mixed into the gap measure costs nothing, 33/36 with the cleanest
  contrast in the study, 17 discordant to 0.

So the observable arm I recommended performs like the measure that ignores its score
entirely, and the arm that needs a soundness lemma more than doubles the baseline. I find
the prediction-before-measurement part genuinely satisfying, and I would find it less
satisfying if the predicted-to-fail arm had been someone else's recommendation.

## The oracle twist

The screen contained arms I did not write about last time, because they require cheating.
Replace the estimated quantities with exact ones, computed by dynamic programming on the
environment: the true remaining improvement $$V^\star - V^{\pi}$$, and the true optimal
occupancy $$d^{\pi^\star}_{\rho}$$.

The exact improvement, used as a weight, solves 27/36. The witnessed estimator beats the
oracle quantity it estimates, 90% interval on the paired difference entirely on the
estimator's side. The reason is that the oracle is two-sided: $$V^\star - V^{\pi}$$ is large
wherever improvement is _possible_, demonstrated or not, so its mass funds restarts from
which the policy has never once succeeded. The witnessed gap funds consolidation, and
consolidation is what restarts are good at.

The exact optimal occupancy is worse. Multiplied into the target it lands at 16/36, the
control's level, erasing the whole gain. It zeroes every state that has wasted a step, and a
doomed state still teaches the remaining rungs. Last time I framed every heuristic as a prior
about where $$\pi^\star$$ goes. The experiment sharpened that: handing the archive the exact
answer to that question is not merely unhelpful, it is destructive, because where the optimal
policy goes and where practice pays are different questions.

## A preregistration story

The first version of this batch produced a beautiful, wrong result. A cell refinement
splitting states by wasted steps came out as the best arm in the study. It turned out every
environment in the batch reset in lockstep, the gcd of rollout length and episode horizon
was 2, and only half the cells were ever archived; the refined cells happened to
disambiguate the aliasing. Staggering episode starts removed the artifact and the best arm's
advantage with it. The preregistered rerun demoted it to a tie, and the earlier batch is
labeled exploratory in the paper. If you run massively parallel environments with synchronised
resets, check your gcds.

## Where the mechanism stops

On DeepSea the archive changes nothing at any size, and the null is the point: discovery
there is a needle no witnessed return precedes, and an archive of visited states cannot
manufacture coverage it has not earned. Restarts consolidate what has been found. Where
nothing has been found, or nothing is left to consolidate, the formulation itself says to
expect nothing, and that is what we measure. A boundary an idea predicts for itself is worth
two benchmarks it wins.

The remaining question is the one the first post opened with, Craftax, and whether
consolidating the frontier pays at a billion steps against a baseline anchored to the public
leaderboard. Those runs are on the cluster as this posts. The formulation says the gain
should live exactly where episodes are long, progress is a ladder, and the frontier is rare
from a cold start. We will see.

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
