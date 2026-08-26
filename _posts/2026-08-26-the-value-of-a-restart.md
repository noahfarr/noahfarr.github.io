---
layout: post
title: the value of a restart
date: 2026-08-26 19:30:00 +0200
description: what a restart is worth, derived as a product of four factors, and why every restart heuristic is a choice of stand-ins for the ones you cannot observe
tags: reinforcement-learning exploration sample-efficiency
categories: research
featured: true
---

In a long-horizon task, an agent spends most of every episode redoing things it already
knows how to do, and if the simulator can reset to any state it has seen, it does not have
to. The open question was never whether to restart somewhere better, it was what "better"
means, and the methods that answer it in practice, reverse curricula, archive-driven
exploration, prioritized replay of levels, each answer with a different hand-designed score.
This post derives the quantity they are all approximating, and the familiar scores come back
as factors of a single product.

## the question, asked properly

Restart selection is itself a sequential decision problem. Its states are training states,
snapshots of the learner. Its actions are restart distributions. Its reward is terminal:
$$J_\rho(\pi_T)$$, the evaluation return of the final policy, when episodes start where the
task says they start. Nothing before the horizon pays anything. That last part is easy to
get subtly wrong, because a restart is not supposed to buy improvement today. It is supposed
to buy a better final policy, and the difference is exactly why curricula exist:
practice can be worthless now and valuable because of what a later restart converts it into.

The exact solution of this problem is a dynamic program over training states, hopeless and
well defined, so every practical rule is an approximation of it. The useful move is to make
the approximations named and separable rather than implicit.

## two cuts

**Freeze the feedback.** Practice now changes the policy, the changed policy collects
different data later, and later training learns from that data. Discard the second channel:
a change made now reaches the horizon only by persisting in the parameters. This is a real
loss, not a formality. Each later step's dependence on shifted data costs one power of the
step size, but a run holds inversely many later steps, so the discarded channel is the same
order as the kept one. What it discards has a name, the stepping stone, practice whose value
is the practice it unlocks, which is the thing Go-Explore's archive growth engineers by hand
{% cite ecoffet2021goexplore --file references %}.

**Go to first order, for a local learner.** Assume the learner is local, improving its
decision at one state moves no other, and incremental, it improves each visited state in
proportion to visitation. A table of per-state policies is the canonical case. The
derivation is then three facts multiplied together. Episodes restarted at $$s$$ deliver
practice to each state $$x$$ in proportion to the occupancy $$d^{\pi_k}_s(x)$$, that is what
incremental means. The terminal objective's sensitivity to the decision at $$x$$ needs no machinery, it pops
out of $$J_\rho$$ decomposing over the states its episodes visit: evaluation trajectories
pass through $$x$$ with weight $$d^{\pi_T}_\rho(x)$$, and a nudge of the decision there is
worth the final advantage of whatever probability it moved, discounted by how much of the
nudge survives later training. And the nudge itself is what reinforcing your own
experience produces, each action shifted by the advantage it collected at the frequency it
was tried, which is where one factor of $$\pi_k$$ and one $$A^{\pi_k}$$ enter, with the
second $$\pi_k$$ coming from converting a preference shift into a probability shift. Because
the learner is local, these multiply state by state and nothing crosses, giving

$$
h_k(s) \;=\; \sum_x d^{\pi_k}_s(x)\; d^{\pi_T}_\rho(x)\; \phi(x)
\sum_a \pi_k(a \mid x)^2\, A^{\pi_k}(x, a)\, A^{\pi_T}(x, a).
$$

This is the value of a restart at $$s$$, at step $$k$$ of a run that ends at $$T$$, and its
factors sort themselves by when they live. The practice factors are now: the footprint
$$d^{\pi_k}_s$$, what a restarted episode actually visits, and one advantage
$$A^{\pi_k}$$, what practicing there teaches today. The payoff factors are at the end of
training: the relevance $$d^{\pi_T}_\rho$$, whether evaluation runs will pass through
$$x$$, the retention $$\phi(x)$$, how much of the nudge later updates leave standing, and
the final advantage $$A^{\pi_T}$$, whether the nudged action still matters when it counts.
A state the final policy has mastered has $$A^{\pi_T} \to 0$$ and is worth nothing to
practice, however badly the current policy stumbles there. That single observation is the
"skip the part you already know" intuition, now a factor instead of a slogan.

## every heuristic is a choice of stand-ins

Three of the factors live at the end of training and the run is standing at $$k$$, so each
needs a stand-in, and this is where the entire heuristic literature lives.

| factor                  | cheapest stand-in                | better stand-in                                  |
| :---------------------- | :------------------------------- | :----------------------------------------------- |
| $$d^{\pi_T}_\rho$$      | current occupancy                | visitation of high-return episodes, subgoal potentials |
| $$A^{\pi_T}$$           | current advantage, giving $$A^2$$ | the witnessed gap $$(G^{\max} - V^{\pi})^{+}$$   |
| $$\phi$$                | one                              | open question                                    |

Take the present everywhere and the cross-time product degenerates to a square,
$$\sum_a \pi^2 A^2$$, the score a purely on-policy estimator can compute. That is the
all-present member of the family, and its defining vice is now visible in its type
signature: it endorses whatever the run already does, and it can rank no state it has never
reached above one it visits every episode.

The familiar prescriptions slot in cleanly. Covering $$d^{\pi^\star}_\rho$$, the answer
concentrability bounds hand you {% cite kakade2002approximately --file references %}, is the
relevance factor, alone, and used alone it is destructive rather than merely incomplete:
relevance multiplies the gain, it does not replace it, and relevance without the practice
factors zeroes states that are off the optimal path but still teaching. Gradient-magnitude
and temporal-difference scores are the practice factor, alone, no relevance, and in a
preregistered screen on a diagnostic environment they measured indistinguishable from
ignoring the score entirely. The witnessed gap won that screen by a wide margin, and the
equation says why: it is the optimistic stand-in for $$A^{\pi_T}$$, so it sees what every
on-policy score is blind to, improvement the current policy has not yet made routine.

## measured, where everything is checkable

On grid worlds every quantity in the equation can be computed exactly by dynamic
programming, which turns the claims from arguments into checks. Three of them, in
increasing order of consequence.

**The identity holds.** Compare the predicted first-order improvement $$\alpha\, h(s)$$
against the true change in $$J_\rho$$ from actually taking the update, at every state, on
boards easy and hard: correlation $$1.000000$$ at small step sizes, with the graceful
quadratic degradation at large ones that a first-order claim owes you. Whatever else is
wrong with the cuts, the algebra is not.

**The factors do their jobs.** Build a board where they have different jobs: a chain of
bottleneck rooms whose start and goal live in one component, welded to a disconnected annex
rich in reward that no evaluation episode can ever enter. Improvement is abundant in the
annex, so a magnitude score spends half its restart budget mastering it. The relevance
factor is identically zero there, so the product never sends a single restart in. At a
matched restart budget the greedy rule on the score reaches 95% of optimal return, the
magnitude score reaches 68%, and standard training from $$\rho$$ learns nothing at all,
across ten seeds with standard errors below $$10^{-3}$$.

<figure style="margin: 1.8rem 0; text-align: center;">
<div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
  <img src="{{ '/assets/img/restarts/hero_drho.png' | relative_url }}" alt="Occupancy of evaluation runs." style="width:32%; min-width:180px; height:auto;">
  <img src="{{ '/assets/img/restarts/hero_g.png' | relative_url }}" alt="Improvement available." style="width:32%; min-width:180px; height:auto;">
  <img src="{{ '/assets/img/restarts/hero_w.png' | relative_url }}" alt="Their product." style="width:32%; min-width:180px; height:auto;">
</div>
<figcaption class="caption" style="margin-top:8px;">The factors, one board. Evaluation occupancy is zero in the annex (left), improvement is abundant there (middle), and the product erases it (right).</figcaption>
</figure>

<figure style="margin: 1.8rem 0; text-align: center;">
<div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
  <img src="{{ '/assets/img/restarts/hero_ours.png' | relative_url }}" alt="Restarts placed by the score." style="width:40%; min-width:200px; height:auto;">
  <img src="{{ '/assets/img/restarts/hero_baseline.png' | relative_url }}" alt="Restarts placed by a magnitude score." style="width:40%; min-width:200px; height:auto;">
</div>
<figcaption class="caption" style="margin-top:8px;">Where the budget goes. The score threads the chain's doors (left); the magnitude score spends half its restarts mastering the annex evaluation can never enter (right).</figcaption>
</figure>
**The schedules emerge.** Nobody told the rule about curricula. On boards where reward
signal binds, its restarts sweep backward from the goal, a reverse curriculum growing
toward the start. On boards where coverage binds, they expand outward from the start, a
growing frontier. Two method families the field designed by hand, produced as the two modes
of one score, with the environment selecting the mode. That is the strongest kind of
evidence a formulation can give: it did not need to be told what the practitioners know.

<figure style="margin: 1.6rem auto; text-align: center;">
<img src="{{ '/assets/img/restarts/curriculum.png' | relative_url }}" alt="Restart mass sweeping backward from the goal over training." style="width:100%; max-width:640px; height:auto;">
<figcaption class="caption">The reverse curriculum nobody asked for: restart mass sweeping backward from the goal as competence grows.</figcaption>
</figure>

## boundaries

The frozen-feedback cut still discards the stepping stones proper, so nothing here prices
practice whose value is the practice it unlocks, and environments that are all
stepping-stone, DeepSea-style needles where no witnessed return precedes discovery, should
and do show nothing {% cite tavakoli2018restart --file references %}. The retention factor
is set to one because I do not yet have an estimator I believe for it. And the equation is
a theorem only for local learners; for a network it is a transplant whose discarded term is
interference, which deserves its own writeup.

The summary has not changed since the first attempt, it has only become precise. Where a
training episode begins is a free variable, and the field has been setting it with
hand-designed schedules. The restart distribution is not a schedule to design. It is a
quantity to estimate, four factors, three stand-ins, and the stand-ins are where the
remaining research is.

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
