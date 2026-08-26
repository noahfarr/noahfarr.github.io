---
layout: post
title: the value of a restart
date: 2026-08-26 22:00:00 +0200
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
proportion to visitation. A table of per-state policies is the canonical case. Then the
first-order change in the terminal objective decomposes over states, and the times separate:

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

## measured, at two horizons

On Craftax, with a recurrent PPO at sixteen thousand parallel environments, the equation
makes a two-sided prediction and both sides came in.

At one billion steps, nothing. Ten seeds of the no-restart baseline against ten-ish of the
restart variants: means within a standard error of each other, every knob and stand-in we
tried inside the noise band. The equation says why. Cold returns are still climbing
steeply there, meaning $$A^{\pi_T}$$-shaped improvement is still available in the states
cold episodes visit for free, so no archived cell can outbid an ordinary reset. Restarts at
that horizon are a free option that correctly stays unexercised.

At five billion steps, the option pays. Same seed, same budget, same code: the no-restart
baseline reaches 53.9, selection by the witnessed gap alone reaches 57.0, and the full
product, current practice times an elite-visitation stand-in for the relevance, reaches
**64.8**. A ten-seed replication of the endpoints is running as this posts, so treat the
gaps as one seed's word for now. The part I trust already is the shape: the win appears
exactly when the cold curve flattens, which is when the future factors detach from the
present ones, and a per-cell self-estimate built from all-present quantities sat at "not
worth it" through the entire breakaway. The value that restarts collect at scale flows
through the factors the present cannot see.

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
