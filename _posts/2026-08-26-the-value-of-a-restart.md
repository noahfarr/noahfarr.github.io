---
layout: post
title: the value of a restart
date: 2026-08-26 19:30:00 +0200
description: what a restart is worth, derived as a product of three factors plus a term for what an episode opens up, and why every restart heuristic is a choice of stand-ins for the ones you cannot observe
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
{% cite ecoffet2021goexplore --file references %}. Not all of it goes. An episode also adds
the states it visits to the archive, and that move is order one in the step size, so this
cut keeps it. It comes back below as a second term, and it is the term Go-Explore is.

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
w_k(s) \;=\; \sum_x d^{\pi_k}_s(x)\; d^{\pi_T}_\rho(x)\; g_k(x),
\qquad
g_k(x) \;=\; \sum_a \pi_k(a \mid x)^2\, A^{\pi_k}(x, a)\, A^{\pi_T}(x, a).
$$

This is the value of a restart at $$s$$, at step $$k$$ of a run that ends at $$T$$, and it
is three factors. The footprint $$d^{\pi_k}_s$$ is what a restarted episode actually
visits. The relevance $$d^{\pi_T}_\rho$$ is whether evaluation runs will pass through $$x$$
when it counts. Between them sits the gain $$g_k$$, which contracts what the learner does
at a state with what a change there is worth at the end, one advantage from now and one
from the horizon. A state the final policy has mastered has $$A^{\pi_T} \to 0$$ and is
worth nothing to practice, however badly the current policy stumbles there. That single
observation is the "skip the part you already know" intuition, now a factor instead of a
slogan.

An earlier version of this post carried a fourth factor, a retention term $$\phi$$ for how
much of a nudge later updates leave standing, with the honest note that I had no estimator
for it. It turns out not to need one. At first order retention is one, so $$\phi$$ leaves
the score rather than sitting inside it unestimated. The learner enters only through its
per-visit update: reinforcing your own experience gives $$u_k = \pi_k^2 A^{\pi_k}$$, which
is what produces the two factors of $$\pi_k$$ above, and a value-based learner changes
$$u_k$$ and nothing else around it.

## what an episode opens up

Everything so far holds the archive fixed. Let the episode add the states it visits, and
the first-order expansion has a second piece, which separates from the first exactly. One
term teaches, holding the archive still while the learner moves. The other opens, holding
the learner still while the archive grows.

What an archive is worth to the refreshes still to come is its best option,
$$w^\star_k = \max_{s' \in \mathcal{A}_k} w_k(s')$$, since the greedy rule takes a vertex.
An episode raises that worth only if it reaches something better than the incumbent, so the
opening term is that improvement, collected once for every refresh that remains,

$$
\xi_k(s) \;=\; (T - k - 1) \sum_{x \notin \mathcal{A}_k} d^{\pi_k}_s(x)\;
\max\big(0,\; w_k(x) - w^\star_k\big),
$$

and the value of a restart under archive access is the two together, $$w_k(s) + \xi_k(s)$$.
Note that the inner $$w_k(x)$$ is the score a state would carry once you can restart at it,
so the display is a recursion, and a practical rule truncates it by ranking unseen states
under a prior.

This term is where several things become visible at once. It is zero when the simulator can
be set to any state, because then the archive is everything and nothing is opened. It is
zero once the archive already holds a state as good as any the policy can reach, which
hands the run back to the teaching term with no schedule to write. And it is zero under a
uniform inner score, so the prior has to actually rank, which is why novelty is doing real
work in the methods that use it.

Then the reframing: a rule that drops the teaching term and restarts by $$\xi_k$$ alone,
with novelty ranking the unseen, is Go-Explore. The stepping stone the first cut seemed to
throw away is not gone under archive access, it is this term, and the method that engineers
it by hand is the special case that keeps only it.

A third piece appears once the score is estimated rather than known. The estimates are
state too, and an episode updates them, so there is a value to what an episode teaches the
estimator. It is identically zero under oracle scoring and at the horizon, and to first
order it is the expected improvement of the next selection from tightening the estimates,
which is the expected improvement of Bayesian optimization wearing different clothes. Its
cheapest stand-in is optimism: a state whose score rests on no success yet is scored at the
archive's best until it has been tried.

## every heuristic is a choice of stand-ins

Two quantities in the score live at the end of training while the run stands at $$k$$, the
relevance and the final advantage, as does the inner score of the opening term. Each needs
a stand-in, and this is where the entire heuristic literature lives.

| factor             | cheapest stand-in                         | optimistic stand-in                                    |
| :----------------- | :---------------------------------------- | :----------------------------------------------------- |
| $$d^{\pi_k}_s$$    | sampled by restarting, no stand-in needed | ---                                                     |
| $$d^{\pi_T}_\rho$$ | uniform over the archive                  | current occupancy $$d^{\pi_k}_\rho$$, elite visitation  |
| $$A^{\pi_T}$$      | current advantage, giving $$A^2$$         | the witnessed gap $$\max(0, G^{\max} - V^{\pi})$$       |
| $$\xi_k$$          | progress along the objective              | witnessed gap of the discovering trail                  |
| $$\epsilon_k$$     | optimism until tried                      | expected improvement                                    |

The columns are cost and optimism, not a recommendation. What the runs actually use is the
optimistic column for relevance and the cheap one for the estimation term.

The footprint is the one factor that needs nothing, since every episode restarted at $$s$$
samples it without bias.

The relevance row is worth dwelling on, because the two ways of being wrong are not
symmetric. Spending restart mass where the final policy never goes costs a bounded
dilution. Withholding mass from a state it does visit makes the mismatch coefficient that
prices a fixed start distribution infinite
{% cite kakade2002approximately --file references %}. Uniform presumes nothing and caps
that coefficient at the size of the archive. Current occupancy ranks far better, since it
actually knows something about where the run goes, but taken alone it is exactly zero on
everything not yet reached, and caps the coefficient at nothing.

The tempting conclusion is to retreat to uniform. That trades away the ranking to buy the
coverage, and it is the wrong trade, because coverage is not a property the relevance
factor has to supply on its own. The estimation term already pays for exactly this. A state
whose score rests on no success yet is scored optimistically until it has been tried, which
puts mass on the unreached without pretending to know that evaluation will go there. So
relevance stays current occupancy and every coverage job lives in that one additive term,
rather than being smeared across the factor that is supposed to be doing the ranking.

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

**The identity holds.** Compare the predicted first-order improvement $$\alpha\, w(s)$$
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

## at scale

Grid worlds are where the claims are checkable, not where they matter. The same estimators
run inside a recurrent PPO learner on Craftax, sixteen thousand parallel environments on
one accelerator, a device-resident archive of sixteen thousand cells holding states and
recurrent carries, and restarts drawn for a quarter of the batch with importance
corrections on the warm data. At a matched hundred-million-step budget the oracle score,
the gradient inner-product estimator, and the factorized score are indistinguishable, and
the factorized one is cheapest because it does not need the extra backward pass. At a
billion steps restarts sit inside the no-restart seed band.

The separation arrives in the long grind after that, which is exactly the frontier regime
where the theory says payment comes due. With the restart share annealed over the full
budget, a restart-scored run reaches $$68.4$$ by five billion steps where a no-restart
baseline reaches $$53.9$$ at the same budget. These are single seeds each, a ten-seed
replication with schedule-matched baselines is in flight, and the margin is provisional
until it reports.

## boundaries

The frozen-feedback cut still discards the stepping stones that work through the learner,
so nothing here prices practice whose value is the practice it unlocks by teaching, and
environments that are all stepping-stone, DeepSea-style needles where no witnessed return
precedes discovery, should and do show nothing
{% cite tavakoli2018restart --file references %}. What the cut does keep is the archive
channel, which is the opening term above, so the stepping stone survives in the one form
the simulator makes cheap.

Under a step budget rather than an episode budget the greedy object is the value density
$$w_k(s) / \ell(s)$$, with $$\ell$$ the expected episode length from $$s$$, since a ratio
of linear functionals is still maximized at a vertex. The footprint's mass saturates at
$$1/(1 - \gamma)$$ while cost keeps accruing, so practice past the discount horizon buys
teaching at a falling rate. Truncating warm episodes is an implication of the formulation
rather than a trick bolted onto it.

And the equation is a theorem only for local learners. For a network it is a transplant,
since shared parameters couple states and the same first-order object becomes an inner
product of gradients whose off-diagonal mass is interference. Adopting the factorization
asserts that mass away, which is a real assumption and deserves its own writeup.

The summary has not changed since the first attempt, it has only become precise. Where a
training episode begins is a free variable, and the field has been setting it with
hand-designed schedules. The restart distribution is not a schedule to design. It is a
quantity to estimate, three factors and a term for what an episode opens up, and the
stand-ins are where the remaining research is.

## References

<div class="publications">
{% bibliography --file references --cited_in_order %}
</div>
