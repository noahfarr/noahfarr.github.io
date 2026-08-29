---
layout: post
title: better demonstrations, not more of them
date: 2026-08-28 02:30:00 +0200
description: a procedurally generated dataset of 11.5M imitation tasks, and the finding that 246x more tasks buys nothing while a 10x better teacher buys a real improvement
tags: behavioural-cloning offline-rl imitation-learning measurement
categories: research
published: true
---

<style>
.fig-svg svg { width: 100%; height: auto; color: var(--global-text-color); }
</style>


Kinetix generates 2D physics tasks without limit {% cite matthews2025kinetix --file
references %}, and its offline dataset spends that freedom on breadth: 11.5M distinct
levels, each solved once by a PPO specialist trained on that level alone, each
contributing exactly one 256-step trajectory. There is no second demonstration of
anything. So a data budget here has two axes and only two, and they trade against each
other directly. You can demonstrate more tasks, or you can train the teacher longer before
recording it.

Twelve of them, drawn from the split this post trains on and rendered from the stored
state, look like this.

<figure style="margin: 2rem 0;">
  <img src="{{ '/assets/img/kinetix/levels.png' | relative_url }}" alt="Twelve generated Kinetix levels" style="width:100%; height:auto;">
  <figcaption class="caption" style="text-align:left; margin-top:0.6rem;">
    Twelve levels from the <code>1M/s</code> split, at the first step of their stored demonstration. Green must reach blue, red is fatal, grey is inert, and the dark band is a wall. Each is a separate task with exactly one demonstration in the dataset, and there are 5.98M more.
  </figcaption>
</figure>

Physics tasks are easier to feel than to describe, so here is one running in the browser.
Click it to give it the keyboard, then drive the motors with the arrow keys and fire the
thrusters with the space bar. The goal is always the same, put the green shape against the
blue one without touching red.

<figure style="margin: 2rem 0;">
  <div class="kinetix-frame" style="position:relative; width:100%; aspect-ratio:4/3; max-height:66vh; border-radius:6px; overflow:hidden;">
    <iframe id="kinetix-embed" title="A Kinetix physics task, playable"
      src="https://kinetix-env.github.io/gallery.html?level=RA5YVsUTXzD7HC4yrpbl&embed=true&editor=true"
      loading="lazy" allowfullscreen allow="clipboard-read; clipboard-write"
      style="position:absolute; inset:0; width:100%; height:100%; border:0;"></iframe>
  </div>
  <figcaption class="caption" style="text-align:left; margin-top:0.6rem;">
    Kinetix.js by Michael Beukman, embedded from kinetix-env.github.io. This is a hand-designed showcase level and the editor it ships with. The levels in this post are smaller, at most five polygons, two circles, one joint and one thruster, and all 11.5M of them were generated rather than authored.
  </figcaption>
</figure>

<script>
  (function () {
    var frame = document.getElementById("kinetix-embed");
    if (!frame) return;
    var grab = function () { try { frame.focus(); } catch (e) {} };
    frame.parentNode.addEventListener("click", grab);
    frame.parentNode.addEventListener("mouseenter", grab);
  })();
</script>

The dataset makes the trade explicit in its own naming. `{policy_steps}/{size}` gives
`1M/s` with 5.98M tasks from experts trained a million steps each, and `10M/s` with 637k
tasks from experts trained ten million. Nine times as many tasks, or a teacher trained ten
times as long. The obvious guess, given that the whole appeal of procedural generation is
unlimited task variety, is that the tasks win.

They do not. Two hundred and forty-six times the tasks is worth nothing measurable, at
either training budget tested. Ten times the teacher's training is worth **2 to 3 points of
success rate**, at every task count tested: 47% to 50% of held-out levels solved at one
task count, 49% to 52% at the other.

Spreading the same budget over more tasks does not destroy that benefit, but it does delay
it. At 610k tasks a run has to get through the data before the better teacher shows up at
all, and a budget that looked sufficient at 23k tasks is not.

All of it, and the evidence, is in the figure below.

## how the comparison is made

Every number below comes from runs that differ in one factor and are otherwise
identical: the same recurrent policy over an entity transformer, the same optimiser and
cosine schedule, and the same training budget within any one comparison. Seed counts run
from three to twenty per arm and are given alongside each result.

Evaluation is on 512 held-out levels drawn from reserved shards, which the policy never
trains on. An untrained network solves 28% of that set, and the ceiling is 100% by construction,
since every stored trajectory succeeded.

One subtlety forces a design decision. Each split contains only the levels its own expert
solved, so `10M/s` holds levels that needed a ten-times-stronger agent. Scoring each arm
on its own reserved shards would confound demonstration quality with evaluation
difficulty. Both arms are therefore scored on the same `1M/s` level bank, the common
ground both experts could handle.

## task diversity does nothing

Fix everything, vary only how many distinct tasks the model may draw from. At a fixed
budget of 100M transitions, the low arm cycles its 23,552 tasks 16.6 times and the high arm
sees 6.7% of its 5,796,864 tasks once.

The top row of the figure is the result. A **246x** change in task count moves held-out
return by **-0.020, which is 1.5 sigma** over six seeds per arm. Held-out loss agrees and
is flatter still, 0.8700 against 0.8703, 0.4 sigma. Nothing here reaches significance, and
the sign is negative throughout, so more tasks is if anything mildly worse.

The obvious objection is that 100M transitions is not enough training for a difference to
appear. It is a fair objection, and it is exactly what happens to the teacher comparison
further down. So the same contrast was run again at **four times the compute**, where the
low arm makes 66 passes over its tasks while the high arm still sees only 27% of its own
once.

That is the second row. Held-out return moves by **-0.017, which is 1.1 sigma**, and
success rate by 0.8 sigma. Both arms gained substantially from the larger budget, 0.529 to
0.588 and 0.510 to 0.571, so the extra compute did move the numbers. It just did not move
them apart.

The effect is also about the same size at both budgets, -0.020 against -0.017. The extra
compute bought a second look at the same small negative number rather than revealing a new
one.

Nor is it an artefact of the learning rate. Run at half and at double the rate used
throughout, five seeds per cell, the comparison stays flat and stays negative: -0.006 at
0.6 sigma and -0.016 at 1.0 sigma, against -0.020 at the tuned rate. A null is more
vulnerable to mistuning than an effect is, and this one is not sitting on one.

<figure class="fig-svg" style="margin: 2rem 0;">
  {% include figures/kinetix-effects.svg %}
  <figcaption class="caption" style="text-align:left; margin-top:0.6rem;">
    Each comparison as a row. Left, every seed as a dot with the mean as a bar. Middle and right, the change and its sampling distribution, shaded to the 95% interval. Endpoints are the mean of each run's last three evaluations. Task count does nothing at either budget. A better teacher pays at 23k tasks, and at 610k once the budget is large enough for it to show.
  </figcaption>
</figure>


The reason is arithmetic rather than anything subtle. The smallest task set is 23,552
trajectories of 256 steps across about five masked action dimensions, roughly 30M discrete
labels, against a 4M parameter model. Even the smallest configuration has well under a
parameter per label. Nothing in this regime is data-limited, so the marginal task cannot
buy anything, and the model that repeats 23k tasks seventeen times does exactly as well as
the one drawing fresh tasks continuously.

## teacher quality does

Now hold the task count fixed and change who produced the demonstrations. `1M/s` at 23,552
tasks against `10M/s` at 22,464, a 5% match, same compute, same evaluation levels.

That is the fourth row. Return improves by **+0.032, at 3.5 sigma**, over ten seeds per
arm, and success rate goes from **46.9% to 49.7%** at 3.4 sigma. Held-out loss moves much
further, 0.870 to 0.722, because data from a better-trained expert is simply more
predictable.

Repeating the comparison at 26x more tasks, 609,024 against 616,576, is the third row,
and the effect on return is **gone**: -0.001, which is 0.07 sigma, about as flat as a null
gets. The loss still improves there, by 0.050 at 5.0 sigma, so the data is still easier to
fit. It just has not bought a better policy yet.

That last word is the important one. At 610k tasks, 100M transitions is 0.64 passes over the
data, and every arm was still climbing when the budget ran out. Give the same comparison
**four times the compute**, 2.6 passes instead of 0.64, and it is the bottom row: return
improves by **+0.023 at 2.5 sigma** over twenty seeds per arm, and success rate goes from
**49.3% to 51.6%** at 3.0 sigma. Both arms also gain a lot in absolute terms, since both
were still learning when the shorter budget cut them off.

So quality is not conditional on repetition. It is worth much the same at both, +0.032 of
return at 23k tasks and +0.023 at 610k. What changes with task count is how long you must
train before you can collect it.

The same four-fold budget was applied to both levers and rescued only one of them. That is
the sharpest evidence here that the two are not interchangeable.

| lever | change | effect on held-out return |
|---|---|---:|
| task count, 100M transitions | 246x more tasks | 1.5 sigma |
| task count, 400M transitions | 246x more tasks | 1.1 sigma |
| teacher, 610k tasks, 0.64 passes | 10x expert training | 0.1 sigma |
| **teacher, 610k tasks, 2.6 passes** | **10x expert training** | **2.5 sigma** |
| **teacher, 23k tasks, 17 passes** | **10x expert training** | **3.5 sigma** |
{: style="margin-left:auto; margin-right:auto; margin-bottom:2rem;"}

For anyone spending a budget on demonstration data in a procedurally generated domain,
that is the actionable pair. Generating more tasks is cheap and appealing and did not help
at either budget we measured. Training each specialist longer did, everywhere we looked. The
only thing task count changed was the training budget needed to see it.

## the learning rate is not doing the work

Every arm above trains at one learning rate, 1.1e-3, chosen from a sweep. A single shared
rate can flatter whichever arm it happens to suit, and it is a stronger lever than anything
being measured here, so the 23k comparison was run again at half and at double that rate,
five seeds per cell.

| learning rate | change in held-out return |
|---|---:|
| 5.5e-4 | +0.078, 7.0 sigma |
| 1.1e-3 | +0.032, 3.5 sigma |
| 2.2e-3 | +0.058, 3.4 sigma |
{: style="margin-left:auto; margin-right:auto; margin-bottom:2rem;"}

The better teacher wins at every rate, and the published number is the smallest of the
three. What moves between rows is mostly the weaker arm, which falls from 0.537 to 0.516
and 0.495 as the rate moves away from its optimum, while the stronger arm holds between
0.553 and 0.594. The two splits do not share an optimal learning rate, and the one used
throughout suits the weaker teacher better, so tuning each arm separately would widen this
gap rather than close it.

## smaller models gain more, not less

The comparison above is at one model size. Capacity is the obvious thing that could decide
it, since a model with room to spare might extract the same policy from worse
demonstrations. Repeating the 23k comparison at a quarter and at four times the width, five
seeds per cell:

| parameters | change in held-out return |
|---|---:|
| 1.02M | +0.089, 6.5 sigma |
| 4.04M | +0.032, 3.5 sigma |
| 16.07M | did not fit the data |
{: style="margin-left:auto; margin-right:auto; margin-bottom:2rem;"}

The effect is real at both sizes that trained, and it is nearly three times larger at the
smaller one. That is the opposite of the usual worry. Better demonstrations matter most
where capacity is scarce, and the 4M number the rest of this post rests on is the more
conservative of the two.

The 16M model is not evidence either way. At 100M transitions it never fit, ending at 0.870
of training loss where the smaller models reach 0.68, and doubling the learning rate moved
that to 0.840. Its loss curve is the same shape as the others, roughly half as far along,
so it is underfitting rather than diverging. Given four times the compute it does fit,
reaching 0.679 before the runs were cut short for cluster time. Reporting a null from a
model that never learned its training data would say something about the budget, not about
demonstration quality.

## why the numbers above are returns and not losses

Behavioural cloning is usually reported as a likelihood, and here that likelihood is
actively misleading, for a reason specific to control.

The policy sees the observation and, if you let it, the expert's previous action. That is
ordinary teacher forcing, the same setup an autoregressive language model trains under,
and there is nothing wrong with it in principle. But in this domain the previous action is
so predictive of the next one that it swamps everything else. Giving the model that one
extra input, with nothing else changed:

<figure class="fig-svg" style="margin: 2rem 0;">
  {% include figures/kinetix-divergence.svg %}
  <figcaption class="caption" style="text-align:left; margin-top:0.6rem;">
    Three seeds per arm, identical but for one input. The likelihood strongly prefers giving the model the expert's previous action, by 0.342. The policy is worse for it, by 0.098 of held-out return at 4.7 sigma.
  </figcaption>
</figure>


Conditioning on the expert's last action improves the loss by **0.342**, a larger effect
than anything else measured here, and costs **0.098 of return at 4.7 sigma** over three
seeds per arm. The likelihood is mostly scoring action persistence, which is available
while training on recorded trajectories and gone at rollout, when the previous action is
the policy's own.

Even without that input the two quantities disagree about timing. At 610k tasks and 0.64
passes the strong-expert data already holds a clear likelihood advantage, 0.821 against
0.871 at 5.0 sigma, while its return advantage is exactly zero. The likelihood registers
the better data long before the policy does.

So the likelihood is reported for completeness and the return is what the claims rest on.

## limitations

The two splits differ in more than the competence of their teachers. Each contains
only the levels its own expert could solve, so the training distributions are not
identical. Fixing the evaluation set to common ground removes that confound from the
scoring but not from the training data, and these two datasets cannot separate the two.

The learning rate is checked at the 23k teacher comparison and at the task-count null,
each across a four-fold span. The 610k comparison is at a single rate.

Everything here is on the small-complexity split. The quality effect replicates across a
26x span of task count and a 4x span of model size, and grows as the model shrinks. Above
4M parameters it is untested, because the 16M model needs more compute to fit than these
runs were given.

Seed-to-seed variation on return is +/- 0.011 to 0.037, comparable to the effects being
measured. Every comparison here is therefore an average over several seeds, ten per arm for
the teacher comparisons at 100M transitions and seven at 400M, six for both task-count
comparisons, three for the previous-action ablation.

No exponent is reported. The irreducible loss is bracketed only between 0.662 and 0.885 by
oracle predictors on this data, an interval wider than the entire range these models
occupy, so a fitted exponent would be dominated by that uncertainty. What can be stated is
a difference between matched conditions with error bars, and that is what is stated above.
