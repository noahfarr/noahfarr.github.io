---
layout: page
title: MCTS
description: Monte Carlo Tree Search that runs end-to-end on JAX.
img: assets/img/projects/mcts.svg
importance: 1
category: demos
github: https://github.com/noahfarr/mcts
---

[MCTS](https://github.com/noahfarr/mcts) is a Monte Carlo Tree Search that runs end-to-end on [JAX](https://github.com/google/jax). Selection, expansion, simulation, and back-propagation all compile with `jax.jit` and run on CPU, GPU, or TPU, with no Python loops inside the algorithm. Every function operates on a single tree, so you batch thousands of self-play searches yourself with one `jax.vmap`, no host round-trips. It builds on [pgx](https://github.com/sotetsuk/pgx), so any pgx environment works out of the box, and a tree can be reused and grown across a whole episode.

To show it running with no server at all, the same search is exported to StableHLO with `jax.export` and compiled to WebAssembly by [whlo](https://github.com/noahfarr/whlo). Play tic-tac-toe as X below: the tree search runs entirely in your browser, and after each move the shaded squares show how many of its simulations it spent on each option.

<div id="mcts-app"></div>

<noscript>This demo needs JavaScript and WebAssembly.</noscript>

<script type="module" src="{{ '/assets/js/mcts/demo.mjs' | relative_url }}"></script>
