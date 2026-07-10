---
layout: page
title: jax3d
description: A JAX 3D rigid-body physics engine, driven in your browser.
img: assets/img/projects/jax3d.svg
importance: 2
category: demos
github: https://github.com/noahfarr/Jax3D
---

[jax3d](https://github.com/noahfarr/Jax3D) is a JAX 3D rigid-body physics engine, a 3D sibling to [Jax2D](https://github.com/MichaelTMatthews/Jax2D). It supports boxes, spheres, wedges, and prisms connected by revolute, fixed, and spherical joints, all fully jittable and batchable with `jax.vmap`.

The scene below is the real engine, not a video. Its physics `step` (contacts, an iterative constraint solver, motorised wheel joints) is exported to StableHLO with `jax.export` and compiled to WebAssembly by [whlo](https://github.com/noahfarr/whlo), so the whole simulation runs client-side with no server. In **Build** mode, drop ramps, platforms, and walls to lay out a track; switch to **Drive** and take the car around what you made.

<div id="jax3d-app"></div>

<noscript>This demo needs JavaScript and WebGL.</noscript>

<script type="module" src="{{ '/assets/js/jax3d/demo.mjs' | relative_url }}"></script>
