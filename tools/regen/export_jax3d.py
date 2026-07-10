"""Export the jax3d car scene for the browser demo.

Rebuilds the exact scene from examples/car.py, then exports the physics step to
StableHLO (which whlo compiles to WebAssembly). sim_params is baked in as a
constant so the only inputs are the flattened SimState + the joint actions.

Writes, under --out:
  step.mlir        SimState leaves + actions -> SimState leaves
  layout.json      per-leaf dtype/shape order for step inputs/outputs
  init_state.json  the initial SimState (flattened leaves), for the JS to load
  scene.json       render config: which leaves hold box/sphere transforms,
                   colours, camera, wheel/chassis indices, drive constants

Usage (from ~/Jax3D):  uv run python examples/export_web.py --out web_export
"""

from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from jax3d.engine import PhysicsEngine, create_empty_sim
from jax3d.maths import quat_from_axis_angle, rotate_vector
from jax3d.scene import (
    add_box_to_scene,
    add_revolute_joint_to_scene,
    add_sphere_to_scene,
)
from jax3d.sim_state import SimParams, StaticSimParams


def build_scene(num_boxes=5, num_spheres=4, num_fixated=4):
    static = StaticSimParams(
        num_boxes=num_boxes,
        num_spheres=num_spheres,
        num_joints=4,
        num_thrusters=0,
        num_static_fixated_boxes=num_fixated,
        solver_batch_size=8,
    )
    sim_params = SimParams()
    engine = PhysicsEngine(static)
    s = create_empty_sim(static, add_floor=True, add_walls_and_ceiling=False, scene_size=7.2)

    ramp_angle = jnp.radians(15.0)
    ramp_half_extents = jnp.array([1.5, 2.6, 0.05])
    ramp_quat = quat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), -ramp_angle)
    ramp_start_x = -3.5
    sin_a, cos_a = jnp.sin(ramp_angle), jnp.cos(ramp_angle)
    ramp_position = jnp.array([
        ramp_start_x + ramp_half_extents[0] * cos_a - ramp_half_extents[2] * sin_a,
        0.0,
        ramp_half_extents[0] * sin_a + ramp_half_extents[2] * cos_a,
    ])
    s, (_, ramp_index) = add_box_to_scene(
        s, static, position=ramp_position, half_extents=ramp_half_extents, quat=ramp_quat, fixated=True
    )

    ramp_far_top_edge = ramp_position + rotate_vector(
        ramp_quat, jnp.array([ramp_half_extents[0], 0.0, ramp_half_extents[2]])
    )
    landing_length_half = 1.0
    landing_half_extents = jnp.array([landing_length_half, ramp_half_extents[1], ramp_half_extents[2]])
    landing_position = jnp.array([
        ramp_far_top_edge[0] + landing_length_half,
        0.0,
        ramp_far_top_edge[2] - ramp_half_extents[2],
    ])
    s, (_, landing_index) = add_box_to_scene(
        s, static, position=landing_position, half_extents=landing_half_extents, fixated=True
    )

    landing_far_edge_x = landing_position[0] + landing_length_half
    down_ramp_angle = jnp.radians(12.0)
    down_ramp_half_x = ramp_far_top_edge[2] / (2.0 * jnp.sin(down_ramp_angle))
    down_ramp_half_extents = jnp.array([down_ramp_half_x, 2.6, 0.05])
    down_ramp_quat = quat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), down_ramp_angle)
    down_ramp_near_local = jnp.array([-down_ramp_half_extents[0], 0.0, down_ramp_half_extents[2]])
    down_ramp_target_near = jnp.array([landing_far_edge_x, 0.0, ramp_far_top_edge[2]])
    down_ramp_position = down_ramp_target_near - rotate_vector(down_ramp_quat, down_ramp_near_local)
    s, (_, down_ramp_index) = add_box_to_scene(
        s, static, position=down_ramp_position, half_extents=down_ramp_half_extents, quat=down_ramp_quat, fixated=True
    )

    # Pool of extra fixated track boxes for the editor, added BEFORE the chassis
    # so every static box is contiguous at the front and the dynamic chassis is
    # last (num_static_fixated_boxes prunes box-box pairs among the first N, so a
    # dynamic body must sit at index >= N).
    spares = []
    for _ in range(max(0, num_boxes - 5)):
        s, (_, idx) = add_box_to_scene(
            s, static, position=jnp.array([0.0, 0.0, -100.0]),
            half_extents=jnp.array([1.0, 1.0, 0.1]), fixated=True,
        )
        spares.append(int(idx))

    chassis_half_extents = jnp.array([1.0, 0.5, 0.22])
    chassis_ground_clearance = 0.12
    chassis_z = chassis_ground_clearance + chassis_half_extents[2]
    chassis_position = jnp.array([-5.4, 0.0, chassis_z])
    s, (_, chassis_index) = add_box_to_scene(
        s, static, position=chassis_position, half_extents=chassis_half_extents, friction=0.5
    )

    wheel_radius = 0.33
    wheel_x_offset = chassis_half_extents[0] * 0.7
    wheel_y_offset = chassis_half_extents[1] + wheel_radius * 0.4
    wheel_z_offset = wheel_radius - chassis_z
    hinge_axis = jnp.array([0.0, 1.0, 0.0])
    wheel_offsets = [
        jnp.array([wheel_x_offset, wheel_y_offset, wheel_z_offset]),
        jnp.array([wheel_x_offset, -wheel_y_offset, wheel_z_offset]),
        jnp.array([-wheel_x_offset, wheel_y_offset, wheel_z_offset]),
        jnp.array([-wheel_x_offset, -wheel_y_offset, wheel_z_offset]),
    ]
    wheel_indices = []
    for offset in wheel_offsets:
        s, (_, wheel_index) = add_sphere_to_scene(
            s, static, position=chassis_position + offset, radius=wheel_radius, friction=1.2
        )
        s, _ = add_revolute_joint_to_scene(
            s, static, chassis_index, wheel_index,
            a_local_anchor=offset, b_local_anchor=jnp.zeros(3),
            a_local_hinge_axis=hinge_axis, b_local_hinge_axis=hinge_axis,
            motor_on=True, motor_speed=1.0, motor_power=1.0,
        )
        wheel_indices.append(int(wheel_index))

    # Deactivate the spare pool boxes; the default track is just the three ramps.
    if spares:
        active = s.box.active.at[jnp.array(spares)].set(False)
        s = s.replace(box=s.box.replace(active=active))
    editable = [int(ramp_index), int(landing_index), int(down_ramp_index)] + spares

    info = {
        "chassis_index": int(chassis_index),
        "floor_index": 0,
        "editable_boxes": editable,
        "wheel_indices": wheel_indices,
    }
    return static, sim_params, engine, s, info


def leaf_specs(avals):
    return [{"dtype": str(a.dtype), "shape": list(a.shape)} for a in avals]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "jax3d")),
    )
    ap.add_argument("--num-boxes", type=int, default=5)
    ap.add_argument("--num-spheres", type=int, default=4)
    ap.add_argument("--num-fixated", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    static, sim_params, engine, state, info = build_scene(args.num_boxes, args.num_spheres, args.num_fixated)
    actions = jnp.zeros(static.num_joints)

    def step_fn(state, actions):
        new_state, _ = engine.step(state, sim_params, actions)
        return new_state

    exported = export.export(jax.jit(step_fn, keep_unused=True))(state, actions)
    mlir = exported.mlir_module()
    with open(os.path.join(args.out, "step.mlir"), "w") as f:
        f.write(mlir)

    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    keystr = jax.tree_util.keystr
    index_of = {keystr(p): i for i, (p, _) in enumerate(leaves)}

    def find(name):
        if name not in index_of:
            raise KeyError(f"{name} not in {list(index_of)[:40]}")
        return index_of[name]

    init_state = [
        {"dtype": str(np.asarray(leaf).dtype), "shape": list(np.asarray(leaf).shape),
         "data": np.asarray(leaf).ravel().tolist()}
        for _, leaf in leaves
    ]
    with open(os.path.join(args.out, "init_state.json"), "w") as f:
        json.dump(init_state, f)

    scene = {
        "num_boxes": static.num_boxes,
        "num_spheres": static.num_spheres,
        "num_joints": static.num_joints,
        "box_colors": [[0.588, 0.588, 0.608]] * 4 + [[0.824, 0.235, 0.235]],
        "sphere_color": [0.18, 0.18, 0.19],
        "camera": {"target": [0.0, 0.0, 0.5], "distance": 8.0, "azimuth": -2.2, "elevation": 0.4, "fov_deg": 50.0},
        "drive": {"left_wheels": [0, 2], "right_wheels": [1, 3], "turn_strength": 0.9,
                   "max_yaw_rate": 1.2, "yaw_smoothing": 0.15, "steps_per_frame": 4},
        "num_state_leaves": len(leaves),
        "leaf": {
            "box_position": find(".box.position"),
            "box_quat": find(".box.quat"),
            "box_half_extents": find(".box.half_extents"),
            "box_active": find(".box.active"),
            "box_angular_velocity": find(".box.angular_velocity"),
            "sphere_position": find(".sphere.position"),
            "sphere_quat": find(".sphere.quat"),
            "sphere_radius": find(".sphere.radius"),
            "sphere_active": find(".sphere.active"),
        },
        **info,
    }
    with open(os.path.join(args.out, "scene.json"), "w") as f:
        json.dump(scene, f, indent=2)

    layout = {"inputs": leaf_specs(exported.in_avals), "outputs": leaf_specs(exported.out_avals)}
    with open(os.path.join(args.out, "layout.json"), "w") as f:
        json.dump(layout, f)

    # op histogram, to spot anything whlo might not support
    ops = {}
    for line in mlir.splitlines():
        for tok in line.split():
            if tok.startswith("stablehlo."):
                name = tok.split("(")[0]
                ops[name] = ops.get(name, 0) + 1
    print(f"step.mlir: {len(mlir):,} bytes, {len(leaves)} state leaves, {len(exported.in_avals)} inputs")
    print("distinct stablehlo ops:", ", ".join(sorted(ops)))
    print("wrote:", ", ".join(sorted(os.listdir(args.out))))


if __name__ == "__main__":
    main()
