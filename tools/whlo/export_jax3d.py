"""Export the jax3d car scene for the browser demo.

Rebuilds the exact scene from examples/car.py, then exports the physics step to
StableHLO (which whlo compiles to WebAssembly). simulation_params is baked in as
a constant so the only inputs are the flattened SimState + the joint actions.

Writes, under --out:
  step.mlir        SimState leaves + actions -> SimState leaves
  layout.json      per-leaf dtype/shape order for step inputs/outputs
  init_state.json  the initial SimState (flattened leaves), for the JS to load
  scene.json       render config: which leaves hold box/sphere transforms,
                   colours, camera, wheel/chassis indices, drive constants

Usage:  uv run python export_jax3d.py --num-boxes 12 --num-fixated 11
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


def build_scene(num_boxes=5, num_spheres=4, num_fixated_boxes=4):
    static_params = StaticSimParams(
        num_boxes=num_boxes,
        num_spheres=num_spheres,
        num_joints=4,
        num_thrusters=0,
        num_static_fixated_boxes=num_fixated_boxes,
        solver_batch_size=8,
    )
    simulation_params = SimParams()
    engine = PhysicsEngine(static_params)
    sim_state = create_empty_sim(static_params, add_floor=True, add_walls_and_ceiling=False, scene_size=7.2)

    ramp_angle = jnp.radians(15.0)
    ramp_half_extents = jnp.array([1.5, 2.6, 0.05])
    ramp_quat = quat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), -ramp_angle)
    ramp_start_x = -3.5
    sin_angle, cos_angle = jnp.sin(ramp_angle), jnp.cos(ramp_angle)
    ramp_position = jnp.array([
        ramp_start_x + ramp_half_extents[0] * cos_angle - ramp_half_extents[2] * sin_angle,
        0.0,
        ramp_half_extents[0] * sin_angle + ramp_half_extents[2] * cos_angle,
    ])
    sim_state, (_, ramp_index) = add_box_to_scene(
        sim_state, static_params, position=ramp_position, half_extents=ramp_half_extents, quat=ramp_quat, fixated=True
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
    sim_state, (_, landing_index) = add_box_to_scene(
        sim_state, static_params, position=landing_position, half_extents=landing_half_extents, fixated=True
    )

    landing_far_edge_x = landing_position[0] + landing_length_half
    down_ramp_angle = jnp.radians(12.0)
    down_ramp_half_x = ramp_far_top_edge[2] / (2.0 * jnp.sin(down_ramp_angle))
    down_ramp_half_extents = jnp.array([down_ramp_half_x, 2.6, 0.05])
    down_ramp_quat = quat_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), down_ramp_angle)
    down_ramp_near_local = jnp.array([-down_ramp_half_extents[0], 0.0, down_ramp_half_extents[2]])
    down_ramp_target_near = jnp.array([landing_far_edge_x, 0.0, ramp_far_top_edge[2]])
    down_ramp_position = down_ramp_target_near - rotate_vector(down_ramp_quat, down_ramp_near_local)
    sim_state, (_, down_ramp_index) = add_box_to_scene(
        sim_state, static_params, position=down_ramp_position, half_extents=down_ramp_half_extents, quat=down_ramp_quat, fixated=True
    )

    # Pool of extra fixated track boxes for the editor, added BEFORE the chassis
    # so every static box is contiguous at the front and the dynamic chassis is
    # last (num_static_fixated_boxes prunes box-box pairs among the first N, so a
    # dynamic body must sit at index >= N).
    spare_indices = []
    for _ in range(max(0, num_boxes - 5)):
        sim_state, (_, box_index) = add_box_to_scene(
            sim_state, static_params, position=jnp.array([0.0, 0.0, -100.0]),
            half_extents=jnp.array([1.0, 1.0, 0.1]), fixated=True,
        )
        spare_indices.append(int(box_index))

    chassis_half_extents = jnp.array([1.0, 0.5, 0.22])
    chassis_ground_clearance = 0.12
    chassis_z = chassis_ground_clearance + chassis_half_extents[2]
    chassis_position = jnp.array([-5.4, 0.0, chassis_z])
    sim_state, (_, chassis_index) = add_box_to_scene(
        sim_state, static_params, position=chassis_position, half_extents=chassis_half_extents, friction=0.5
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
        sim_state, (_, wheel_index) = add_sphere_to_scene(
            sim_state, static_params, position=chassis_position + offset, radius=wheel_radius, friction=1.2
        )
        sim_state, _ = add_revolute_joint_to_scene(
            sim_state, static_params, chassis_index, wheel_index,
            a_local_anchor=offset, b_local_anchor=jnp.zeros(3),
            a_local_hinge_axis=hinge_axis, b_local_hinge_axis=hinge_axis,
            motor_on=True, motor_speed=1.0, motor_power=1.0,
        )
        wheel_indices.append(int(wheel_index))

    # Deactivate the spare pool boxes; the default track is just the three ramps.
    if spare_indices:
        active = sim_state.box.active.at[jnp.array(spare_indices)].set(False)
        sim_state = sim_state.replace(box=sim_state.box.replace(active=active))
    editable_boxes = [int(ramp_index), int(landing_index), int(down_ramp_index)] + spare_indices

    scene_info = {
        "chassis_index": int(chassis_index),
        "floor_index": 0,
        "editable_boxes": editable_boxes,
        "wheel_indices": wheel_indices,
    }
    return static_params, simulation_params, engine, sim_state, scene_info


def leaf_specs(abstract_values):
    return [{"dtype": str(value.dtype), "shape": list(value.shape)} for value in abstract_values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "jax3d")),
    )
    parser.add_argument("--num-boxes", type=int, default=5)
    parser.add_argument("--num-spheres", type=int, default=4)
    parser.add_argument("--num-fixated", type=int, default=4)
    arguments = parser.parse_args()
    os.makedirs(arguments.out, exist_ok=True)

    static_params, simulation_params, engine, state, scene_info = build_scene(
        arguments.num_boxes, arguments.num_spheres, arguments.num_fixated
    )
    actions = jnp.zeros(static_params.num_joints)

    def step_function(state, actions):
        new_state, _ = engine.step(state, simulation_params, actions)
        return new_state

    exported_module = export.export(jax.jit(step_function, keep_unused=True))(state, actions)
    mlir_text = exported_module.mlir_module()
    with open(os.path.join(arguments.out, "step.mlir"), "w") as file:
        file.write(mlir_text)

    state_leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    key_to_string = jax.tree_util.keystr
    leaf_index_by_path = {key_to_string(path): index for index, (path, _) in enumerate(state_leaves)}

    def find_leaf(path):
        if path not in leaf_index_by_path:
            raise KeyError(f"{path} not in {list(leaf_index_by_path)[:40]}")
        return leaf_index_by_path[path]

    initial_state = [
        {"dtype": str(np.asarray(leaf).dtype), "shape": list(np.asarray(leaf).shape),
         "data": np.asarray(leaf).ravel().tolist()}
        for _, leaf in state_leaves
    ]
    with open(os.path.join(arguments.out, "init_state.json"), "w") as file:
        json.dump(initial_state, file)

    scene_config = {
        "num_boxes": static_params.num_boxes,
        "num_spheres": static_params.num_spheres,
        "num_joints": static_params.num_joints,
        "box_colors": [[0.588, 0.588, 0.608]] * 4 + [[0.824, 0.235, 0.235]],
        "sphere_color": [0.18, 0.18, 0.19],
        "camera": {"target": [0.0, 0.0, 0.5], "distance": 8.0, "azimuth": -2.2, "elevation": 0.4, "fov_deg": 50.0},
        "drive": {"left_wheels": [0, 2], "right_wheels": [1, 3], "turn_strength": 0.9,
                   "max_yaw_rate": 1.2, "yaw_smoothing": 0.15, "steps_per_frame": 4},
        "num_state_leaves": len(state_leaves),
        "leaf": {
            "box_position": find_leaf(".box.position"),
            "box_quat": find_leaf(".box.quat"),
            "box_half_extents": find_leaf(".box.half_extents"),
            "box_active": find_leaf(".box.active"),
            "box_angular_velocity": find_leaf(".box.angular_velocity"),
            "sphere_position": find_leaf(".sphere.position"),
            "sphere_quat": find_leaf(".sphere.quat"),
            "sphere_radius": find_leaf(".sphere.radius"),
            "sphere_active": find_leaf(".sphere.active"),
        },
        **scene_info,
    }
    with open(os.path.join(arguments.out, "scene.json"), "w") as file:
        json.dump(scene_config, file, indent=2)

    layout = {"inputs": leaf_specs(exported_module.in_avals), "outputs": leaf_specs(exported_module.out_avals)}
    with open(os.path.join(arguments.out, "layout.json"), "w") as file:
        json.dump(layout, file)

    # op histogram, to spot anything whlo might not support
    op_counts = {}
    for line in mlir_text.splitlines():
        for token in line.split():
            if token.startswith("stablehlo."):
                op_name = token.split("(")[0]
                op_counts[op_name] = op_counts.get(op_name, 0) + 1
    print(f"step.mlir: {len(mlir_text):,} bytes, {len(state_leaves)} state leaves, {len(exported_module.in_avals)} inputs")
    print("distinct stablehlo ops:", ", ".join(sorted(op_counts)))
    print("wrote:", ", ".join(sorted(os.listdir(arguments.out))))


if __name__ == "__main__":
    main()
