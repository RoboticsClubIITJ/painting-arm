"""
Continuous task-space trajectory generation with kinematic limits.

This is the core planning loop: for a polyline of task-space points, it steps forward in distance one simulation frame (DT) at a time, computing a desired speed from a bell-shaped velocity profile capped by a look-ahead braking curve, then clamps that speed so every joint stays within MAX_JOINT_VEL and MAX_JOINT_ACC (via the Jacobian at the current pose). Each output q is one animation frame.
"""
import numpy as np
from Configurations import (
    MAX_LINEAR_SPEED, MAX_JOINT_VEL, MAX_JOINT_ACC, DT,
    SAFE_DECEL_RATE, MIN_SPEED_FLOOR, NEAR_TARGET_DIST,
    NEAR_TARGET_SPEED_FLOOR, FAR_TARGET_SPEED_FLOOR_FRAC,
)
from Kinematics import ik_fast_dls, compute_jacobian, damped_pseudo_inverse

def get_interpolated_point(points, cum_dist, current_segment, target_dist):
    # Linearly interpolate a point at `target_dist` along the polyline, within `current_segment`.
    seg_start = cum_dist[current_segment]
    seg_end = cum_dist[current_segment + 1]
    seg_len = seg_end - seg_start

    ratio = (target_dist - seg_start) / seg_len if seg_len > 1e-6 else 1.0
    ratio = max(0.0, min(1.0, ratio))

    p0 = points[current_segment]
    p1 = points[current_segment + 1]
    return p0 + ratio * (p1 - p0)

def _desired_speed(s_val, dist_left, v_max):
    # Bell-shaped speed profile (peaks mid-path) capped by a look-ahead braking curve, with a floor so the arm doesn't stall mid-line.
    v_desired = v_max * (16 * s_val ** 2 * (1 - s_val) ** 2)

    v_brake = np.sqrt(2 * SAFE_DECEL_RATE * max(0.0, dist_left))
    v_desired = min(v_desired, v_brake)

    if dist_left < NEAR_TARGET_DIST:
        v_desired = max(v_desired, NEAR_TARGET_SPEED_FLOOR)
    else:
        v_desired = max(v_desired, FAR_TARGET_SPEED_FLOOR_FRAC * v_max)

    return v_desired

def _joint_limited_speed_bounds(L_vec, dq_curr):
    # Given the joint-space direction per unit task-space speed (L_vec) and the previous frame's joint velocities, find the range of task-space speeds that keep every joint within MAX_JOINT_VEL and MAX_JOINT_ACC for this frame.
    v_vlim = min(
        MAX_JOINT_VEL / abs(l) if abs(l) > 1e-6 else float('inf')
        for l in L_vec
    )

    v_max_allowed = float('inf')
    v_min_allowed = 0.0

    for i in range(3):
        l = L_vec[i]
        bound_upper = dq_curr[i] + MAX_JOINT_ACC * DT
        bound_lower = dq_curr[i] - MAX_JOINT_ACC * DT

        if l > 1e-6:
            v_max_i = bound_upper / l
            v_min_i = bound_lower / l
        elif l < -1e-6:
            v_max_i = bound_lower / l
            v_min_i = bound_upper / l
        else:
            v_max_i = float('inf')
            v_min_i = -float('inf')

        v_max_allowed = min(v_max_allowed, v_max_i)
        v_min_allowed = max(v_min_allowed, v_min_i)

    if v_min_allowed > v_max_allowed:
        v_min_allowed = v_max_allowed

    return v_vlim, v_min_allowed, v_max_allowed

def generate_continuous_trajectory(points, q_start, v_max=MAX_LINEAR_SPEED):
    # Generate one joint-angle array per animation frame (DT) tracing the polyline `points`, starting from q_start, respecting joint vel/accel limits throughout.
    if len(points) < 2:
        return []


    points = np.array(points)
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_dist = cum_dist[-1]

    if total_dist < 1e-4:
        q_end, _, _ = ik_fast_dls(points[-1, 0], points[-1, 1], q_init=q_start)
        return [q_end]

    trajectory_q = []
    q_curr = np.copy(q_start)
    dq_curr = np.zeros(3)

    dist = 0.0
    current_segment = 0

    while dist < total_dist:
        dist_left = total_dist - dist
        s_val = dist / total_dist
        v_desired = _desired_speed(s_val, dist_left, v_max)

        # Path tangent direction of the current segment
        p0 = points[current_segment]
        p1 = points[current_segment + 1]
        vec = p1 - p0
        norm_vec = np.linalg.norm(vec)
        u_hat = (vec / norm_vec) if norm_vec > 1e-6 else np.array([1.0, 0.0])

        # Map task-space direction to joint-space direction at current pose
        J = compute_jacobian(q_curr)
        J_inv = damped_pseudo_inverse(J)
        L_vec = J_inv @ u_hat

        v_vlim, v_min_allowed, v_max_allowed = _joint_limited_speed_bounds(L_vec, dq_curr)

        v_actual = min(v_desired, v_vlim, v_max_allowed)
        v_actual = max(v_actual, v_min_allowed)
        speed_floor = NEAR_TARGET_SPEED_FLOOR if dist_left < NEAR_TARGET_DIST else MIN_SPEED_FLOOR
        v_actual = max(v_actual, speed_floor)

        step_dist = v_actual * DT
        if dist + step_dist > total_dist:
            step_dist = total_dist - dist
        dist += step_dist

        while current_segment < len(cum_dist) - 2 and dist > cum_dist[current_segment + 1]:
            current_segment += 1

        p_target = get_interpolated_point(points, cum_dist, current_segment, dist)
        q_next, _, _ = ik_fast_dls(p_target[0], p_target[1], q_init=q_curr)

        dq_curr = (q_next - q_curr) / DT
        q_curr = q_next

        trajectory_q.append(np.copy(q_curr))

    return trajectory_q
