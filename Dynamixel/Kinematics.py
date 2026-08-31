import numpy as np
from Configurations import L_TUPLE, JOINT_LIMITS, MAX_REACH, IK_MAX_ITER, IK_TOL, IK_DAMPING, IK_MAX_STEP

def FK(q, L=L_TUPLE):
    # End-effector (pen tip) x, y position for joint angles q.
    t1, t2, t3 = q[0], q[1], q[2]
    t12 = t1 + t2
    t123 = t1 + t2 + t3
    x = L[0] * np.cos(t1) + L[1] * np.cos(t12) + L[2] * np.cos(t123)
    y = L[0] * np.sin(t1) + L[1] * np.sin(t12) + L[2] * np.sin(t123)
    return x, y

def arm_link_positions(q, L=L_TUPLE):
    # Cartesian position of base, elbow, wrist, and end-effector. Used for rendering the full arm (draw_arm), not just the IK target.
    t1, t2, t3 = q[0], q[1], q[2]
    p0 = np.array([0.0, 0.0])
    p1 = p0 + np.array([L[0] * np.cos(t1), L[0] * np.sin(t1)])
    p2 = p1 + np.array([L[1] * np.cos(t1 + t2), L[1] * np.sin(t1 + t2)])
    p3 = p2 + np.array([L[2] * np.cos(t1 + t2 + t3), L[2] * np.sin(t1 + t2 + t3)])
    return p0, p1, p2, p3

def compute_jacobian(q, L=L_TUPLE):
    # 2x3 Jacobian mapping joint velocities to end-effector linear velocity.
    t1, t2, t3 = q[0], q[1], q[2]
    t12, t123 = t1 + t2, t1 + t2 + t3
    s1, s12, s123 = np.sin(t1), np.sin(t12), np.sin(t123)
    c1, c12, c123 = np.cos(t1), np.cos(t12), np.cos(t123)
    J = np.array([
        [-L[0] * s1 - L[1] * s12 - L[2] * s123, -L[1] * s12 - L[2] * s123, -L[2] * s123],
        [ L[0] * c1 + L[1] * c12 + L[2] * c123,  L[1] * c12 + L[2] * c123,  L[2] * c123]
    ])
    return J

def damped_pseudo_inverse(J, damping=IK_DAMPING):
    # Damped least-squares pseudo-inverse of a (2x3) Jacobian.
    I_task = np.identity(J.shape[0])
    J_damp = J @ J.T + (damping ** 2) * I_task
    return J.T @ np.linalg.solve(J_damp, I_task)

def clip_to_joint_limits(q):
    q = np.copy(q)
    for i in range(3):
        q[i] = np.clip(q[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])
    return q

def ik_fast_dls(target_x, target_y, q_init, L=L_TUPLE, max_iter=IK_MAX_ITER, tol=IK_TOL):
    # Damped least-squares IK, warm-started from q_init.
    dist = np.hypot(target_x, target_y)
    if dist > MAX_REACH:
        target_x *= (MAX_REACH - 0.01) / dist
        target_y *= (MAX_REACH - 0.01) / dist

    q = np.copy(q_init).astype(float)

    for iteration in range(max_iter):
        curr_x, curr_y = FK(q, L)
        error = np.array([target_x - curr_x, target_y - curr_y])
        if np.linalg.norm(error) < tol:
            return q, True, iteration + 1

        J = compute_jacobian(q, L)
        J_inv = damped_pseudo_inverse(J)
        dq = J_inv @ error

        dq_norm = np.linalg.norm(dq)
        if dq_norm < 1e-4:
            break
        if dq_norm > IK_MAX_STEP:
            dq = dq * (IK_MAX_STEP / dq_norm)

        q += dq
        q = (q + np.pi) % (2 * np.pi) - np.pi
        q = clip_to_joint_limits(q)

    return q, False, max_iter