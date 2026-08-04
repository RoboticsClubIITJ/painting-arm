import numpy as np
import matplotlib.pyplot as plt

def get_matrix(theta_rad, a):
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, -s, a * c],
        [s,  c, a * s],
        [0,  0,   1  ]
    ])

def FKsolver(theta1, theta2, theta3, l1, l2, l3):
    T01 = get_matrix(theta1, l1)
    T12 = get_matrix(theta2, l2)
    T23 = get_matrix(theta3, l3)
    T03 = T01 @ T12 @ T23
    return T03[0, 2], T03[1, 2]

def get_jacobian(thetas, lengths):
    t1, t2, t3 = thetas
    l1, l2, l3 = lengths
    t12 = t1 + t2
    t123 = t1 + t2 + t3
    
    dx_dt1 = -l1*np.sin(t1) - l2*np.sin(t12) - l3*np.sin(t123)
    dx_dt2 = -l2*np.sin(t12) - l3*np.sin(t123)
    dx_dt3 = -l3*np.sin(t123)
    
    dy_dt1 = l1*np.cos(t1) + l2*np.cos(t12) + l3*np.cos(t123)
    dy_dt2 = l2*np.cos(t12) + l3*np.cos(t123)
    dy_dt3 = l3*np.cos(t123)
    
    return np.array([
        [dx_dt1, dx_dt2, dx_dt3],
        [dy_dt1, dy_dt2, dy_dt3],
        [1.0, 1.0, 1.0]
    ])

def IKsolver_Geometric(x, y, θ, l1, l2, l3): # as θ in input is there so we get 2 solutions or else we would get ∞ solutions
    xref = x - l3 * np.cos(θ)
    yref = y - l3 * np.sin(θ)

    # θ2
    c2 = (xref**2 + yref**2 - l1**2 - l2**2)/(2*l1*l2) # we can apply edge case detection here
    s2a = np.sqrt(1-c2**2)
    s2b = -s2a
    θ2a = np.arctan2(s2a, c2)
    θ2b = np.arctan2(s2b, c2)

    # θ1 or gemini method
    c1a = (xref*(l1+l2*c2) + yref*l2*s2a)/((l1+l2*c2)**2 + (l2*s2a)**2)
    c1b = (xref*(l1+l2*c2) + yref*l2*s2b)/((l1+l2*c2)**2 + (l2*s2b)**2)
    s1a = np.sqrt(1-c1a**2)
    s1b = np.sqrt(1-c1b**2)
    θ1a = np.arctan2(s1a, c1a)
    θ1b = np.arctan2(s1b, c1b)

    # θ3
    θ3a = θ - θ2a - θ1a
    θ3b = θ - θ2b - θ1b

    # error
    xa, ya = FKsolver(θ1a, θ2a, θ3a, l1, l2, l3)
    xb, yb = FKsolver(θ1b, θ2b, θ3b, l1, l2, l3)
    errora = np.sqrt((xa - x)**2 + (ya - y)**2)
    errorb = np.sqrt((xb - x)**2 + (yb - y)**2)
    if(errora > 0.1):
        s1a = -s1a
        θ1a = np.arctan2(s1a, c1a)
        θ3a = θ - θ2a - θ1a
    if(errorb > 0.1):
        s1b = -s1b
        θ1b = np.arctan2(s1b, c1b)
        θ3b = θ - θ2b - θ1b
    
    ans1 = np.array([θ1a, θ2a, θ3a])
    ans2 = np.array([θ1b, θ2b, θ3b])
    return ans1, ans2

def IKsolver_Damped_Least_Squares_Memory(x, y, phi, l1, l2, l3, lam, iters, initial_guess):
    # CRITICAL FIX: The solver no longer wipes its memory. 
    # It starts searching from the provided initial_guess.
    thetas = np.copy(initial_guess) 
    I = np.identity(3)
    lam_sq = lam ** 2
    
    for _ in range(iters):
        t1, t2, t3 = thetas[0, 0], thetas[1, 0], thetas[2, 0]
        
        x_curr, y_curr = FKsolver(t1, t2, t3, l1, l2, l3)
        phi_curr = t1 + t2 + t3
        
        err_x = x - x_curr
        err_y = y - y_curr
        # Wrap angle error to [-pi, pi]
        err_phi = np.arctan2(np.sin(phi - phi_curr), np.cos(phi - phi_curr))
        err_vec = np.array([[err_x], [err_y], [err_phi]])
        
        J = get_jacobian([t1, t2, t3], [l1, l2, l3])
        J_T = np.transpose(J)
        
        damped_inv = J_T @ np.linalg.inv((J @ J_T) + lam_sq * I)
        thetas = thetas + (damped_inv @ err_vec)
        
    return thetas

def IKsolver_Damped_Least_Squares_Memory_Clipped(x, y, φ, l1, l2, l3, lam, iters, initial_guess, joint_limits, tolerance, step_size):
    thetas = np.copy(initial_guess)
    I = np.identity(3)
    lam_sq = lam ** 2
    min_limits = np.array([[joint_limits[0][0]], [joint_limits[1][0]], [joint_limits[2][0]]])
    max_limits = np.array([[joint_limits[0][1]], [joint_limits[1][1]], [joint_limits[2][1]]])
    for _ in range(iters):
        θ1, θ2, θ3 = thetas[0, 0], thetas[1, 0], thetas[2, 0]

        x_curr, y_curr = FKsolver(θ1, θ2, θ3, l1, l2, l3)
        φ_curr = θ1 + θ2 + θ3

        err_x = x - x_curr
        err_y = y - y_curr
        err_φ = np.arctan2(np.sin(φ - φ_curr), np.cos(φ - φ_curr))
        err_vec = np.array([[err_x], [err_y], [err_φ]])

        # optional convergence check
        # if np.linalg.norm(err_vec) < tolerance:
        # thetas_copy = np.clip(thetas, min_limits, max_limits)
        # φ_copy = np.sum(thetas_copy)
        # err_φ_copy = np.arctan2(np.sin(φ_copy - φ_curr), np.cos(φ_copy - φ_curr))
        # err_vec_copy = err_vec = np.array([[err_x], [err_y], [err_φ_copy]])
        # if np.linalg.norm(err_vec_copy) < tolerance:
        # return thetas

        J = get_jacobian([θ1, θ2, θ3], [l1, l2, l3])
        J_T = np.transpose(J)

        # damped_inv = J_T @ np.linalg.inv((J @ J_T) + lam_sq * I)
        # thetas = thetas + (damped_inv @ err_vec)
        # used solve instead of inv to make math more faster
        thetas = thetas + J_T @ np.linalg.solve((J @ J_T) + lam_sq * I, err_vec) # we can also add step size here if needed at time of hardware execution

        # Constraint clipping
        thetas = np.clip(thetas, min_limits, max_limits)
    return thetas

def IKsolver_Damped_Least_Squares_Memory_Clipped_Error(x, y, φ, l1, l2, l3, lam, iters, initial_guess, joint_limits, tolerance, step_size):
    thetas = np.copy(initial_guess)
    I = np.identity(3)
    lam_sq = lam ** 2
    min_limits = np.array([[joint_limits[0][0]], [joint_limits[1][0]], [joint_limits[2][0]]])
    max_limits = np.array([[joint_limits[0][1]], [joint_limits[1][1]], [joint_limits[2][1]]])
    
    for _ in range(iters):
        θ1, θ2, θ3 = thetas[0, 0], thetas[1, 0], thetas[2, 0]

        x_curr, y_curr = FKsolver(θ1, θ2, θ3, l1, l2, l3)
        φ_curr = θ1 + θ2 + θ3

        err_x = x - x_curr
        err_y = y - y_curr
        err_φ = np.arctan2(np.sin(φ - φ_curr), np.cos(φ - φ_curr))
        err_vec = np.array([[err_x], [err_y], [err_φ]])

        # Stop if tolerance is met
        # if np.linalg.norm(err_vec) < tolerance:
        #     break

        J = get_jacobian([θ1, θ2, θ3], [l1, l2, l3])
        
        # 1. Check which joints are physically saturated
        at_min = (thetas <= min_limits).flatten()
        at_max = (thetas >= max_limits).flatten()
        
        # 2. Check which direction the math wants to push the joints
        gradient = (np.transpose(J) @ err_vec).flatten()
        
        # 3. Mute the Jacobian column for any joint pushing against a hard limit
        for k in range(3):
            if (at_min[k] and gradient[k] < 0) or (at_max[k] and gradient[k] > 0):
                J[:, k] = 0.0 

        # 4. Calculate the mathematically correct update with the muted Jacobian
        J_T = np.transpose(J)
        delta_theta = step_size * (J_T @ np.linalg.solve((J @ J_T) + lam_sq * I, err_vec))

        thetas = thetas + delta_theta
        thetas = np.clip(thetas, min_limits, max_limits)
        
    return thetas

def curve(l1, l2, l3, lam, iters):
    x_start, x_end = 1.5, 6.0
    num_waypoints = 1000
    x_path = np.linspace(x_start, x_end, num_waypoints)
    y_path = []
    for x in x_path:
        a = -3.0
        b = 11.0 * x - 5.0
        c = 2.0 * (x**2) + 4.0 * x - 21.0
        discriminant = (b**2) - (4 * a * c)
        if discriminant < 0:
            print(f"Mathematical void hit at x={x}")
            continue
        y = (-b - np.sqrt(discriminant)) / (2 * a) ################################################################################################
        y_path.append(y)
    y_path = np.array(y_path)
    y_start, y_end = y_path[0], y_path[-1]
    target_phis = []
    for i in range(len(x_path) - 1):
        phi = np.arctan2(y_path[i+1] - y_path[i], x_path[i+1] - x_path[i])
        target_phis.append(phi)
    target_phis.append(target_phis[-1])
    target_phis = np.unwrap(target_phis)
    target_phis = np.array(target_phis)

    joint_trajectory = []
    actual_x = []
    actual_y = []
    current_guess = np.array([[0.5], [0.5], [0.5]])

    ################## (either add this homing phase or take input in the og fxn)
    x_rest, y_rest = FKsolver(current_guess[0,0], current_guess[1,0], current_guess[2,0], l1, l2, l3)
    phi_rest = current_guess[0,0] + current_guess[1,0] + current_guess[2,0]
    homing_steps = 100
    x_home_path = np.linspace(x_rest, x_path[0], homing_steps)
    y_home_path = np.linspace(y_rest, y_path[0], homing_steps)
    phi_home_path = np.linspace(phi_rest, target_phis[0], homing_steps)

    limit_rad = np.deg2rad(155.0)
    joint_limits = [(-np.pi, np.pi),(-limit_rad, limit_rad),(-limit_rad, limit_rad)]

    for i in range(homing_steps):
        # solved_thetas = IKsolver_Damped_Least_Squares_Memory(x_home_path[i], y_home_path[i], phi_home_path[i], l1, l2, l3, lam, iters, current_guess)
        solved_thetas = IKsolver_Damped_Least_Squares_Memory_Clipped_Error(x_home_path[i], y_home_path[i], phi_home_path[i], l1, l2, l3, lam, iters, current_guess, joint_limits, 1e-4, 1.0)
        current_guess = solved_thetas
    ##################

    for i in range(num_waypoints):
        # solved_thetas = IKsolver_Damped_Least_Squares_Memory(x_path[i], y_path[i], target_phis[i], l1, l2, l3, lam, iters, current_guess)
        solved_thetas = IKsolver_Damped_Least_Squares_Memory_Clipped_Error(x_path[i], y_path[i], target_phis[i], l1, l2, l3, lam, iters, current_guess, joint_limits, 1e-4, 1.0)
        current_guess = solved_thetas
        joint_trajectory.append(solved_thetas.flatten())
        x_reached, y_reached = FKsolver(solved_thetas[0,0], solved_thetas[1,0], solved_thetas[2,0], l1, l2, l3)
        actual_x.append(x_reached)
        actual_y.append(y_reached)
    
    joint_trajectory = np.array(joint_trajectory)
    joint_velocities = np.diff(joint_trajectory, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(x_path, y_path, 'r--', linewidth=2, label='Target Trajectory')
    ax1.plot(actual_x, actual_y, 'b-', linewidth=4, alpha=0.5, label='Executed Path')
    ax1.scatter([x_start, x_end], [y_start, y_end], color='black', zorder=5)
    ax1.set_title("Cartesian Path Execution")
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(joint_velocities[:, 0], label='Joint 1 Velocity', color='r')
    ax2.plot(joint_velocities[:, 1], label='Joint 2 Velocity', color='g')
    ax2.plot(joint_velocities[:, 2], label='Joint 3 Velocity', color='b')
    ax2.set_title("Joint Velocity Profile (dTheta/dt)")
    ax2.set_xlabel("Waypoint Transition Index")
    ax2.set_ylabel("Angular Velocity (rad/step)")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.show() 

def curve1(l1, l2, l3, lam, iters):
    x_start, x_end = 1.5, 6.0
    num_waypoints = 1000
    x_path = np.linspace(x_start, x_end, num_waypoints)
    y_path = []
    for x in x_path:
        a = -3.0
        b = 11.0 * x - 5.0
        c = 2.0 * (x**2) + 4.0 * x - 21.0
        discriminant = (b**2) - (4 * a * c)
        if discriminant < 0:
            print(f"Mathematical void hit at x={x}")
            continue
        y = (-b + np.sqrt(discriminant)) / (2 * a) ################################################################################################
        y_path.append(y)
    y_path = np.array(y_path)
    y_start, y_end = y_path[0], y_path[-1]
    target_phis = []
    for i in range(len(x_path) - 1):
        phi = np.arctan2(y_path[i+1] - y_path[i], x_path[i+1] - x_path[i])
        target_phis.append(phi)
    target_phis.append(target_phis[-1])
    target_phis = np.unwrap(target_phis)
    target_phis = np.array(target_phis)

    joint_trajectory = []
    actual_x = []
    actual_y = []
    current_guess = np.array([[0.5], [0.5], [0.5]])

    ################## (either add this homing phase or take input in the og fxn)
    x_rest, y_rest = FKsolver(current_guess[0,0], current_guess[1,0], current_guess[2,0], l1, l2, l3)
    phi_rest = current_guess[0,0] + current_guess[1,0] + current_guess[2,0]
    homing_steps = 100
    x_home_path = np.linspace(x_rest, x_path[0], homing_steps)
    y_home_path = np.linspace(y_rest, y_path[0], homing_steps)
    phi_home_path = np.linspace(phi_rest, target_phis[0], homing_steps)

    limit_rad = np.deg2rad(155.0)
    joint_limits = [(-np.pi, np.pi),(-limit_rad, limit_rad),(-limit_rad, limit_rad)]

    for i in range(homing_steps):
        # solved_thetas = IKsolver_Damped_Least_Squares_Memory(x_home_path[i], y_home_path[i], phi_home_path[i], l1, l2, l3, lam, iters, current_guess)
        solved_thetas = IKsolver_Damped_Least_Squares_Memory_Clipped_Error(x_home_path[i], y_home_path[i], phi_home_path[i], l1, l2, l3, lam, iters, current_guess, joint_limits, 1e-4, 1.0)
        current_guess = solved_thetas
    ##################

    for i in range(num_waypoints):
        # solved_thetas = IKsolver_Damped_Least_Squares_Memory(x_path[i], y_path[i], target_phis[i], l1, l2, l3, lam, iters, current_guess)
        solved_thetas = IKsolver_Damped_Least_Squares_Memory_Clipped_Error(x_path[i], y_path[i], target_phis[i], l1, l2, l3, lam, iters, current_guess, joint_limits, 1e-4, 1.0)
        current_guess = solved_thetas
        joint_trajectory.append(solved_thetas.flatten())
        x_reached, y_reached = FKsolver(solved_thetas[0,0], solved_thetas[1,0], solved_thetas[2,0], l1, l2, l3)
        actual_x.append(x_reached)
        actual_y.append(y_reached)
    
    joint_trajectory = np.array(joint_trajectory)
    joint_velocities = np.diff(joint_trajectory, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(x_path, y_path, 'r--', linewidth=2, label='Target Trajectory')
    ax1.plot(actual_x, actual_y, 'b-', linewidth=4, alpha=0.5, label='Executed Path')
    ax1.scatter([x_start, x_end], [y_start, y_end], color='black', zorder=5)
    ax1.set_title("Cartesian Path Execution")
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(joint_velocities[:, 0], label='Joint 1 Velocity', color='r')
    ax2.plot(joint_velocities[:, 1], label='Joint 2 Velocity', color='g')
    ax2.plot(joint_velocities[:, 2], label='Joint 3 Velocity', color='b')
    ax2.set_title("Joint Velocity Profile (dTheta/dt)")
    ax2.set_xlabel("Waypoint Transition Index")
    ax2.set_ylabel("Angular Velocity (rad/step)")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.show() 

def st_line(l1, l2, l3, lam, iters):
    x_start, y_start = 6.0, 0.0
    x_end, y_end = 0.0, 8.0
    num_waypoints = 500

    x_path = np.linspace(x_start, x_end, num_waypoints)
    y_path = np.linspace(y_start, y_end, num_waypoints)
    target_phi = np.arctan2(y_end - y_start, x_end - x_start)
    joint_trajectory = []
    actual_x = []
    actual_y = []

    limit_rad = np.deg2rad(155.0)
    joint_limits = [(-np.pi, np.pi),(-limit_rad, limit_rad),(-limit_rad, limit_rad)]
    _, current_guess = IKsolver_Geometric(x_path[0], y_path[0], target_phi, l1, l2, l3)
    current_guess = np.array([[current_guess[0]], [current_guess[1]], [current_guess[2]]])

    for i in range(num_waypoints):
        # solved_thetas = IKsolver_Damped_Least_Squares_Memory(x_path[i], y_path[i], target_phi, l1, l2, l3, lam, iters, current_guess)
        solved_thetas = IKsolver_Damped_Least_Squares_Memory_Clipped_Error(x_path[i], y_path[i], target_phi, l1, l2, l3, lam, iters, current_guess, joint_limits, 1e-4, 1.0)
        current_guess = solved_thetas
        joint_trajectory.append(solved_thetas.flatten())
        x_reached, y_reached = FKsolver(solved_thetas[0,0], solved_thetas[1,0], solved_thetas[2,0], l1, l2, l3)
        actual_x.append(x_reached)
        actual_y.append(y_reached)

    joint_trajectory = np.array(joint_trajectory)
    joint_velocities = np.diff(joint_trajectory, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(x_path, y_path, 'r--', linewidth=2, label='Target Trajectory')
    ax1.plot(actual_x, actual_y, 'b-', linewidth=4, alpha=0.5, label='Executed Path')
    ax1.scatter([x_start, x_end], [y_start, y_end], color='black', zorder=5)
    ax1.set_title("Cartesian Path Execution")
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(joint_velocities[:, 0], label='Joint 1 Velocity', color='r')
    ax2.plot(joint_velocities[:, 1], label='Joint 2 Velocity', color='g')
    ax2.plot(joint_velocities[:, 2], label='Joint 3 Velocity', color='b')
    ax2.set_title("Joint Velocity Profile (dTheta/dt)")
    ax2.set_xlabel("Waypoint Transition Index")
    ax2.set_ylabel("Angular Velocity (rad/step)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    st_line(5.0, 4.0, 3.0, 0.1, 50)
    curve(7.0, 8.0, 9.0, 0.1, 50)
    curve1(7.0, 8.0, 9.0, 0.1, 50)
