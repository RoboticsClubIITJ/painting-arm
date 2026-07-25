import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

def get_matrix(θ_rad, a):
    c = np.cos(θ_rad)
    s = np.sin(θ_rad)
    return np.array([
        [c, -s, a * c],
        [s,  c, a * s],
        [0,  0,   1  ]
    ])

def FKsolver(θ1, θ2, θ3, l1 ,l2, l3):
    T01 = get_matrix(θ1, l1)
    T12 = get_matrix(θ2, l2)
    T23 = get_matrix(θ3, l3)

    T03 = T01 @ T12 @ T23

    X_coords = T03[0, 2]
    Y_coords = T03[1, 2]
    
    return X_coords, Y_coords

def get_arm_points(theta1, theta2, theta3, l1, l2, l3):

    x0, y0 = 0, 0

    x1 = l1*np.cos(theta1)
    y1 = l1*np.sin(theta1)

    x2 = x1 + l2*np.cos(theta1+theta2)
    y2 = y1 + l2*np.sin(theta1+theta2)

    x3 = x2 + l3*np.cos(theta1+theta2+theta3)
    y3 = y2 + l3*np.sin(theta1+theta2+theta3)

    return [x0,x1,x2,x3],[y0,y1,y2,y3]

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

def get_jacobian(thetas, lengths):
    θ1, θ2, θ3 = thetas
    l1, l2, l3 = lengths
    
    θ12 = θ1 + θ2
    θ123 = θ1 + θ2 + θ3
    
    dx_dθ1 = -l1*np.sin(θ1) - l2*np.sin(θ12) - l3*np.sin(θ123)
    dx_dθ2 = -l2*np.sin(θ12) - l3*np.sin(θ123)
    dx_dθ3 = -l3*np.sin(θ123)
    
    dy_dθ1 = l1*np.cos(θ1) + l2*np.cos(θ12) + l3*np.cos(θ123)
    dy_dθ2 = l2*np.cos(θ12) + l3*np.cos(θ123)
    dy_dθ3 = l3*np.cos(θ123)
    
    dφ_dθ1 = 1.0
    dφ_dθ2 = 1.0
    dφ_dθ3 = 1.0
    
    J = np.array([
        [dx_dθ1, dx_dθ2, dx_dθ3],
        [dy_dθ1, dy_dθ2, dy_dθ3],
        [dφ_dθ1, dφ_dθ2, dφ_dθ3]
    ])
    
    return J

def IKsolver_Damped_Least_Squares_Memory_Clipped(x, y, φ, l1, l2, l3, lam, iters, initial_guess, joint_limits, tolerance, step_size):
    thetas = np.copy(initial_guess) # used memory
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
        #     thetas_copy = np.clip(thetas, min_limits, max_limits)
        #     φ_copy = np.sum(thetas_copy)
        #     err_φ_copy = np.arctan2(np.sin(φ_copy - φ_curr), np.cos(φ_copy - φ_curr))
        #     err_vec_copy = err_vec = np.array([[err_x], [err_y], [err_φ_copy]])
        #     if np.linalg.norm(err_vec_copy) < tolerance:
        #         return thetas
        
        J = get_jacobian([θ1, θ2, θ3], [l1, l2, l3])
        J_T = np.transpose(J)

        # damped_inv = J_T @ np.linalg.inv((J @ J_T) + lam_sq * I)
        # thetas = thetas + (damped_inv @ err_vec) 

        # used solve instead of inv to make math more faster
        thetas = thetas + J_T @ np.linalg.solve((J @ J_T) + lam_sq * I, err_vec) # we can also add step size here if needed at time of hardware execution

        # Constraint clipping
        thetas = np.clip(thetas, min_limits, max_limits)
        
    return thetas

def show_results(geom1, geom2, dls, lengths, target):
    l1, l2, l3 = lengths
    tx, ty = target

    dls = dls.flatten()

    plt.figure(figsize=(10, 8))

    # Plot Geometric Solution 1
    x, y = get_arm_points(geom1[0], geom1[1], geom1[2], l1, l2, l3)
    plt.plot(x, y, '--o', color='orange', linewidth=2, label='Geometric (Solution 1)')

    # Plot Geometric Solution 2
    x, y = get_arm_points(geom2[0], geom2[1], geom2[2], l1, l2, l3)
    plt.plot(x, y, '--o', color='purple', linewidth=2, label='Geometric (Solution 2)')

    # Plot DLS Solution
    x, y = get_arm_points(dls[0], dls[1], dls[2], l1, l2, l3)
    # plt.plot(x, y, '-o', color='red', linewidth=3, label='Damped Least Squares')

    plt.scatter(tx, ty, color='green', marker='*', s=180, label='Target')

    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title("Geometric vs Damped Least Squares IK Comparison")
    plt.show()

def solve():
    # Geometry
    l1 = float(l1_entry.get())
    l2 = float(l2_entry.get())
    l3 = float(l3_entry.get())

    # Target Space
    x = float(x_entry.get())
    y = float(y_entry.get())
    phi = np.deg2rad(float(phi_entry.get()))

    # DLS Core Parameters
    lam = float(lambda_entry.get())
    dls_iters = int(dls_iter_entry.get())

    # DLS User-Defined Constraints (Un-hardcoding)
    init_guess = np.array([
        [float(init_t1_entry.get())], 
        [float(init_t2_entry.get())], 
        [float(init_t3_entry.get())]
    ])
    
    limit_rad = np.deg2rad(float(limit_entry.get()))
    # Joint 1 unrestricted (-pi to pi), Joint 2 and 3 clipped to user limit
    joint_limits = [
        [-np.pi, np.pi], 
        [-limit_rad, limit_rad], 
        [-limit_rad, limit_rad]
    ]
    
    tolerance = float(tol_entry.get())
    step_size = float(step_entry.get())

    # Execute Solvers
    geom1, geom2 = IKsolver_Geometric(x, y, phi, l1, l2, l3)
    dls = IKsolver_Damped_Least_Squares_Memory_Clipped(
        x, y, phi, l1, l2, l3, lam, dls_iters, init_guess, joint_limits, tolerance, step_size
    )

    print("Geometric Solution 1 (Radians):\n", geom1)
    print("\nGeometric Solution 2 (Radians):\n", geom2)
    print("\nDamped Least Squares Solution (Radians):\n", dls.flatten())

    show_results(geom1, geom2, dls, [l1, l2, l3], (x, y))

# --- UI Setup ---
root = tk.Tk()
root.title("IK Comparison: Geometric vs DLS")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

# 1. Link Lengths
ttk.Label(frame, text="Link Lengths", font='Helvetica 10 bold').grid(row=0, column=0, columnspan=2, pady=(5,0))
ttk.Label(frame, text="l1").grid(row=1, column=0)
l1_entry = ttk.Entry(frame); l1_entry.insert(0, "5"); l1_entry.grid(row=1, column=1)

ttk.Label(frame, text="l2").grid(row=2, column=0)
l2_entry = ttk.Entry(frame); l2_entry.insert(0, "4"); l2_entry.grid(row=2, column=1)

ttk.Label(frame, text="l3").grid(row=3, column=0)
l3_entry = ttk.Entry(frame); l3_entry.insert(0, "3"); l3_entry.grid(row=3, column=1)

# 2. Target Position
ttk.Label(frame, text="Target Position", font='Helvetica 10 bold').grid(row=4, column=0, columnspan=2, pady=(10,0))
ttk.Label(frame, text="X").grid(row=5, column=0)
x_entry = ttk.Entry(frame); x_entry.insert(0, "7"); x_entry.grid(row=5, column=1)

ttk.Label(frame, text="Y").grid(row=6, column=0)
y_entry = ttk.Entry(frame); y_entry.insert(0, "3"); y_entry.grid(row=6, column=1)

ttk.Label(frame, text="Phi (deg)").grid(row=7, column=0)
phi_entry = ttk.Entry(frame); phi_entry.insert(0, "0"); phi_entry.grid(row=7, column=1)

# 3. DLS Parameters
ttk.Label(frame, text="DLS Basic Parameters", font='Helvetica 10 bold').grid(row=8, column=0, columnspan=2, pady=(10,0))
ttk.Label(frame, text="Lambda").grid(row=9, column=0)
lambda_entry = ttk.Entry(frame); lambda_entry.insert(0, "0.1"); lambda_entry.grid(row=9, column=1)

ttk.Label(frame, text="Iterations").grid(row=10, column=0)
dls_iter_entry = ttk.Entry(frame); dls_iter_entry.insert(0, "100"); dls_iter_entry.grid(row=10, column=1)

# 4. DLS Un-hardcoded Constraints
ttk.Label(frame, text="DLS Memory & Constraints", font='Helvetica 10 bold').grid(row=11, column=0, columnspan=2, pady=(10,0))
ttk.Label(frame, text="Initial Guess θ1 (rad)").grid(row=12, column=0)
init_t1_entry = ttk.Entry(frame); init_t1_entry.insert(0, "0.5"); init_t1_entry.grid(row=12, column=1)

ttk.Label(frame, text="Initial Guess θ2 (rad)").grid(row=13, column=0)
init_t2_entry = ttk.Entry(frame); init_t2_entry.insert(0, "0.5"); init_t2_entry.grid(row=13, column=1)

ttk.Label(frame, text="Initial Guess θ3 (rad)").grid(row=14, column=0)
init_t3_entry = ttk.Entry(frame); init_t3_entry.insert(0, "0.5"); init_t3_entry.grid(row=14, column=1)

ttk.Label(frame, text="Joint Limit θ2, θ3 (±deg)").grid(row=15, column=0)
limit_entry = ttk.Entry(frame); limit_entry.insert(0, "155"); limit_entry.grid(row=15, column=1)

ttk.Label(frame, text="Tolerance").grid(row=16, column=0)
tol_entry = ttk.Entry(frame); tol_entry.insert(0, "0.0001"); tol_entry.grid(row=16, column=1)

ttk.Label(frame, text="Step Size").grid(row=17, column=0)
step_entry = ttk.Entry(frame); step_entry.insert(0, "1.0"); step_entry.grid(row=17, column=1)

# Solve Button
ttk.Button(frame, text="Solve & Compare", command=solve).grid(row=18, column=0, columnspan=2, pady=15)

root.mainloop()

# FABRIK is out of scope
"""
Pros and cons
Geom: 
    # pro:
    -> Instantaneous computation O(1).
    -> Gives you every possible valid configuration, letting your planner pick the best one. 
    -> Mathematically safe to singularities.
    ~ con:
    -> Highly rigid:- hardcoded formulas for 3 link arm
    -> Configuration Flipping:- This method just gives two distinct, isolated solutions but has no built-in memory of where the arm was in the previous millisecond. It might choose the a solution for point A and suddenly switch to the opposite sign solution for point B(just next waypoint). This way robot will attempt to violently flip its entire physical structure 180 degrees in a fraction of a second mid-stroke. It will rip hardware apart, tear the canvas, amny many more undefined behaviours
Grad des:
    # pro:
    -> Computationally Cheap:- Matrix transposition is practically free on a processor compared to calculating inverses.
    -> Mathematical Safety: It never divides by a determinant. Therefore, it is entirely safe from singularity explosions.
    ~ con:
    -> Agonizingly Slow: It can take hundreds or thousands of iterations to converge, making it useless for fast, high-resolution drawing.
    -> The Parameter Trap: It is completely dependent on tuning the learning rate α. If α is too high, the arm stutters and oscillates wildly. If α is too low, the arm stalls out before reaching the target. Tuning this for a dynamic path is a waste of time.
    -> Convergence jitter:- In last few iterations when q is just going to reach q_desired then α changes too slowly sometimes changing angles values abruptly reducing arm's performance
Pinv:
    # pro:
    -> Quadratic Convergence: It snaps to the target in incredibly few iterations (often under 10).
    -> Direct Paths: It takes the most mathematically direct path in joint space toward the goal.
    -> Step-Size Control: By introducing a step_size multiplier, you can forcefully restrict the algorithm from taking massive, non-linear leaps in Cartesian space, allowing you to manually force the brush to move in straight, predictable lines.
    ~ con:
    -> High Processor Load: Computing Singular Value Decomposition (SVD) for the pseudoinverse is computationally heavy for a microcontroller therfore we need to make an offline IK solver.
    -> The Singularity Bomb: If a target is even 0.1 mm outside your maximum reach, or if the arm stretches completely straight, the matrix determinant hits zero. The internal math explodes, calculating joint velocities in the millions. Code outputs NaN and the system crashes.
Damped_square:
    # pro:
    -> Total Mathematical Stability: It never crashes. When approaching an unreachable target or a stretched-out singularity, it gracefully slows down instead of exploding.
    -> Adaptive Speed: When far from a boundary, it runs as fast as the pure Pseudoinverse.
    -> The Built-In Brake: It completely eliminates the need for the manual step_size hack we used in the Pseudoinverse. The λ²I term natively acts as a mathematical floor, naturally preventing the joint velocities from ever leaping to dangerous magnitudes.
    ~ con:
    -> Slight Accuracy Trade-off: To maintain that bulletproof stability, we have to trade a microscopic fraction of accuracy. It might stop a fraction of a millimeter short of absolute zero error.
"""
