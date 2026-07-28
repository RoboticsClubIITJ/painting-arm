import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- 1. CORE LOGIC (Homogeneous Transformation Matrix Approach) ---

def get_transform_matrix(theta_rad, length):
    """
    Creates a 3x3 homogeneous transformation matrix for a single link frame.
    T = [ cos(th)  -sin(th)   length*cos(th) ]
        [ sin(th)   cos(th)   length*sin(th) ]
        [    0         0            1        ]
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, -s, length * c],
        [s,  c, length * s],
        [0,  0,     1     ]
    ])

def forward_kinematics(q, lengths):
    """
    Computes joint positions and end-effector pose using Homogeneous Transformations.
    q: [theta1, theta2, theta3] in radians
    lengths: [l1, l2, l3]
    """
    # Relative Transformation Matrices
    T01 = get_transform_matrix(q[0], lengths[0])
    T12 = get_transform_matrix(q[1], lengths[1])
    T23 = get_transform_matrix(q[2], lengths[2])

    # Cumulative Transformation Matrices (Matrix Multiplication @)
    T02 = T01 @ T12
    T03 = T02 @ T23

    # Extract positions from matrix translation column (last column)
    p0 = np.array([0.0, 0.0])
    p1 = T01[:2, 2]
    p2 = T02[:2, 2]
    p3 = T03[:2, 2]

    # Extract orientation angle from final rotation matrix elements
    orientation_rad = np.arctan2(T03[1, 0], T03[0, 0])
    
    # End-effector pose: [x, y, orientation_degrees]
    ee = np.array([p3[0], p3[1], np.degrees(orientation_rad)])

    return {
        'joints': np.array([p0, p1, p2, p3]),
        'ee': ee,
        'orientation_rad': orientation_rad
    }


# --- 2. INTERACTIVE GUI ---

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.38)
    
    # Plot Setup
    max_reach = 30.0
    ax.set_xlim(-max_reach, max_reach)
    ax.set_ylim(-max_reach, max_reach)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_title("3-Link Kinematics (Homogeneous Transformations)", fontsize=12, fontweight='bold')

    # Workspace Circle
    workspace_circle = plt.Circle((0, 0), max_reach, color='gray', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(workspace_circle)

    # Initial Values
    init_q_deg = [30.0, 45.0, -30.0]
    init_l = [10.0, 10.0, 10.0]
    
    res = forward_kinematics(np.radians(init_q_deg), init_l)
    joints = res['joints']

    # Draw Robot Elements
    arm_plot, = ax.plot(joints[:, 0], joints[:, 1], 'o-', lw=3, ms=6, color='#2B5797', label='Arm Links')
    ax.plot(0, 0, '^k', ms=12, label='Base')
    
    # End-Effector Arrow
    ee_x, ee_y, _ = res['ee']
    ee_rad = res['orientation_rad']
    ee_arrow = ax.quiver(ee_x, ee_y, np.cos(ee_rad), np.sin(ee_rad), scale=15, color='red', width=0.008)

    # Info Text Display
    info_box = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='linen', alpha=0.9))

    def update_info_text(ee):
        info_box.set_text(f"End-Effector Pose:\nX: {ee[0]:6.2f}\nY: {ee[1]:6.2f}\nPhi: {ee[2]:5.1f}°")

    update_info_text(res['ee'])

    # --- SLIDERS ---
    slider_ax_t1 = plt.axes([0.15, 0.24, 0.65, 0.025])
    slider_ax_t2 = plt.axes([0.15, 0.20, 0.65, 0.025])
    slider_ax_t3 = plt.axes([0.15, 0.16, 0.65, 0.025])

    slider_ax_l1 = plt.axes([0.15, 0.10, 0.65, 0.025])
    slider_ax_l2 = plt.axes([0.15, 0.06, 0.65, 0.025])
    slider_ax_l3 = plt.axes([0.15, 0.02, 0.65, 0.025])

    s_t1 = Slider(slider_ax_t1, 'θ1 (°)', -180, 180, valinit=init_q_deg[0], valfmt='%.0f°')
    s_t2 = Slider(slider_ax_t2, 'θ2 (°)', -180, 180, valinit=init_q_deg[1], valfmt='%.0f°')
    s_t3 = Slider(slider_ax_t3, 'θ3 (°)', -180, 180, valinit=init_q_deg[2], valfmt='%.0f°')

    s_l1 = Slider(slider_ax_l1, 'L1', 2.0, 10.0, valinit=init_l[0], valfmt='%.1f')
    s_l2 = Slider(slider_ax_l2, 'L2', 2.0, 10.0, valinit=init_l[1], valfmt='%.1f')
    s_l3 = Slider(slider_ax_l3, 'L3', 2.0, 10.0, valinit=init_l[2], valfmt='%.1f')

    # Update Callback
    def update(val):
        q_rad = np.radians([s_t1.val, s_t2.val, s_t3.val])
        lengths = [s_l1.val, s_l2.val, s_l3.val]

        result = forward_kinematics(q_rad, lengths)
        pts = result['joints']
        ee = result['ee']
        orient = result['orientation_rad']

        arm_plot.set_data(pts[:, 0], pts[:, 1])
        ee_arrow.set_offsets([ee[0], ee[1]])
        ee_arrow.set_UVC(np.cos(orient), np.sin(orient))

        update_info_text(ee)
        fig.canvas.draw_idle()

    for s in [s_t1, s_t2, s_t3, s_l1, s_l2, s_l3]:
        s.on_changed(update)

    plt.show()