import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- 1. THE MATHEMATICS ---

def get_matrix(θ_rad, a):
    c = np.cos(θ_rad)
    s = np.sin(θ_rad)
    return np.array([
        [c, -s, a * c],
        [s,  c, a * s],
        [0,  0,   1  ]
    ])

def FK_solver(thetas, link_lengths):
    T01 = get_matrix(thetas[0], link_lengths[0])
    T12 = get_matrix(thetas[1], link_lengths[1])
    T23 = get_matrix(thetas[2], link_lengths[2])

    T02 = T01 @ T12
    T03 = T01 @ T12 @ T23

    X_coords = [0.0, T01[0, 2], T02[0, 2], T03[0, 2]]
    Y_coords = [0.0, T01[1, 2], T02[1, 2], T03[1, 2]]
    
    return X_coords, Y_coords

# --- 2. THE GUI BOILERPLATE --- (written by gemini)

if __name__ == "__main__":
    # Setup the Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(bottom=0.45) # Expanded to fit 6 sliders
    ax.set_aspect('equal')
    
    # Lock the camera to the maximum possible reach (20 + 20 + 20)
    max_reach = 62 
    ax.set_xlim(-max_reach, max_reach)
    ax.set_ylim(-max_reach, max_reach)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("3-Link Planar Robot (Parametric)", fontweight='bold')

    # Initial state
    initial_thetas = [0.0, 0.0, 0.0]
    initial_L = [10.0, 10.0, 10.0]

    # Draw the initial arm state
    X, Y = FK_solver(initial_thetas, initial_L)
    arm_line, = ax.plot(X, Y, 'o-', linewidth=4, markersize=8, color='#2c3e50')
    ax.plot(0, 0, 'rs', markersize=10) # Base marker
    brush_text = ax.text(0.05, 0.95, f"Brush: ({X[-1]:.1f}, {Y[-1]:.1f})", 
                        transform=ax.transAxes, fontsize=12, fontweight='bold', 
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Sliders Setup ---
    # Axes arrays: [left, bottom, width, height]
    ax_t1 = plt.axes([0.15, 0.35, 0.70, 0.03])
    ax_t2 = plt.axes([0.15, 0.30, 0.70, 0.03])
    ax_t3 = plt.axes([0.15, 0.25, 0.70, 0.03])
    
    ax_l1 = plt.axes([0.15, 0.15, 0.70, 0.03])
    ax_l2 = plt.axes([0.15, 0.10, 0.70, 0.03])
    ax_l3 = plt.axes([0.15, 0.05, 0.70, 0.03])

    # Angle Sliders
    slider_t1 = Slider(ax_t1, 'Theta 1', -180.0, 180.0, valinit=0.0, valfmt='%0.1f°')
    slider_t2 = Slider(ax_t2, 'Theta 2', -180.0, 180.0, valinit=0.0, valfmt='%0.1f°')
    slider_t3 = Slider(ax_t3, 'Theta 3', -180.0, 180.0, valinit=0.0, valfmt='%0.1f°')

    # Length Sliders (Range 1cm to 20cm)
    slider_l1 = Slider(ax_l1, 'Length 1', 1.0, 20.0, valinit=10.0, valfmt='%0.1f cm')
    slider_l2 = Slider(ax_l2, 'Length 2', 1.0, 20.0, valinit=10.0, valfmt='%0.1f cm')
    slider_l3 = Slider(ax_l3, 'Length 3', 1.0, 20.0, valinit=10.0, valfmt='%0.1f cm')

    # Update Function
    def update(val):
        # 1. Grab angles and convert to radians
        t1 = np.radians(slider_t1.val)
        t2 = np.radians(slider_t2.val)
        t3 = np.radians(slider_t3.val)
        
        # 2. Grab current lengths
        l1 = slider_l1.val
        l2 = slider_l2.val
        l3 = slider_l3.val
        
        # 3. Recalculate kinematics
        X_new, Y_new = FK_solver([t1, t2, t3], [l1, l2, l3])
        
        # 4. Redraw the arm
        arm_line.set_data(X_new, Y_new)
        brush_text.set_text(f"Brush: ({X_new[-1]:.1f}, {Y_new[-1]:.1f})")
        fig.canvas.draw_idle()

    # Link all sliders to the update trigger
    slider_t1.on_changed(update)
    slider_t2.on_changed(update)
    slider_t3.on_changed(update)
    slider_l1.on_changed(update)
    slider_l2.on_changed(update)
    slider_l3.on_changed(update)

    plt.show()
