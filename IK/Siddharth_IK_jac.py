import tkinter as tk
from tkinter import messagebox
import numpy as np

# Embedding Matplotlib into Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# 1. ITERATIVE JACOBIAN PSEUDOINVERSE LOGIC
# ==========================================
def calculateIK_Jacobian(l1, l2, l3, targetx, targety, max_iterations=500, tolerance=0.01):
    """
    Calculates Inverse Kinematics using the 2x3 Jacobian Pseudoinverse method.
    Iteratively converges on the target (targetx, targety).
    """
    # Step 1: Initialize starting guess for joint angles (in radians)
    t1 = np.radians(30.0)
    t2 = np.radians(30.0)
    t3 = np.radians(30.0)
    
    # Damping factor / step size adjustment to maintain numeric stability
    step_size = 0.5 

    for _ in range(max_iterations):
        # Step 2: Compute current End-Effector position via Forward Kinematics
        x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2) + l3 * np.cos(t1 + t2 + t3)
        y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2) + l3 * np.sin(t1 + t2 + t3)
        
        # Step 3: Calculate spatial error vector (Delta X, Delta Y)
        error_x = targetx - x
        error_y = targety - y
        error = np.array([error_x, error_y])
        
        # Termination condition: Exit early if target is reached within tolerance
        if np.linalg.norm(error) < tolerance:
            return t1, t2, t3
            
        # Step 4: Construct the 2x3 Jacobian Matrix via analytical partial derivatives
        # Row 1: dX / dTheta
        j11 = -l1 * np.sin(t1) - l2 * np.sin(t1 + t2) - l3 * np.sin(t1 + t2 + t3)
        j12 = -l2 * np.sin(t1 + t2) - l3 * np.sin(t1 + t2 + t3)
        j13 = -l3 * np.sin(t1 + t2 + t3)
        
        # Row 2: dY / dTheta
        j21 = l1 * np.cos(t1) + l2 * np.cos(t1 + t2) + l3 * np.cos(t1 + t2 + t3)
        j22 = l2 * np.cos(t1 + t2) + l3 * np.cos(t1 + t2 + t3)
        j23 = l3 * np.cos(t1 + t2 + t3)
        
        J = np.array([[j11, j12, j13],
                      [j21, j22, j23]])
        
        # Step 5: Compute the 3x2 Moore-Penrose Pseudoinverse Matrix
        J_pinv = np.linalg.pinv(J)
        
        # Step 6: Calculate joint angular updates ("nudges")
        d_theta = J_pinv.dot(error)
        
        # Step 7: Apply the adjustments to the current state variables
        t1 += step_size * d_theta[0]
        t2 += step_size * d_theta[1]
        t3 += step_size * d_theta[2]

    # Post-Loop Validation: Verify if the system actually reached the coordinate boundary
    final_x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2) + l3 * np.cos(t1 + t2 + t3)
    final_y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2) + l3 * np.sin(t1 + t2 + t3)
    
    if np.hypot(targetx - final_x, targety - final_y) > (tolerance * 10):
        return None, None, None
        
    return t1, t2, t3

# ==========================================
# 2. INTERACTION & SIMULATION LOGIC
# ==========================================
def on_calculate_and_plot():
    try:
        # Fetch configurations from UI Entry instances
        l1 = float(entry_l1.get())
        l2 = float(entry_l2.get())
        l3 = float(entry_l3.get())
        
        targetx = float(entry_tx.get())
        targety = float(entry_ty.get())
        
        # 1. Run the iterative Jacobian Pseudoinverse solver routine
        t1, t2, t3 = calculateIK_Jacobian(l1, l2, l3, targetx, targety)
        
        # Error handling if system cannot resolve target boundaries
        if t1 is None:
            messagebox.showerror("Kinematics Error", "Target position is out of reach or mathematically unstable!")
            return
            
        # Convert output rad metrics back into degrees for readability
        deg1 = np.degrees(t1)
        deg2 = np.degrees(t2)
        deg3 = np.degrees(t3)
        
        # Update UI numeric indicator displays
        label_result_a1.config(text=f"Angle 1 (Shoulder): {deg1:.2f}°")
        label_result_a2.config(text=f"Angle 2 (Elbow): {deg2:.2f}°")
        label_result_a3.config(text=f"Angle 3 (Wrist): {deg3:.2f}°")
        
        # 2. Chronological geometric node rebuilding for linear mapping
        x0, y0 = 0.0, 0.0
        x1 = l1 * np.cos(t1)
        y1 = l1 * np.sin(t1)
        x2 = x1 + l2 * np.cos(t1 + t2)
        y2 = y1 + l2 * np.sin(t1 + t2)
        x3 = x2 + l3 * np.cos(t1 + t2 + t3)
        y3 = y2 + l3 * np.sin(t1 + t2 + t3)
        
        # 3. Live Matplotlib Frame Redrawing
        ax.clear()  
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        
        # Plot physical linked skeletal mesh
        ax.plot([x0, x1, x2, x3], [y0, y1, y2, y3], color='#2c3e50', linewidth=4, marker='o', 
                markersize=8, markerfacecolor='#e74c3c', markeredgecolor='black', label="Arm Mesh Links")
        
        # Highlight targeting intercept objective point
        ax.plot(targetx, targety, marker='X', markersize=10, markerfacecolor='#9b59b6', markeredgecolor='black', label="Target Point")
        
        # Bound visualization coordinates responsively
        max_reach = l1 + l2 + l3 + 5
        ax.set_xlim(-max_reach, max_reach)
        ax.set_ylim(-max_reach, max_reach)
        ax.set_aspect('equal', adjustable='box')  
        ax.set_title("3-DOF Jacobian Pseudoinverse Real-time Mesh", fontsize=10, fontweight='bold')
        ax.legend(loc="upper right")
        
        canvas.draw()
        
    except ValueError:
        messagebox.showerror("Input Error", "Please fill all fields with valid numbers!")

def reset_lengths():
    entry_l1.delete(0, tk.END)
    entry_l2.delete(0, tk.END)
    entry_l3.delete(0, tk.END)

def reset_targets():
    entry_tx.delete(0, tk.END)
    entry_ty.delete(0, tk.END)
    
    label_result_a1.config(text="Angle 1 (Shoulder): --")
    label_result_a2.config(text="Angle 2 (Elbow): --")
    label_result_a3.config(text="Angle 3 (Wrist): --")
    ax.clear()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal', adjustable='box')
    canvas.draw()

# ==========================================
# 3. WINDOW & GRID BUILDING BLOCK LAYOUT
# ==========================================
root = tk.Tk()
root.title("Inter-IIT: Bob Ross 3-DOF Jacobian Pseudoinverse Environment")
root.geometry("950x520")

# Setup container layouts
left_control_panel = tk.Frame(root, padx=15, pady=10)
left_control_panel.pack(side="left", fill="y")

right_simulation_panel = tk.Frame(root, bg="white", padx=5, pady=5)
right_simulation_panel.pack(side="right", fill="both", expand=True)

# --- LEFT PANEL CONFIGURATION ---

# Frame 1: Link Structural Parameters
frame_lengths = tk.LabelFrame(left_control_panel, text=" 1. Link Dimensions (cm) ", padx=10, pady=10, font=("Arial", 10, "bold"))
frame_lengths.pack(fill="x", pady=5)

tk.Label(frame_lengths, text="Link 1:").grid(row=0, column=0, sticky="w", pady=2)
entry_l1 = tk.Entry(frame_lengths, width=8)
entry_l1.grid(row=0, column=1, padx=5, pady=2)

tk.Label(frame_lengths, text="Link 2:").grid(row=1, column=0, sticky="w", pady=2)
entry_l2 = tk.Entry(frame_lengths, width=8)
entry_l2.grid(row=1, column=1, padx=5, pady=2)

tk.Label(frame_lengths, text="Link 3:").grid(row=2, column=0, sticky="w", pady=2)
entry_l3 = tk.Entry(frame_lengths, width=8)
entry_l3.grid(row=2, column=1, padx=5, pady=2)

btn_reset_l = tk.Button(frame_lengths, text="Reset Lengths", command=reset_lengths, fg="white", bg="#c0392b")
btn_reset_l.grid(row=1, column=2, padx=15)

# Frame 2: Operational Target Setup
frame_targets = tk.LabelFrame(left_control_panel, text=" 2. Coordinate Tracking Target ", padx=10, pady=10, font=("Arial", 10, "bold"))
frame_targets.pack(fill="x", pady=5)

tk.Label(frame_targets, text="Target X:").grid(row=0, column=0, sticky="w", pady=2)
entry_tx = tk.Entry(frame_targets, width=8)
entry_tx.grid(row=0, column=1, padx=5, pady=2)

tk.Label(frame_targets, text="Target Y:").grid(row=1, column=0, sticky="w", pady=2)
entry_ty = tk.Entry(frame_targets, width=8)
entry_ty.grid(row=1, column=1, padx=5, pady=2)

btn_reset_t = tk.Button(frame_targets, text="Reset Target", command=reset_targets, fg="white", bg="#c0392b")
btn_reset_t.grid(row=0, column=2, rowspan=2, padx=15)

# Execution Engine Trigger (Calls the interactive Jacobian matrix calculations)
btn_plot = tk.Button(left_control_panel, text="Compute & Iterate Jacobian IK", command=on_calculate_and_plot, 
                     bg="#9b59b6", fg="white", font=("Arial", 11, "bold"), pady=6)
btn_plot.pack(fill="x", pady=10)

# Frame 3: System Outputs Read Panel Configuration
frame_results = tk.LabelFrame(left_control_panel, text=" 3. Calculated Servo Angle Values ", padx=10, pady=10, font=("Arial", 10, "bold"))
frame_results.pack(fill="x", pady=5)

label_result_a1 = tk.Label(frame_results, text="Angle 1 (Shoulder): --", font=("Arial", 11, "bold"), fg="#2c3e50")
label_result_a1.pack(anchor="w", pady=2)

label_result_a2 = tk.Label(frame_results, text="Angle 2 (Elbow): --", font=("Arial", 11, "bold"), fg="#2c3e50")
label_result_a2.pack(anchor="w", pady=2)

label_result_a3 = tk.Label(frame_results, text="Angle 3 (Wrist): --", font=("Arial", 11, "bold"), fg="#2c3e50")
label_result_a3.pack(anchor="w", pady=2)


# --- RIGHT PANEL CONFIGURATION ---
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal', adjustable='box')
ax.set_title("3-DOF Planar Robot Arm Real-time Simulation Mesh", fontsize=10, fontweight='bold')

canvas = FigureCanvasTkAgg(fig, master=right_simulation_panel)
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()