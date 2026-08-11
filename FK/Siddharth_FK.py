import tkinter as tk
from tkinter import messagebox
import numpy as np

# Embedding Matplotlib into Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# 1. YOUR CORE MATH FUNCTION (UNTOUCHED)
# ==========================================
def calculatefk(l1,l2,l3,a1,a2,a3):
    b1 = np.radians(a1)
    b2 = b1+np.radians(a2)
    b3 = b2+np.radians(a3)

    x=l1*np.cos(b1)+l2*np.cos(b2)+l3*np.cos(b3)
    y=l1*np.sin(b1)+l2*np.sin(b2)+l3*np.sin(b3)

    return x,y

# ==========================================
# 2. INTERACTION & SIMULATION LOGIC
# ==========================================
def on_calculate_and_plot():
    try:
        # Get numeric values from GUI inputs
        l1 = float(entry_l1.get())
        l2 = float(entry_l2.get())
        l3 = float(entry_l3.get())
        
        a1 = float(entry_a1.get())
        a2 = float(entry_a2.get())
        a3 = float(entry_a3.get())
        
        # 1. Run your exact Forward Kinematics function for the final point
        x_final, y_final = calculatefk(l1, l2, l3, a1, a2, a3)
        
        # Update the text labels on the GUI panel
        label_result_x.config(text=f"X Coordinate: {x_final:.2f} cm")
        label_result_y.config(text=f"Y Coordinate: {y_final:.2f} cm")
        
        # 2. Calculate individual joint locations using your exact angle definition logic
        b1 = np.radians(a1)
        b2 = b1 + np.radians(a2)
        
        x0, y0 = 0.0, 0.0                             # Base origin
        x1 = l1 * np.cos(b1)                          # Elbow joint
        y1 = l1 * np.sin(b1)
        x2 = x1 + l2 * np.cos(b2)                     # Wrist joint
        y2 = y1 + l2 * np.sin(b2)
        x3, y3 = x_final, y_final                     # Brush tip (End Effector)
        
        # 3. Update the Matplotlib Plot Live
        ax.clear()  # Clear the old arm drawing
        
        # Draw the grid background and axes lines
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        
        # Plot the structural bones of the arm
        ax.plot([x0, x1, x2, x3], [y0, y1, y2, y3], color='#2c3e50', linewidth=4, marker='o', 
                markersize=8, markerfacecolor='#e74c3c', markeredgecolor='black', label="Arm Links")
        
        # Highlight the brush tip painting tool with a distinct green dot
        ax.plot(x3, y3, marker='s', markersize=10, markerfacecolor='#2ecc71', markeredgecolor='black', label="Brush Tip")
        
        # Dynamically scale the graph limits to match the length size perfectly
        max_reach = l1 + l2 + l3 + 5
        ax.set_xlim(-max_reach, max_reach)
        ax.set_ylim(-max_reach, max_reach)
        ax.set_aspect('equal', adjustable='box')  # Forces a 1:1 square aspect ratio
        ax.set_title("3-DOF Planar Robot Arm Real-time Mesh", fontsize=10, fontweight='bold')
        ax.legend(loc="upper right")
        
        # Tell the Tkinter canvas container to refresh and display the update
        canvas.draw()
        
    except ValueError:
        messagebox.showerror("Input Error", "Please fill in all length and angle boxes with valid numbers!")

def reset_lengths():
    entry_l1.delete(0, tk.END)
    entry_l2.delete(0, tk.END)
    entry_l3.delete(0, tk.END)

def reset_angles():
    entry_a1.delete(0, tk.END)
    entry_a2.delete(0, tk.END)
    entry_a3.delete(0, tk.END)
    
    # Re-initialize labels and wipe the plot canvas back to home state
    label_result_x.config(text="X Coordinate: --")
    label_result_y.config(text="Y Coordinate: --")
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
root.title("Inter-IIT: Bob Ross 3-DOF Robot Arm Sim Environment")
root.geometry("900x500")  # Wide canvas display structure

# Create side-by-side main container layouts
left_control_panel = tk.Frame(root, padx=15, pady=10)
left_control_panel.pack(side="left", fill="y")

right_simulation_panel = tk.Frame(root, bg="white", padx=5, pady=5)
right_simulation_panel.pack(side="right", fill="both", expand=True)

# --- LEFT PANEL CONFIGURATION: USER INPUT FIELDS ---

# Section 1 Frame: Hardware Scale Parameters (FIXED FONT SYNTAX HERE)
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

# Section 2 Frame: Servo Angle Configurations (FIXED FONT SYNTAX HERE)
frame_angles = tk.LabelFrame(left_control_panel, text=" 2. Joint Rotations (Degrees) ", padx=10, pady=10, font=("Arial", 10, "bold"))
frame_angles.pack(fill="x", pady=5)

tk.Label(frame_angles, text="Angle 1 (Shoulder):").grid(row=0, column=0, sticky="w", pady=2)
entry_a1 = tk.Entry(frame_angles, width=8)
entry_a1.grid(row=0, column=1, padx=5, pady=2)

tk.Label(frame_angles, text="Angle 2 (Elbow):").grid(row=1, column=0, sticky="w", pady=2)
entry_a2 = tk.Entry(frame_angles, width=8)
entry_a2.grid(row=1, column=1, padx=5, pady=2)

tk.Label(frame_angles, text="Angle 3 (Wrist):").grid(row=2, column=0, sticky="w", pady=2)
entry_a3 = tk.Entry(frame_angles, width=8)
entry_a3.grid(row=2, column=1, padx=5, pady=2)

btn_reset_a = tk.Button(frame_angles, text="Reset Angles", command=reset_angles, fg="white", bg="#c0392b")
btn_reset_a.grid(row=1, column=2, padx=15)

# Section 3: Computational Tool Execution Buttons
btn_plot = tk.Button(left_control_panel, text="Compute & Render Kinematics", command=on_calculate_and_plot, 
                     bg="#2980b9", fg="white", font=("Arial", 11, "bold"), pady=6)
btn_plot.pack(fill="x", pady=10)

# Section 4 Frame: Live Math Coordinates Readout Display (FIXED FONT SYNTAX HERE)
frame_results = tk.LabelFrame(left_control_panel, text=" 3. System Coordinate Position ", padx=10, pady=10, font=("Arial", 10, "bold"))
frame_results.pack(fill="x", pady=5)

label_result_x = tk.Label(frame_results, text="X Coordinate: --", font=("Arial", 11, "bold"), fg="#2c3e50")
label_result_x.pack(anchor="w", pady=2)

label_result_y = tk.Label(frame_results, text="Y Coordinate: --", font=("Arial", 11, "bold"), fg="#2c3e50")
label_result_y.pack(anchor="w", pady=2)


# --- RIGHT PANEL CONFIGURATION: EMBEDDED MATPLOTLIB AXIS MAP ---
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal', adjustable='box')
ax.set_title("3-DOF Planar Robot Arm Real-time Mesh", fontsize=10, fontweight='bold')

# Bridge the figure layout context directly inside the Tkinter graphic layout canvas
canvas = FigureCanvasTkAgg(fig, master=right_simulation_panel)
canvas.get_tk_widget().pack(fill="both", expand=True)

# Keep system process loop open
root.mainloop()
