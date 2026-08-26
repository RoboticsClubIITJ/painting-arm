"""
Tkinter + matplotlib GUI for the 3-DOF drawing arm simulator.

This module is orchestration only: it wires together vision.py
(image -> paths), kinematics.py (FK/IK, arm rendering), and
trajectory.py (velocity-profiled joint trajectories), plus the
animation loop and Savitzky-Golay post-smoothing. No CV or kinematics
math lives here anymore.

NAYA: dynamixel_controller.py se DynamixelArm bhi wire kiya hai, taaki
sim ke saath physical motors bhi move hon.
"""
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import scipy.signal as signal

from Configurations import (
    MAX_REACH, MAX_LINEAR_SPEED, MAX_JOINT_VEL, DT, FPS, REVIEW_WINDOW, 
    TRAVEL_SPEED_MULT, PAUSE_DURATION, SG_WINDOW_LENGTH,
    SG_POLYORDER, JOINT_LIMITS,
    XDOG_SIGMA, XDOG_K_SIGMA, XDOG_EPSILON, XDOG_PHI, XDOG_GAMMA,
)
from Kinematics import FK, arm_link_positions, clip_to_joint_limits
from Trajectory import generate_continuous_trajectory
from Vision import image_to_robot_paths
from dynamixel_controller import DynamixelArm   # <-- NAYA


class Planar3DOFSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3-DOF Drawing (Kinematic Limits + Face Portrait Mode)")
        self.q_current = np.array([np.pi / 4, -np.pi / 2, np.pi / 4], dtype=float)

        self.raw_image = None
        self.final_cv_paths = []

        self.traj_q = []
        self.traj_dq = []
        self.traj_draw_flags = []
        self.traj_pen_status = []
        self.traj_idx = 0
        self.is_running = False
        self.stroke_index = 0

        # ---- NAYA: hardware arm connect karo ----
        # Agar port connect nahi hota, app crash nahi hogi - sim-only chalegi.
        self.arm = None
        try:
            self.arm = DynamixelArm()
        except Exception as e:
            print(f"[HARDWARE] Dynamixel arm connect nahi hui, sim-only mode: {e}")

        self.setup_gui()
        self.setup_plots()

        self.root.bind('<Return>', self.update_preview)

    # GUI construction
    def setup_gui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(control_frame, text="Upload Image", command=self.upload_image).pack(pady=10, fill=tk.X)

        xdog_frame = ttk.LabelFrame(control_frame, text="XDoG Parameters")
        xdog_frame.pack(pady=10, fill=tk.X)

        self.sigma_var = tk.DoubleVar(value=XDOG_SIGMA)
        self.lbl_sigma = ttk.Label(xdog_frame, text=f"Sigma: {self.sigma_var.get():.2f}")
        self.lbl_sigma.pack()
        ttk.Scale(xdog_frame, from_=1.1, to=3.5, variable=self.sigma_var, orient=tk.HORIZONTAL, command=lambda v: self.lbl_sigma.config(text=f"Sigma: {float(v):.2f}")).pack(fill=tk.X, padx=5)

        self.k_sigma_var = tk.DoubleVar(value=XDOG_K_SIGMA)
        self.lbl_k_sigma = ttk.Label(xdog_frame, text=f"k-Sigma: {self.k_sigma_var.get():.2f}")
        self.lbl_k_sigma.pack()
        ttk.Scale(xdog_frame, from_=2.5, to=5.5, variable=self.k_sigma_var, orient=tk.HORIZONTAL, command=lambda v: self.lbl_k_sigma.config(text=f"k-Sigma: {float(v):.2f}")).pack(fill=tk.X, padx=5)

        self.epsilon_var = tk.DoubleVar(value=XDOG_EPSILON)
        self.lbl_epsilon = ttk.Label(xdog_frame, text=f"Epsilon: {self.epsilon_var.get():.3f}")
        self.lbl_epsilon.pack()
        ttk.Scale(xdog_frame, from_=0.005, to=0.08, variable=self.epsilon_var, orient=tk.HORIZONTAL, command=lambda v: self.lbl_epsilon.config(text=f"Epsilon: {float(v):.3f}")).pack(fill=tk.X, padx=5)

        self.phi_var = tk.IntVar(value=XDOG_PHI)
        self.lbl_phi = ttk.Label(xdog_frame, text=f"Phi: {self.phi_var.get()}")
        self.lbl_phi.pack()
        ttk.Scale(xdog_frame, from_=20, to=60, variable=self.phi_var, orient=tk.HORIZONTAL, command=lambda v: self.lbl_phi.config(text=f"Phi: {int(float(v))}")).pack(fill=tk.X, padx=5)

        self.gamma_var = tk.DoubleVar(value=XDOG_GAMMA)
        self.lbl_gamma = ttk.Label(xdog_frame, text=f"Gamma: {self.gamma_var.get():.2f}")
        self.lbl_gamma.pack()
        ttk.Scale(xdog_frame, from_=0.88, to=0.98, variable=self.gamma_var, orient=tk.HORIZONTAL, command=lambda v: self.lbl_gamma.config(text=f"Gamma: {float(v):.2f}")).pack(fill=tk.X, padx=5)

        ttk.Button(xdog_frame, text="Update Preview (Enter)", command=self.update_preview).pack(pady=10, fill=tk.X)

        self.btn_start = ttk.Button(control_frame, text="Start Draw", command=self.start_drawing, state=tk.DISABLED)
        self.btn_start.pack(pady=15, fill=tk.X)

        ttk.Button(control_frame, text="Reset", command=self.reset_sim).pack(pady=5, fill=tk.X)

        review_frame = ttk.LabelFrame(control_frame, text="Timeline Review (Post-Draw)")
        review_frame.pack(pady=15, fill=tk.X)

        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_slider = ttk.Scale(review_frame, from_=0.0, to=5.0, variable=self.timeline_var, orient=tk.HORIZONTAL, command=self.on_scroll_timeline)
        self.timeline_slider.pack(fill=tk.X, padx=5, pady=5)
        self.timeline_slider.state(['disabled'])

        self.timeline_label = ttk.Label(review_frame, text="Window: 0.0s - 5.0s")
        self.timeline_label.pack(pady=(0, 5))

        # ---- NAYA: hardware status dikhane ke liye ----
        hw_text = "HARDWARE: CONNECTED" if self.arm else "HARDWARE: NOT CONNECTED (sim-only)"
        ttk.Label(control_frame, text=hw_text, font=("Arial", 9, "italic")).pack(pady=(0, 5))

        self.status_var = tk.StringVar(value="STATUS: IDLE")
        ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(pady=10)

    def setup_plots(self):
        self.fig = plt.figure(figsize=(14, 7), dpi=100)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.4)

        self.ax_arm = self.fig.add_subplot(gs[:, 0])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.ax_arm.set_title("Task Space Cartesian Paths", fontweight='bold')
        self.ax_arm.set_aspect('equal')
        self.ax_arm.set_xlim(-5, MAX_REACH + 5)
        self.ax_arm.set_ylim(-15, 15)
        self.ax_arm.grid(True, linestyle='--')

        self.arm_line, = self.ax_arm.plot([], [], 'o-', lw=6, color='#2c3e50', zorder=4)
        self.brush_line, = self.ax_arm.plot([], [], 'o-', lw=4, color='#e74c3c', zorder=5)

        self.pen_status_text = self.ax_arm.text(0.03, 0.95, '', transform=self.ax_arm.transAxes,
                                                  fontsize=11, fontweight='bold', color='#c0392b',
                                                  bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        self.preview_lines = []
        self.trail_lines = []
        self.current_stroke_x = []
        self.current_stroke_y = []

        self.vel_axes = [
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[1, 1]),
            self.fig.add_subplot(gs[2, 1])
        ]

        joint_names = ['Shoulder', 'Elbow', 'Wrist']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        self.v_lines = []

        for i, ax in enumerate(self.vel_axes):
            ax.set_title(f"{joint_names[i]} Velocity (Limit: {MAX_JOINT_VEL} rad/s)", fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--')
            ax.set_ylim(-0.4, 0.4)
            ax.axhline(MAX_JOINT_VEL, color='red', linestyle=':', alpha=0.5)
            ax.axhline(-MAX_JOINT_VEL, color='red', linestyle=':', alpha=0.5)

            line, = ax.plot([], [], lw=2, color=colors[i])
            self.v_lines.append(line)

        self.vel_axes[-1].set_xlabel("Time (sec)")

        self.draw_arm()

    # State management
    def reset_sim(self):
        self.is_running = False
        self.traj_q = []
        self.traj_dq = []
        self.traj_draw_flags = []
        self.traj_pen_status = []
        self.traj_idx = 0
        self.stroke_index = 0
        self.raw_image = None
        self.final_cv_paths = []

        for p in self.preview_lines:
            p.remove()
        self.preview_lines.clear()

        for t in self.trail_lines:
            t.remove()
        self.trail_lines.clear()

        self.current_stroke_x = []
        self.current_stroke_y = []
        for v in self.v_lines:
            v.set_data([], [])

        self.timeline_slider.state(['disabled'])
        self.timeline_var.set(0.0)
        self.timeline_label.config(text="Window: 0.0s - 5.0s")

        self.btn_start.config(state=tk.DISABLED)
        self.pen_status_text.set_text('')
        self.status_var.set("STATUS: RESET")
        self.draw_arm()

    def on_scroll_timeline(self, val):
        if self.is_running:
            return

        t_start = float(val)
        t_end = t_start + REVIEW_WINDOW

        self.timeline_label.config(text=f"Window: {t_start:.1f}s - {t_end:.1f}s")
        for ax in self.vel_axes:
            ax.set_xlim(t_start, t_end)

        self.canvas.draw_idle()

    # Image upload + preview (vision.py does the actual CV work)
    def upload_image(self):
        if self.is_running:
            return
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not filepath:
            return

        self.reset_sim()
        self.raw_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        self.status_var.set("STATUS: IMAGE UPLOADED. ADJUST SLIDERS & ENTER.")
        self.update_preview()

    def update_preview(self, event=None):
        if self.raw_image is None or self.is_running:
            return

        self.status_var.set("STATUS: UPDATING PREVIEW...")
        self.root.update()

        sigma = self.sigma_var.get()
        k_sigma = self.k_sigma_var.get()
        epsilon = self.epsilon_var.get()
        phi = self.phi_var.get()
        gamma = self.gamma_var.get()

        self.final_cv_paths = image_to_robot_paths(self.raw_image, sigma, k_sigma, epsilon, phi, gamma)

        for p in self.preview_lines:
            p.remove()
        self.preview_lines.clear()

        for mapped in self.final_cv_paths:
            p_arr = np.array(mapped)

            if len(p_arr) < 2:
                continue

            l, = self.ax_arm.plot(p_arr[:, 0], p_arr[:, 1], color='gray', linestyle='-', alpha=0.6)
            self.preview_lines.append(l)

        self.canvas.draw_idle()
        self.status_var.set("STATUS: PREVIEW READY. PRESS START.")
        self.btn_start.config(state=tk.NORMAL)

    # Trajectory generation + playback
    def start_drawing(self):
        if not self.final_cv_paths:
            return
        self.btn_start.config(state=tk.DISABLED)
        self.generate_full_trajectory(self.final_cv_paths)

    def generate_full_trajectory(self, cv_paths):
        self.status_var.set("STATUS: PLANNING TASK SPACE PATH...")
        self.root.update()

        self.traj_q = []
        self.traj_dq = []
        self.traj_draw_flags = []
        self.traj_pen_status = []
        self.traj_idx = 0
        self.stroke_index = 0
        self.current_stroke_x = []
        self.current_stroke_y = []

        q_sim = self.q_current.copy()
        pause_frames = int(PAUSE_DURATION * FPS)

        def smooth_segment(traj_list):
            if len(traj_list) > SG_WINDOW_LENGTH:
                arr = np.array(traj_list)
                for joint_idx in range(3):
                    arr[:, joint_idx] = signal.savgol_filter(
                        arr[:, joint_idx],
                        window_length=SG_WINDOW_LENGTH,
                        polyorder=SG_POLYORDER
                    )
                    arr[:, joint_idx] = np.clip(
                        arr[:, joint_idx],
                        JOINT_LIMITS[joint_idx][0],
                        JOINT_LIMITS[joint_idx][1]
                    )
                return [row for row in arr]
            return traj_list

        for i, path in enumerate(cv_paths):
            curr_x, curr_y = FK(q_sim)
            start_x, start_y = path[0]

            # --- Pen Up & Travel ---
            travel_traj = generate_continuous_trajectory([(curr_x, curr_y), (start_x, start_y)], q_sim, v_max=MAX_LINEAR_SPEED * TRAVEL_SPEED_MULT)
            travel_traj = smooth_segment(travel_traj)

            if travel_traj:
                if i > 0:
                    self.traj_q.extend([q_sim] * pause_frames)
                    self.traj_draw_flags.extend([False] * pause_frames)
                    self.traj_pen_status.extend(["LIFTING PEN"] * pause_frames)

                self.traj_q.extend(travel_traj)
                self.traj_draw_flags.extend([False] * len(travel_traj))
                self.traj_pen_status.extend(["PEN UP (TRAVEL)"] * len(travel_traj))
                q_sim = travel_traj[-1]

            self.traj_q.extend([q_sim] * pause_frames)
            self.traj_draw_flags.extend([False] * pause_frames)
            self.traj_pen_status.extend(["LOWERING PEN"] * pause_frames)

            # --- Pen Down & Draw ---
            draw_traj = generate_continuous_trajectory(path, q_sim, v_max=MAX_LINEAR_SPEED)
            draw_traj = smooth_segment(draw_traj)
            if draw_traj:
                self.traj_q.extend(draw_traj)
                self.traj_draw_flags.extend([True] * len(draw_traj))
                self.traj_pen_status.extend(["PEN DOWN (DRAWING)"] * len(draw_traj))
                q_sim = draw_traj[-1]

        if not self.traj_q:
            self.status_var.set("STATUS: NO VALID PATHS")
            return

        self.traj_q = np.array(self.traj_q)

        self.traj_dq = np.gradient(self.traj_q, DT, axis=0)

        total_duration = len(self.traj_q) * DT
        max_start_time = max(0.0, total_duration - REVIEW_WINDOW)
        self.timeline_slider.config(to=max_start_time)

        for ax in self.vel_axes:
            ax.set_ylim(-0.8, 0.8)

        self.is_running = True
        self.status_var.set("STATUS: EXECUTING...")
        self.run_animation_loop()

    # Rendering + animation
    def draw_arm(self):
        p0, p1, p2, p3 = arm_link_positions(self.q_current)
        self.arm_line.set_data([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]])
        self.brush_line.set_data([p2[0], p3[0]], [p2[1], p3[1]])
        self.canvas.draw_idle()
        return p3

    def run_animation_loop(self):
        if self.is_running and self.traj_idx < len(self.traj_q):
            self.q_current = self.traj_q[self.traj_idx]
            is_drawing = self.traj_draw_flags[self.traj_idx]
            pen_status = self.traj_pen_status[self.traj_idx]
            p3 = self.draw_arm()

            # ---- NAYA: physical motors ko bhi wahi angle bhejo jo sim mein dikh raha hai ----
            if self.arm:
                self.arm.move_to_angles(self.q_current)

            self.pen_status_text.set_text(f"Z-AXIS: {pen_status}")

            if is_drawing:
                self.current_stroke_x.append(p3[0])
                self.current_stroke_y.append(p3[1])
                if len(self.trail_lines) == self.stroke_index:
                    new_trail, = self.ax_arm.plot([], [], 'b-', lw=1.5, zorder=2)
                    self.trail_lines.append(new_trail)
                self.trail_lines[self.stroke_index].set_data(self.current_stroke_x, self.current_stroke_y)
            else:
                if len(self.current_stroke_x) > 0:
                    self.current_stroke_x, self.current_stroke_y = [], []
                    self.stroke_index += 1

            t_arr = np.arange(self.traj_idx + 1) * DT
            current_time = t_arr[-1]

            t_min = max(0, current_time - REVIEW_WINDOW)
            t_max = max(REVIEW_WINDOW, current_time)

            # ---- FIX: sirf visible window ka data plot karo, poori history nahi -
            # warna lambi trajectory pe har frame heavier hota jaata hai aur
            # asli drawing speed slow padne lagti hai (regardless of MAX_LINEAR_SPEED).
            window_start_idx = max(0, self.traj_idx + 1 - int(REVIEW_WINDOW * FPS) - 5)
            t_window = t_arr[window_start_idx:]
            dq_window = self.traj_dq[window_start_idx:self.traj_idx + 1]

            for i, ax in enumerate(self.vel_axes):
                self.v_lines[i].set_data(t_window, dq_window[:, i])
                ax.set_xlim(t_min, t_max)

            self.timeline_var.set(t_min)
            self.timeline_label.config(text=f"Window: {t_min:.1f}s - {t_max:.1f}s")

            self.traj_idx += 1
            self.root.after(int(DT * 1000), self.run_animation_loop)

        elif self.is_running:
            self.is_running = False
            self.pen_status_text.set_text("DRAWING COMPLETE")
            self.status_var.set("STATUS: COMPLETE. USE SLIDER TO REVIEW.")
            self.timeline_slider.state(['!disabled'])

    def on_close(self):
        # ---- NAYA: band karte waqt torque off + port close, motors safe rahenge ----
        if self.arm:
            self.arm.close()
        self.root.destroy()
        plt.close('all')
