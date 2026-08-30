# Physical parameters, kinematic limits, and tuning constants for the 3-DOF planar drawing arm.
import numpy as np
# Link Geometry
L1 = 15.9
L2 = 15.05
L3 = 11.1
L_TUPLE = (L1, L2, L3)
MAX_REACH = L1 + L2 + L3

# Joint Limits
# JOINT_LIMITS = [
#     (np.radians(-135), np.radians(135)),   # Shoulder
#     (np.radians(-130), np.radians(130)),   # Elbow
#     (np.radians(-130), np.radians(130)),   # Wrist
# ]

JOINT_LIMITS = [
    (-2.416, 2.252),   # Shoulder
    (-2.233, 1.927),   # Elbow
    (-2.056, 2.141),   # Wrist
]
# Task-Space / Joint-Space Limits
MAX_LINEAR_SPEED = 15.0       # cm/s
MAX_JOINT_VEL = 1.0           # rad/s
MAX_JOINT_ACC = 0.2           # rad/s^2

# Timing
FPS = 30
DT = 1.0 / FPS
REVIEW_WINDOW = 5.0           # seconds

# IK Solver - Damped Least Squares
IK_MAX_ITER = 100
IK_TOL = 1e-2
IK_DAMPING = 0.05
IK_MAX_STEP = 0.30            # rad per iteration

# Trajectory Shaping
SAFE_DECEL_RATE = 3.5         # cm/s^2
MIN_SPEED_FLOOR = 0.5         # cm/s
NEAR_TARGET_DIST = 0.1        # cm
NEAR_TARGET_SPEED_FLOOR = 0.01
FAR_TARGET_SPEED_FLOOR_FRAC = 0.05
TRAVEL_SPEED_MULT = 1.5
PAUSE_DURATION = 0.2 