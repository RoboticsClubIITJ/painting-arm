# Physical parameters, kinematic limits, and tuning constants for the 3-DOF planar drawing arm.
import numpy as np

# --- HARDWARE PORTS & SETTINGS ---
DXL_COM_PORT = '/dev/ttyUSB1'
ARDUINO_COM_PORT = '/dev/ttyUSB0'   
PEN_UP_ANGLE = 45
PEN_DOWN_ANGLE = 95

# --- LINK GEOMETRY ---
L1 = 15.95
L2 = 15.05
L3 = 11.0938
L_TUPLE = (L1, L2, L3)
MAX_REACH = L1 + L2 + L3

# --- JOINT LIMITS (Hardware limits in radians from Calibration) ---
JOINT_LIMITS = [
    (-2.416, 2.252),   # Shoulder
    (-2.233, 1.927),   # Elbow
    (-2.056, 2.141),   # Wrist
]

# --- TASK-SPACE / JOINT-SPACE LIMITS ---
MAX_LINEAR_SPEED = 15.0       # cm/s
MAX_JOINT_VEL = 1.0           # rad/s
MAX_JOINT_ACC = 0.2           # rad/s^2

# --- TIMING ---
FPS = 30
DT = 1.0 / FPS
REVIEW_WINDOW = 5.0           # seconds

# --- IK SOLVER (Damped Least Squares) ---
IK_MAX_ITER = 100
IK_TOL = 1e-2
IK_DAMPING = 0.05
IK_MAX_STEP = 0.30            # rad per iteration

# --- TRAJECTORY SHAPING ---
SAFE_DECEL_RATE = 3.5         # cm/s^2
MIN_SPEED_FLOOR = 0.5         # cm/s
NEAR_TARGET_DIST = 0.1        # cm
NEAR_TARGET_SPEED_FLOOR = 0.01
FAR_TARGET_SPEED_FLOOR_FRAC = 0.05
TRAVEL_SPEED_MULT = 1.5
PAUSE_DURATION = 0.2          # seconds

# --- SAVITZKY-GOLAY FILTERING ---
SG_WINDOW_LENGTH = 15
SG_POLYORDER = 3

# --- XDOG VISION PIPELINE ---
XDOG_SIGMA = 1.4
XDOG_K_SIGMA = 1.6
XDOG_EPSILON = 0.01
XDOG_PHI = 20
XDOG_GAMMA = 0.98
XDOG_AUTO_TUNE = True

# --- VISION / CONTOUR PROCESSING ---
MIN_CONTOUR_ARC_LEN = 12.0
MIN_CONTOUR_POINTS = 4
SPLINE_SMOOTHING = 2.0
APPROX_POLY_EPSILON = 1.0

# --- IMAGE -> ROBOT WORKSPACE MAPPING ---
TARGET_CANVAS_W = 20.0
TARGET_CANVAS_H = 20.0
CANVAS_CENTER_X = 10.0 + (TARGET_CANVAS_W / 2.0)
CANVAS_CENTER_Y = 0.0
