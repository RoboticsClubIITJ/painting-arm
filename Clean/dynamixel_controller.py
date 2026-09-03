"""
dynamixel_controller.py
------------------------
XM540-W270-T motors (Protocol 2.0) ko control karne ke liye wrapper.
3 motors: Shoulder (ID 1), Elbow (ID 2), Wrist (ID 3) - same order jaisa
tumhare sim ke q_current = [t1, t2, t3] mein hai.

Install pehle:
    pip install dynamixel-sdk
"""

import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite
from dynamixel_sdk import COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD

# ============================================================
# === DYNAMIXEL SETTINGS (XM540-W270-T, Protocol 2.0) ===
# ============================================================
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000        # tumhare tested motor code se confirmed
DEVICE_NAME = "COM6"      # tumhare tested motor code se confirmed

DXL_IDS = [1, 2, 3]       # [Shoulder, Elbow, Wrist]

# --- Control Table Addresses (X-series, Protocol 2.0 common table) ---
ADDR_OPERATING_MODE       = 11
ADDR_TORQUE_ENABLE        = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY     = 112
ADDR_GOAL_POSITION        = 116
ADDR_PRESENT_POSITION     = 132

LEN_GOAL_POSITION = 4

TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0
OPERATING_MODE_POSITION = 3   # Position Control Mode -> 0-4095 ticks, single turn (0-360 deg)

TICKS_PER_REV = 4096
TICKS_PER_RAD = TICKS_PER_REV / (2 * np.pi)   # ~651.74 ticks per radian

# ============================================================
# === CALIBRATION - ASSEMBLY KE BAAD YE VALUES SET KARNI HAI ===
# ============================================================
# HOME_TICKS[i] = raw "Present Position" tick jab us joint ka angle
# tumhare kinematics convention mein 0 rad ho (matlab jab poora arm
# q = [0, 0, 0] wali pose mein khada ho, jaisa forward_kinematics_2d
# mein define hai).
#
# Calibrate kaise karo: Dynamixel Wizard 2.0 kholo, torque off karke
# motor ko haath se us pose mein le jaao jo simulation ke q=[0,0,0]
# arm shape se match kare, fir wahan "Present Position" field mein
# jo number dikh raha hai wahi yahan daalo.
HOME_TICKS = [2048, 2048, 2048]     # <-- PLACEHOLDER, calibrate karke badlo

# SIGN[i] = +1 ya -1. Test script chalane pe agar motor ulti
# direction mein ghoome jab q[i] positive diya, toh us index ka
# sign -1 kar do.
SIGN = [1, 1, 1]                    # <-- PLACEHOLDER, calibrate karke badlo

# Safety margin - joint ke mechanical limit ke bilkul edge tak kabhi na jaane do
TICK_MIN, TICK_MAX = 50, 4045


def angle_to_tick(joint_idx, angle_rad):
    """Ek joint ka radian angle -> raw Dynamixel tick (calibration + clip ke saath)"""
    raw = HOME_TICKS[joint_idx] + SIGN[joint_idx] * angle_rad * TICKS_PER_RAD
    return int(np.clip(round(raw), TICK_MIN, TICK_MAX))


class DynamixelArm:
    def __init__(self, device=DEVICE_NAME, baudrate=BAUDRATE, ids=DXL_IDS):
        self.ids = ids
        self.port = PortHandler(device)
        self.packet = PacketHandler(PROTOCOL_VERSION)

        if not self.port.openPort():
            raise RuntimeError(f"Port {device} open nahi ho paaya. Cable/port name check karo.")
        if not self.port.setBaudRate(baudrate):
            raise RuntimeError(f"Baudrate {baudrate} set nahi hua.")

        self.group_write = GroupSyncWrite(self.port, self.packet,
                                           ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            self._write1(dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
            # Profile velocity/accel = 0 -> motor "max speed, no on-board ramping" use karta hai.
            # Ye zaroori hai kyunki hum already sim mein poori smooth trajectory (velocity-limited,
            # accel-limited) generate kar chuke hai aur har 33ms (30 FPS) pe naya target bhej rahe
            # hai - agar yahan bhi motor apna slow profile lagayega toh wo target se peeche reh
            # jaayega aur drawing lag/distort karegi.
            self._write4(dxl_id, ADDR_PROFILE_VELOCITY, 0)
            self._write4(dxl_id, ADDR_PROFILE_ACCELERATION, 0)
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        print("Dynamixel arm ready.")

    def _write1(self, dxl_id, addr, value):
        result, error = self.packet.write1ByteTxRx(self.port, dxl_id, addr, value)
        self._check(dxl_id, result, error)

    def _write4(self, dxl_id, addr, value):
        result, error = self.packet.write4ByteTxRx(self.port, dxl_id, addr, value)
        self._check(dxl_id, result, error)

    def _check(self, dxl_id, result, error):
        if result != COMM_SUCCESS:
            print(f"[ID {dxl_id}] Comm error: {self.packet.getTxRxResult(result)}")
        elif error != 0:
            print(f"[ID {dxl_id}] Dynamixel error: {self.packet.getRxPacketError(error)}")

    def move_to_angles(self, q):
        """
        q = [t1, t2, t3] radians, tumhare forward_kinematics_2d wale convention mein.
        Teeno motors ko ek hi sync write packet mein target position bhejta hai
        (isse sab joints ek saath move start karte hai, ek ek karke nahi).
        """
        self.group_write.clearParam()
        for i, dxl_id in enumerate(self.ids):
            tick = angle_to_tick(i, q[i])
            param = [DXL_LOBYTE(DXL_LOWORD(tick)), DXL_HIBYTE(DXL_LOWORD(tick)),
                     DXL_LOBYTE(DXL_HIWORD(tick)), DXL_HIBYTE(DXL_HIWORD(tick))]
            self.group_write.addParam(dxl_id, param)
        self.group_write.txPacket()

    def read_present_angles(self):
        """Debug ke liye - abhi motors kaha khade hai, radians mein"""
        angles = []
        for i, dxl_id in enumerate(self.ids):
            pos, result, error = self.packet.read4ByteTxRx(self.port, dxl_id, ADDR_PRESENT_POSITION)
            self._check(dxl_id, result, error)
            rad = (pos - HOME_TICKS[i]) / (SIGN[i] * TICKS_PER_RAD)
            angles.append(rad)
        return angles

    def torque_off(self):
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

    def close(self):
        self.torque_off()
        self.port.closePort()