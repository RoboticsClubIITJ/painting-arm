import time
import math
import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite
from dynamixel_sdk import COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD
from Configurations import DXL_COM_PORT

# ============================================================
# === DYNAMIXEL SETTINGS (XM540-W270-T / XM430, Protocol 2.0) ===
# ============================================================
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000

DXL_IDS = [0, 1, 2] # [Shoulder, Elbow, Wrist]

# --- Control Table Addresses ---
ADDR_OPERATING_MODE       = 11
ADDR_TORQUE_ENABLE        = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY     = 112
ADDR_GOAL_POSITION        = 116
ADDR_PRESENT_POSITION     = 132

LEN_GOAL_POSITION = 4

TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0
EXT_POSITION_CONTROL_MODE = 4   # Multi-turn position control

TICKS_PER_REV = 4096
TICKS_PER_RAD = 4096 / (2 * math.pi)

# Physical joint mapping and safety limits (from Calibration)
JOINT_INVERT = [True, True, True]
REL_LIMIT_1 = [-1575, -1456, -1340]
REL_LIMIT_2 = [1468, 1256, 1396]

class DynamixelArm:
    def __init__(self, device=DXL_COM_PORT, baudrate=BAUDRATE, ids=DXL_IDS):
        self.ids = ids
        self.port = PortHandler(device)
        self.packet = PacketHandler(PROTOCOL_VERSION)

        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port {device}. Check COM port and power.")
        if not self.port.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate {baudrate}.")

        self.group_write = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        
        # Internal state
        self.home_ticks = [0, 0, 0]
        self.max_ticks = [0, 0, 0]
        self.min_ticks = [0, 0, 0]
        
        # 1. Disable torque for setup
        self.torque_off()
        
        # 2. Set Operating Mode to Extended Position Control Mode
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_OPERATING_MODE, EXT_POSITION_CONTROL_MODE)
            self._write4(dxl_id, ADDR_PROFILE_VELOCITY, 0)
            self._write4(dxl_id, ADDR_PROFILE_ACCELERATION, 0)
            
        print(f"Dynamixel communication ready on {device}. Proceeding to Calibration...")
        self.calibrate()

    def calibrate(self):
        """Interactive calibration to set zero positions dynamically."""
        print("\n" + "="*40)
        print("         ZERO CALIBRATION")
        print("="*40)
        input(">> Move the ENTIRE arm straight along the X-axis (0 radians).\n>> Press Enter when ready...")
        
        for i, dxl_id in enumerate(self.ids):
            self.home_ticks[i] = self.read_signed_position(dxl_id)
            
        print(f"Zero offsets recorded: {self.home_ticks}")

        print("\n--- APPLYING RELATIVE SAFETY LIMITS ---")
        for i, dxl_id in enumerate(self.ids):
            limit_a = self.home_ticks[i] + REL_LIMIT_1[i]
            limit_b = self.home_ticks[i] + REL_LIMIT_2[i]
            self.max_ticks[i] = max(limit_a, limit_b)
            self.min_ticks[i] = min(limit_a, limit_b)
            print(f"Servo {dxl_id} ticks limited to: {self.min_ticks[i]} to {self.max_ticks[i]}")

        time.sleep(1)
        self.torque_on()
        print("\nTorque ENABLED. Arm is locked and ready for drawing.")
        print("="*40 + "\n")

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

    def read_signed_position(self, dxl_id):
        pos, result, error = self.packet.read4ByteTxRx(self.port, dxl_id, ADDR_PRESENT_POSITION)
        self._check(dxl_id, result, error)
        if pos > 2147483647:
            pos -= 4294967296
        return pos
        
    def angle_to_tick(self, joint_idx, angle_rad):
        direction = -1 if JOINT_INVERT[joint_idx] else 1
        ticks_offset = int(direction * angle_rad * TICKS_PER_RAD)
        goal_tick = self.home_ticks[joint_idx] + ticks_offset
        
        # Clamp to dynamically set safety limits
        clamped_tick = max(min(goal_tick, self.max_ticks[joint_idx]), self.min_ticks[joint_idx])
        return clamped_tick

    def move_to_angles(self, q):
        """
        q = [t1, t2, t3] in radians.
        Uses GroupSyncWrite for simultaneous smooth movement of all joints.
        """
        self.group_write.clearParam()
        for i, dxl_id in enumerate(self.ids):
            tick = self.angle_to_tick(i, q[i])
            
            # Convert signed 32-bit tick to unsigned representation for writing
            if tick < 0:
                tick += 4294967296
                
            param = [DXL_LOBYTE(DXL_LOWORD(tick)), DXL_HIBYTE(DXL_LOWORD(tick)),
                     DXL_LOBYTE(DXL_HIWORD(tick)), DXL_HIBYTE(DXL_HIWORD(tick))]
            self.group_write.addParam(dxl_id, param)
            
        self.group_write.txPacket()

    def torque_off(self):
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

    def torque_on(self):
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

    def close(self):
        self.torque_off()
        self.port.closePort()
