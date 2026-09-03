"""
dynamixel_controller.py

A thin, GUI-friendly wrapper around the Dynamixel SDK calls that used
to live only inside Calibration.py's top-level script code.

Nothing here runs at import time and nothing blocks: opening the port,
calibrating the zero reference, and enabling torque are all explicit
method calls, so a missing or unplugged robot never crashes a caller
(e.g. Pen_up_down.py) -- it just leaves you in sim-only mode until you
retry.

Address map, joint ordering, invert flags and relative safety limits
are copied as-is from Calibration.py (DXL_IDs = [0, 1, 2] -> Shoulder
(XM540), Elbow (XM430), Wrist (XM430)) so behaviour matches the
existing console calibration tool exactly.
"""
import math
import dynamixel_sdk as dxl


class DynamixelArm:
    DXL_IDS = [0, 1, 2]              # Shoulder (XM540), Elbow (XM430), Wrist (XM430)
    PROTOCOL_VERSION = 2.0

    ADDR_OPERATING_MODE = 11
    EXT_POSITION_CONTROL_MODE = 4    # Multi-turn position control
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132

    # True if increasing ticks produces CW rotation physically, per joint.
    JOINT_INVERT = [True, True, True]

    # Safety window around the recorded zero, in encoder ticks per joint.
    REL_LIMIT_1 = [-1575, -1456, -1340]
    REL_LIMIT_2 = [1468, 1256, 1396]

    TICKS_PER_REV = 4096

    def __init__(self, port='COM3', baudrate=1000000):
        self.port_name = port
        self.baudrate = baudrate
        self.port_handler = dxl.PortHandler(port)
        self.packet_handler = dxl.PacketHandler(self.PROTOCOL_VERSION)

        self.connected = False
        self.calibrated = False
        self.zero_ticks = [0] * len(self.DXL_IDS)
        self.max_ticks = [0] * len(self.DXL_IDS)
        self.min_ticks = [0] * len(self.DXL_IDS)

        if not self.port_handler.openPort():
            raise ConnectionError(f"Could not open port '{port}'.")
        if not self.port_handler.setBaudRate(baudrate):
            self.port_handler.closePort()
            raise ConnectionError(f"Could not set baudrate {baudrate} on '{port}'.")

        self.connected = True
        self._set_torque(False)
        self._set_operating_mode(self.EXT_POSITION_CONTROL_MODE)

    # ---- low-level SDK helpers (unchanged logic from Calibration.py) ----
    def _read_signed_position(self, dxl_id):
        pos, _, _ = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, self.ADDR_PRESENT_POSITION
        )
        if pos > 2147483647:
            pos -= 4294967296
        return pos

    def _write_signed_position(self, dxl_id, tick):
        tick = int(tick)
        if tick < 0:
            tick += 4294967296
        self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, self.ADDR_GOAL_POSITION, tick
        )

    def _set_torque(self, enable):
        for dxl_id in self.DXL_IDS:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, self.ADDR_TORQUE_ENABLE, 1 if enable else 0
            )

    def _set_operating_mode(self, mode):
        for dxl_id in self.DXL_IDS:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, self.ADDR_OPERATING_MODE, mode
            )

    def _radians_to_ticks(self, radians, zero_offset, invert):
        direction = -1 if invert else 1
        ticks_offset = int(direction * radians * (self.TICKS_PER_REV / (2 * math.pi)))
        return zero_offset + ticks_offset

    def _ticks_to_radians(self, tick, zero_offset, invert):
        direction = -1 if invert else 1
        return direction * (tick - zero_offset) * (2 * math.pi / self.TICKS_PER_REV)

    # ---- calibration ----
    def calibrate_zero(self):
        """
        Call this once the arm has been positioned BY HAND straight
        along the X-axis (q = [0, 0, 0]). Records the current encoder
        ticks as the zero reference and derives safety limits around
        it -- this is the GUI equivalent of Calibration.py's
        `input("Move the ENTIRE arm...")` console step, and must stay
        a deliberate, human-triggered action for the same safety
        reason: the software has no way to know the arm's real pose
        until you tell it.
        """
        self.zero_ticks = [self._read_signed_position(i) for i in self.DXL_IDS]
        self.max_ticks = []
        self.min_ticks = []
        for i in range(len(self.DXL_IDS)):
            limit_a = self.zero_ticks[i] + self.REL_LIMIT_1[i]
            limit_b = self.zero_ticks[i] + self.REL_LIMIT_2[i]
            self.max_ticks.append(max(limit_a, limit_b))
            self.min_ticks.append(min(limit_a, limit_b))
        self.calibrated = True
        self._set_torque(True)
        return list(self.zero_ticks)

    # ---- motion ----
    def move_to_angles(self, q):
        """
        q: sequence of 3 joint angles in radians [shoulder, elbow, wrist].
        No-ops (returns False) if calibrate_zero() hasn't run yet --
        driving un-referenced ticks would be unsafe, so callers should
        treat a False return as "hardware not ready", not an error.
        """
        if not self.calibrated:
            return False
        for i, dxl_id in enumerate(self.DXL_IDS):
            goal_tick = self._radians_to_ticks(q[i], self.zero_ticks[i], self.JOINT_INVERT[i])
            clamped = max(min(goal_tick, self.max_ticks[i]), self.min_ticks[i])
            self._write_signed_position(dxl_id, clamped)
        return True

    def go_to_zero(self):
        if not self.calibrated:
            return False
        for i, dxl_id in enumerate(self.DXL_IDS):
            self._write_signed_position(dxl_id, self.zero_ticks[i])
        return True

    # ---- feedback ----
    def get_present_angles(self):
        """
        Reads the servos' ACTUAL present position back over the bus and
        converts to radians. This is the live signal from the hardware
        -- useful for confirming the physical arm is actually tracking
        what was commanded, independent of the simulated/commanded
        state the GUI keeps internally.
        """
        if not self.calibrated:
            return None
        angles = []
        for i, dxl_id in enumerate(self.DXL_IDS):
            tick = self._read_signed_position(dxl_id)
            angles.append(self._ticks_to_radians(tick, self.zero_ticks[i], self.JOINT_INVERT[i]))
        return angles

    # ---- shutdown ----
    def close(self):
        if self.connected:
            try:
                self._set_torque(False)
            except Exception:
                pass
            self.port_handler.closePort()
            self.connected = False
