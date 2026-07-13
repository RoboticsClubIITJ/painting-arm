import numpy as np
import matplotlib.pyplot as plt


class PlanarRobotArm:
    """A 3-link planar robot arm (e.g. shoulder-elbow-wrist)."""

    def __init__(self, link_lengths):
        self.l1, self.l2, self.l3 = link_lengths

    def compute_fk(self, thetas_deg):
        """
        Forward Kinematics: joint angles -> (x, y) position of each joint.
        For a planar chain, each joint's position is just the running sum
        of angles applied to each link -- no matrices needed.
        """
        t1, t2, t3 = np.radians(thetas_deg)
        running_angle = 0.0
        x, y = 0.0, 0.0
        positions = [(x, y)]

        for theta, length in zip((t1, t2, t3), (self.l1, self.l2, self.l3)):
            running_angle += theta
            x += length * np.cos(running_angle)
            y += length * np.sin(running_angle)
            positions.append((x, y))

        return positions

    def compute_ik(self, target_x, target_y, target_phi_deg, elbow_up=False):
        """
        Analytical Inverse Kinematics: (x, y, brush angle) -> joint angles.
        Uses kinematic decoupling + the Law of Cosines.

        elbow_up=False -> elbow bends "down" (default, matches original code)
        elbow_up=True  -> mirror-image elbow configuration
        """
        L1, L2, L3 = self.l1, self.l2, self.l3
        phi = np.radians(target_phi_deg)

        # 1. Decouple: find where the WRIST needs to be, ignoring link 3
        #    for a moment (back off from the target by link 3's length).
        wrist_x = target_x - L3 * np.cos(phi)
        wrist_y = target_y - L3 * np.sin(phi)
        dist = np.hypot(wrist_x, wrist_y)

        # 2. Reachability check (triangle inequality for a 2-link arm)
        if dist > L1 + L2:
            raise ValueError(f"Target ({target_x}, {target_y}) is out of reach for this brush angle.")
        if dist < abs(L1 - L2):
            raise ValueError(f"Target ({target_x}, {target_y}) is too close for this brush angle.")

        # 3. Law of Cosines -> elbow angle (theta2)
        cos_t2 = np.clip((dist**2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
        theta2 = np.arccos(cos_t2)
        if elbow_up:
            theta2 = -theta2

        # 4. Triangle geometry -> shoulder angle (theta1)
        alpha = np.arctan2(wrist_y, wrist_x)
        beta = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        theta1 = alpha - beta

        # 5. Whatever orientation is left over is the wrist angle (theta3)
        theta3 = phi - theta1 - theta2

        return [np.degrees(theta1), np.degrees(theta2), np.degrees(theta3)]


if __name__ == "__main__":
    print("--- Bob Ross Robot Arm Command Center ---")

    def get_float(prompt, default):
        try:
            return float(input(prompt))
        except ValueError:
            print(f"Invalid input. Using default: {default}")
            return default

    l1 = get_float("Enter length for Link 1 (Shoulder) in cm (e.g., 15): ", 15.0)
    l2 = get_float("Enter length for Link 2 (Elbow) in cm (e.g., 12): ", 12.0)
    l3 = get_float("Enter length for Link 3 (Brush) in cm (e.g., 8): ", 8.0)

    arm = PlanarRobotArm([l1, l2, l3])
    print(f"\nArm configured! Reach: {l1 + l2 + l3} cm.")

    target_x = get_float("Enter Target X coordinate (e.g., 10): ", 10.0)
    target_y = get_float("Enter Target Y coordinate (e.g., 15): ", 15.0)
    target_angle = get_float("Enter Desired Brush Angle in degrees (e.g., 45): ", 45.0)

    try:
        angles = arm.compute_ik(target_x, target_y, target_angle)
        print("\nCalculated Motor Angles:")
        print(f"Shoulder (Theta 1): {angles[0]:.2f}°")
        print(f"Elbow (Theta 2):    {angles[1]:.2f}°")
        print(f"Wrist (Theta 3):    {angles[2]:.2f}°")
    except ValueError as e:
        print(f"\nERROR: {e}")
        raise SystemExit

    # Draw the result to confirm it worked
    pos = arm.compute_fk(angles)
    X, Y = zip(*pos)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(target_x, target_y, 'yo', markersize=20, alpha=0.5, label='Target Point')
    ax.plot(X, Y, 'o-', lw=5, ms=8, color='#2c3e50', label='Robot Arm')
    ax.plot(0, 0, 'rs', ms=10, label='Base')
    ax.plot(X[-1], Y[-1], 'g*', ms=15, label='Brush Tip')

    angle_rad = np.radians(target_angle)
    ax.plot([target_x, target_x + 5 * np.cos(angle_rad)],
             [target_y, target_y + 5 * np.sin(angle_rad)],
             'k--', lw=2, label='Requested Angle')

    reach = l1 + l2 + l3 + 5
    ax.set(title='Analytical Inverse Kinematics', xlabel='X (cm)', ylabel='Y (cm)',
           aspect='equal', xlim=(-reach, reach), ylim=(-reach, reach))
    ax.grid(True, ls='--')
    ax.legend()
    plt.show()