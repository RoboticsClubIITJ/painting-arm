import numpy as np
import matplotlib.pyplot as plt


class PlanarRobotArm:
    """A 3-link planar robot arm (e.g. shoulder-elbow-wrist)."""

    def __init__(self, link_lengths):
        self.l1, self.l2, self.l3 = link_lengths

    def compute_fk(self, thetas_deg):
        """
        Forward Kinematics: joint angles -> (x, y) position of each joint.
        Running-angle-sum version -- no matrices needed for a planar chain.
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

    def _pose_and_jacobian(self, q):
        """
        Shared helper: given joint angles q (radians), return the current
        end-effector pose [x, y, phi] and the 3x3 Jacobian at that pose.
        Used by both gradient descent solvers below so the math lives
        in exactly one place.
        """
        L1, L2, L3 = self.l1, self.l2, self.l3
        t1, t2, t3 = q
        t12 = t1 + t2
        t123 = t1 + t2 + t3

        x = L1 * np.cos(t1) + L2 * np.cos(t12) + L3 * np.cos(t123)
        y = L1 * np.sin(t1) + L2 * np.sin(t12) + L3 * np.sin(t123)
        phi = t123
        pose = np.array([x, y, phi])

        J = np.array([
            [-L1*np.sin(t1) - L2*np.sin(t12) - L3*np.sin(t123), -L2*np.sin(t12) - L3*np.sin(t123), -L3*np.sin(t123)],
            [ L1*np.cos(t1) + L2*np.cos(t12) + L3*np.cos(t123),  L2*np.cos(t12) + L3*np.cos(t123),  L3*np.cos(t123)],
            [1, 1, 1],
        ])
        return pose, J

    def compute_ik_gradient_descent(self, target_x, target_y, target_phi_deg,
                                      initial_guess=(0, 0, 0),
                                      alpha=0.01, tolerance=0.01, max_iterations=5000):
        """
        Jacobian Transpose IK Solver (gradient descent).

        Iteratively nudges the joint angles in the steepest-descent
        direction (J^T @ error) until the end effector is close enough
        to the target, or max_iterations is hit.

        Returns (angles_deg, converged: bool, iterations: int)
        """
        q = np.radians(initial_guess)
        target = np.array([target_x, target_y, np.radians(target_phi_deg)])

        for i in range(max_iterations):
            pose, J = self._pose_and_jacobian(q)
            error = pose - target

            if np.linalg.norm(error) < tolerance:
                return np.degrees(q), True, i

            q = q - alpha * (J.T @ error)

        return np.degrees(q), False, max_iterations

    def compute_ik_pseudoinverse(self, target_x, target_y, target_phi_deg,
                                   initial_guess=(0, 0, 0),
                                   alpha=0.5, tolerance=0.01, max_iterations=500):
        """
        Jacobian Pseudoinverse IK Solver.

        Same idea as gradient descent, but uses the Moore-Penrose
        pseudoinverse instead of the transpose -- converges in far
        fewer iterations and tolerates a much larger step size.

        Returns (angles_deg, converged: bool, iterations: int)
        """
        q = np.radians(initial_guess)
        target = np.array([target_x, target_y, np.radians(target_phi_deg)])

        for i in range(max_iterations):
            pose, J = self._pose_and_jacobian(q)
            error = pose - target

            if np.linalg.norm(error) < tolerance:
                return np.degrees(q), True, i

            q = q - alpha * (np.linalg.pinv(J) @ error)

        return np.degrees(q), False, max_iterations


if __name__ == "__main__":
    print("--- Bob Ross Robot Arm: Gradient Descent IK ---")

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
    target_angle = get_float("Enter Desired Brush Angle in degrees (e.g., -45): ", -45.0)

    angles, converged, iters = arm.compute_ik_gradient_descent(
        target_x, target_y, target_angle, initial_guess=(10, 10, 10)
    )

    if converged:
        print(f"\nConverged in {iters} iterations.")
    else:
        print(f"\nDid NOT converge after {iters} iterations -- result may be inaccurate.")
        print("Try the pseudoinverse solver instead, or a different initial_guess/alpha.")

    print("Calculated Motor Angles:")
    print(f"Theta 1 (Base):  {angles[0]:.2f}°")
    print(f"Theta 2 (Elbow): {angles[1]:.2f}°")
    print(f"Theta 3 (Wrist): {angles[2]:.2f}°")

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
             'k--', lw=2, label='Target Angle')

    reach = l1 + l2 + l3 + 5
    ax.set(title='Gradient Descent (Jacobian Transpose) IK', xlabel='X (cm)', ylabel='Y (cm)',
           aspect='equal', xlim=(-reach, reach), ylim=(-reach, reach))
    ax.grid(True, ls='--')
    ax.legend()
    plt.show()