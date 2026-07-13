import numpy as np
import math


# ==========================================================
# Inverse Kinematics using Gradient Descent
# 3-Link Planar Manipulator
# ==========================================================

def forward_kinematics(L1, L2, L3, theta1, theta2, theta3):
    """
    Computes the end-effector position using Forward Kinematics.
    """

    x = (L1 * np.cos(theta1) +
         L2 * np.cos(theta1 + theta2) +
         L3 * np.cos(theta1 + theta2 + theta3))

    y = (L1 * np.sin(theta1) +
         L2 * np.sin(theta1 + theta2) +
         L3 * np.sin(theta1 + theta2 + theta3))

    return x, y


def compute_jacobian(L1, L2, L3, theta1, theta2, theta3):
    """
    Computes the Jacobian matrix of the manipulator.
    """

    J = np.array([

        [-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2) - L3 * np.sin(theta1 + theta2 + theta3),
         -L2 * np.sin(theta1 + theta2) - L3 * np.sin(theta1 + theta2 + theta3),
         -L3 * np.sin(theta1 + theta2 + theta3)],

        [L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3),
         L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3),
         L3 * np.cos(theta1 + theta2 + theta3)]

    ])

    return J


def inverse_kinematics_gradient_descent(
        L1, L2, L3,
        target_x, target_y,
        theta1, theta2, theta3,
        learning_rate=0.05,
        tolerance=1e-3,
        max_iterations=1000):
    """
    Solves inverse kinematics using Gradient Descent.
    """

    for iteration in range(max_iterations):

        # Current End-Effector Position
        x, y = forward_kinematics(
            L1, L2, L3,
            theta1, theta2, theta3
        )

        # Position Error
        error = np.array([
            [target_x - x],
            [target_y - y]
        ])

        # Check convergence
        if np.linalg.norm(error) < tolerance:
            print(f"\nSolution converged in {iteration} iterations.")
            break

        # Compute Jacobian
        J = compute_jacobian(
            L1, L2, L3,
            theta1, theta2, theta3
        )

        # Gradient Descent Update
        delta_theta = learning_rate * (J.T @ error)

        # Update Joint Angles
        theta1 += delta_theta[0, 0]
        theta2 += delta_theta[1, 0]
        theta3 += delta_theta[2, 0]

    else:
        print("\nMaximum iterations reached before convergence.")

    return theta1, theta2, theta3


# ==========================================================
# Main Program
# ==========================================================

print("\n3-Link Planar Manipulator")
print("Inverse Kinematics using Gradient Descent\n")

L1 = float(input("Enter Link 1 length : "))
L2 = float(input("Enter Link 2 length : "))
L3 = float(input("Enter Link 3 length : "))

target_x = float(input("\nEnter desired x-coordinate : "))
target_y = float(input("Enter desired y-coordinate : "))

theta1 = math.radians(float(input("\nInitial Theta1 (degrees) : ")))
theta2 = math.radians(float(input("Initial Theta2 (degrees) : ")))
theta3 = math.radians(float(input("Initial Theta3 (degrees) : ")))

theta1, theta2, theta3 = inverse_kinematics_gradient_descent(
    L1, L2, L3,
    target_x, target_y,
    theta1, theta2, theta3
)

print("\n==========================================")
print("Final Joint Angles")
print("==========================================")
print(f"Theta 1 : {math.degrees(theta1):8.2f}°")
print(f"Theta 2 : {math.degrees(theta2):8.2f}°")
print(f"Theta 3 : {math.degrees(theta3):8.2f}°")
