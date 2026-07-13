import math


# ==========================================================
# Inverse Kinematics of a 3-Link Planar Manipulator
# ==========================================================

def inverse_kinematics(L1, L2, L3, x, y, phi_deg):
    """
    Solves the inverse kinematics of a 3-link planar manipulator.

    Parameters:
        L1, L2, L3 : Link lengths
        x, y       : Desired end-effector position
        phi_deg    : Desired end-effector orientation (degrees)

    Returns:
        Two possible joint configurations (Elbow-Up and Elbow-Down)
    """

    # Convert orientation to radians
    phi = math.radians(phi_deg)

    # ------------------------------------------------------
    # Step 1 : Compute Wrist Position
    # ------------------------------------------------------
    wrist_x = x - L3 * math.cos(phi)
    wrist_y = y - L3 * math.sin(phi)

    # ------------------------------------------------------
    # Step 2 : Compute cos(theta2)
    # ------------------------------------------------------
    cos_theta2 = (
        wrist_x**2 + wrist_y**2 - L1**2 - L2**2
    ) / (2 * L1 * L2)

    # Check reachability
    if abs(cos_theta2) > 1:
        print("\nTarget point is outside the robot workspace.")
        return

    print("\n==========================================")
    print("Inverse Kinematics Solution")
    print("==========================================")

    # ======================================================
    # Compute both configurations
    # ======================================================
    for configuration, theta2 in [

        ("Elbow-Down", math.acos(cos_theta2)),
        ("Elbow-Up", -math.acos(cos_theta2))

    ]:

        theta1 = (
            math.atan2(wrist_y, wrist_x)
            - math.atan2(
                L2 * math.sin(theta2),
                L1 + L2 * math.cos(theta2)
            )
        )

        theta3 = phi - theta1 - theta2

        print(f"\n{configuration} Configuration")
        print("-" * 30)

        print(f"Theta 1 : {math.degrees(theta1):8.2f}°")
        print(f"Theta 2 : {math.degrees(theta2):8.2f}°")
        print(f"Theta 3 : {math.degrees(theta3):8.2f}°")


# ==========================================================
# Main Program
# ==========================================================

print("\n3-Link Planar Manipulator Inverse Kinematics\n")

L1 = float(input("Enter Link 1 length : "))
L2 = float(input("Enter Link 2 length : "))
L3 = float(input("Enter Link 3 length : "))

x = float(input("\nEnter desired x-coordinate : "))
y = float(input("Enter desired y-coordinate : "))
phi = float(input("Enter end-effector orientation (degrees) : "))

inverse_kinematics(L1, L2, L3, x, y, phi)
