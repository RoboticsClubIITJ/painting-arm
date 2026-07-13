import numpy as np
import math

# ------------------------
# Link Lengths
# ------------------------
L1 = float(input("Enter Link 1 length: "))
L2 = float(input("Enter Link 2 length: "))
L3 = float(input("Enter Link 3 length: "))

# Desired Position
xd = float(input("Enter desired x: "))
yd = float(input("Enter desired y: "))

# Initial Guess (degrees)
t1 = math.radians(float(input("Initial theta1: ")))
t2 = math.radians(float(input("Initial theta2: ")))
t3 = math.radians(float(input("Initial theta3: ")))

# Iterative Solution
for i in range(100):

    # Forward Kinematics
    x = (L1*np.cos(t1) +
         L2*np.cos(t1+t2) +
         L3*np.cos(t1+t2+t3))

    y = (L1*np.sin(t1) +
         L2*np.sin(t1+t2) +
         L3*np.sin(t1+t2+t3))

    # Error
    e = np.array([[xd-x],
                  [yd-y]])

    # Stop if error is very small
    if np.linalg.norm(e) < 0.001:
        break

    # Jacobian Matrix
    J = np.array([

        [-L1*np.sin(t1)-L2*np.sin(t1+t2)-L3*np.sin(t1+t2+t3),
         -L2*np.sin(t1+t2)-L3*np.sin(t1+t2+t3),
         -L3*np.sin(t1+t2+t3)],

        [L1*np.cos(t1)+L2*np.cos(t1+t2)+L3*np.cos(t1+t2+t3),
         L2*np.cos(t1+t2)+L3*np.cos(t1+t2+t3),
         L3*np.cos(t1+t2+t3)]

    ])

    # Pseudo Inverse
    J_inv = np.linalg.pinv(J)

    # Joint Angle Update
    dtheta = J_inv @ e

    # Update Angles
    t1 += dtheta[0,0]
    t2 += dtheta[1,0]
    t3 += dtheta[2,0]

# Output
print("\nInverse Kinematics Solution")
print("Theta1 =", round(math.degrees(t1),2), "degrees")
print("Theta2 =", round(math.degrees(t2),2), "degrees")
print("Theta3 =", round(math.degrees(t3),2), "degrees")
