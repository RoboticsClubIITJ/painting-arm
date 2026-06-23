import numpy as np

# Function to generate transformation matrix
def dh_transform(theta, a):
    theta = np.radians(theta)

    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta), 0, a*np.sin(theta)],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])

    return T

# User Inputs
L1 = float(input("Enter Link 1 Length: "))
L2 = float(input("Enter Link 2 Length: "))
L3 = float(input("Enter Link 3 Length: "))

theta1 = float(input("Enter θ1 (degrees): "))
theta2 = float(input("Enter θ2 (degrees): "))
theta3 = float(input("Enter θ3 (degrees): "))

# Individual Transformation Matrices
T01 = dh_transform(theta1, L1)
T12 = dh_transform(theta2, L2)
T23 = dh_transform(theta3, L3)

# Final Transformation Matrix
T03 = T01 @ T12 @ T23

# End Effector Position
x = T03[0, 3]
y = T03[1, 3]

# Orientation
phi = theta1 + theta2 + theta3

# Results
print("\nT01 =\n", np.round(T01, 4))
print("\nT12 =\n", np.round(T12, 4))
print("\nT23 =\n", np.round(T23, 4))

print("\nFinal Transformation Matrix T03 =\n")
print(np.round(T03, 4))

print("\nEnd Effector Position:")
print(f"x = {x:.4f}")
print(f"y = {y:.4f}")

print(f"\nEnd Effector Orientation = {phi:.2f} degrees")
