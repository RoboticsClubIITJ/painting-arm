import numpy as np

def ik_3link_all(xd, yd, phid, l1, l2, l3):
    # Wrist position
    xw = xd - l3*np.cos(phid)
    yw = yd - l3*np.sin(phid)

    r2 = xw**2 + yw**2
    cos_t2 = (r2 - l1**2 - l2**2) / (2*l1*l2)
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)

    # Two possible theta2 values
    t2_options = [np.arccos(cos_t2), -np.arccos(cos_t2)]
    solutions = []

    for t2 in t2_options:
        t1 = np.arctan2(yw, xw) - np.arctan2(l2*np.sin(t2), l1 + l2*np.cos(t2))
        t3 = phid - (t1 + t2)
        solutions.append((t1, t2, t3))

    return solutions

# Example
l1, l2, l3 = 1.0, 1.0, 1.0
xd, yd, phid = 1.5, 1.0, np.radians(90)

sols = ik_3link_all(xd, yd, phid, l1, l2, l3)
for i, (t1, t2, t3) in enumerate(sols, 1):
    print(f"Solution {i}: θ1={t1:.3f}, θ2={t2:.3f}, θ3={t3:.3f}")
