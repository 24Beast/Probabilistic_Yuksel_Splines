import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
NUM_POINTS = 11
NUM_DIMS = 2
B = np.array([[1, 0, 0], [-2, 2, 0], [1, -2, 1]]) / 1.0
T_EXP = np.expand_dims(np.array([i for i in range(3)]), 1)


# Helper Functions
def transformQuadSpline(P, t_val):
    t_val = t_val.repeat(3, axis=1)
    t_exp = T_EXP.repeat(t_val.shape[0], 1).T
    t = np.power(t_val, t_exp)
    f = np.matmul(t, B)
    f = np.matmul(f, P)
    return f


def plot(points, title):
    x, y = points[:, 0], points[:, 1]
    plt.plot(x, y)
    plt.scatter(x, y, c="red")
    plt.title(title)
    plt.show()


# Try to see seperate visuals
def predict(P, t_val):
    num_splines = P.shape[0] - 2
    probs = np.zeros((len(t_val), P.shape[1]))
    P_new = P[:3].copy()
    probs_1 = probs * 0
    for i in range(num_splines - 1):
        d = ((t_val * (num_splines - 1)) - i) * 0.5
        """
        if (i == 0):
            choice = (0 <= d) * (d < 0.5)
            f_curr = transformQuadSpline(P_new, d)
            probs += choice * (np.sin(np.pi * d) ** 2) * f_curr
            probs_1 += choice * (np.sin(np.pi * d) ** 2) * f_curr
        elif (i==num_splines):
            choice = (0 <= d) * (d <= 0.5)
            P_temp = P_new
            f_prev = transformQuadSpline(P_temp, d+0.5)
            probs += choice * (np.sin(np.pi * d) ** 2) * f_prev
            probs_1 += choice * (np.sin(np.pi * d) ** 2) * f_prev
        else:
        """
        choice = (0 <= d) * (d <= 0.5)
        probs_1 = probs_1 * (1 - choice)
        P_temp = P_new.copy()
        P_new[0] = (P_temp[0] + P_temp[2]) / 4 + P_temp[1] / 2
        P_new[2] = P[i + 3]
        P_new[1] = 2 * (P_temp[2] - (P_new[0] + P_new[2]) / 4)
        f_prev = transformQuadSpline(P_temp, d + 0.5)
        f_curr = transformQuadSpline(P_new, d)
        probs += choice * (
            (((np.cos(np.pi * d) ** 2) * f_prev) + ((np.sin(np.pi * d) ** 2) * f_curr))
        )
        probs_1 += choice * (
            ((np.cos(np.pi * d) ** 2) * f_prev) + ((np.sin(np.pi * d) ** 2) * f_curr)
        )
        print(f"{choice.shape=}, {probs_1.shape=}")
        plt.plot(
            probs_1[choice[:, 0] > 0][:, 0],
            probs_1[choice[:, 0] > 0][:, 1],
            label=f"Curve {i}",
        )
        print(f"{P_new=}")
    plt.legend()
    plt.show()
    return probs_1


p = np.random.random((NUM_POINTS, NUM_DIMS)) * 10
p = np.cumsum(p, axis=0)
p = (p - p.mean()) / p.std()
print(f"{p=}")

plt.scatter(p[:, 0], p[:, 1])
plt.title("Original Points")
plt.show()

filt = np.zeros((NUM_POINTS, 1))
filt[1::2] = 1
zero = np.zeros((1, NUM_DIMS))

p_new = (p * (1 - filt)) + (p / 2 * filt)
p_new = p_new + filt * (np.concatenate((zero, p[:-1]))) / 4
p_new = p_new + filt * (np.concatenate((p[1:], zero))) / 4

t_val = np.array([i / 100 for i in range(101)])
t_val = np.expand_dims(t_val, 1)

test_vals = transformQuadSpline(p[:3], t_val)
plot(test_vals, "Bezier Sanity Check")

final_vals = predict(p, t_val)
plot(final_vals, "C2 Spline Check")
