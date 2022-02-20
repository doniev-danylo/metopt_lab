import numpy as np


def F (x, y):
    return 12 * x ** 2 + 14 * y ** 2 + 0.01 * x * y + 3 * x


def Grad (x, y):
    return np.array([24 * x + 0.01 * y + 3,  28 * y + 0.01 * x])


# сам метод

def Gr_m (x1, x2):
    alpha = 0.1  # Шаг сходимости
    eps = 0.000001  # точность
    X_prev = np.array([x1, x2])
    X = X_prev - alpha * Grad(X_prev[0], X_prev[1])

    while np.linalg.norm(X - X_prev) > eps :
        X_prev = X.copy()
        X = X_prev - alpha * Grad(X_prev[0], X_prev[1])
        print(X_prev, X)

    return X


result = Gr_m(0.3, 0.3)

print(result)
