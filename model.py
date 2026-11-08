import math
import random

IN = 2
H = 8
OUT = 2


def relu(x):
    return max(0.0, x)


def softmax(z):
    m = max(z)
    exps = [math.exp(zi - m) for zi in z]
    s = sum(exps)
    return [e / s for e in exps]


def zeros_mat(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def zeros_vec(n):
    return [0.0] * n


def init_model(seed=0):
    rng = random.Random(seed)
    scale1 = math.sqrt(2.0 / IN)
    scale2 = math.sqrt(2.0 / H)
    return {
        "W1": [[rng.gauss(0, scale1) for _ in range(IN)] for _ in range(H)],
        "b1": zeros_vec(H),
        "W2": [[rng.gauss(0, scale2) for _ in range(H)] for _ in range(OUT)],
        "b2": zeros_vec(OUT),
    }


def forward(model, x):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    z1 = [sum(W1[i][j] * x[j] for j in range(IN)) + b1[i] for i in range(H)]
    h = [relu(z) for z in z1]
    z2 = [sum(W2[k][i] * h[i] for i in range(H)) + b2[k] for k in range(OUT)]
    p = softmax(z2)
    return z1, h, z2, p


def predict_class(model, x):
    _, _, _, p = forward(model, x)
    return 0 if p[0] >= p[1] else 1


def point_x(pt):
    return [pt["x0"], pt["x1"]]


def gradients(model, data):
    dW1 = zeros_mat(H, IN)
    db1 = zeros_vec(H)
    dW2 = zeros_mat(OUT, H)
    db2 = zeros_vec(OUT)
    n = len(data)
    for pt in data:
        x = point_x(pt)
        y = int(pt["y"])
        z1, h, z2, p = forward(model, x)
        dz2 = [p[k] - (1.0 if k == y else 0.0) for k in range(OUT)]
        for k in range(OUT):
            db2[k] += dz2[k]
            for i in range(H):
                dW2[k][i] += dz2[k] * h[i]
        dh = [sum(model["W2"][k][i] * dz2[k] for k in range(OUT)) for i in range(H)]
        dz1 = [dh[i] if z1[i] > 0 else 0.0 for i in range(H)]
        for i in range(H):
            db1[i] += dz1[i]
            for j in range(IN):
                dW1[i][j] += dz1[i] * x[j]
    inv = 1.0 / n
    return {
        "W1": [[v * inv for v in row] for row in dW1],
        "b1": [v * inv for v in db1],
        "W2": [[v * inv for v in row] for row in dW2],
        "b2": [v * inv for v in db2],
    }
