import json
import os
import struct
from glob import glob

from model import IN, H, OUT, forward, point_x

QMAX = 127
ACT_PERCENTILE = 80  # activation PTQ on one client; outliers clip
EXPORT = "export"


def qmax_scale(values, percentile=100):
    vs = sorted(abs(float(v)) for v in values) or [0.0]
    idx = int(round((percentile / 100.0) * (len(vs) - 1)))
    peak = vs[idx]
    return max(peak, 1e-12) / QMAX


def quantize(values, scale):
    out = []
    for v in values:
        q = int(round(float(v) / scale))
        q = max(-QMAX, min(QMAX, q))
        out.append(q)
    return out


def flatten(model):
    vals = []
    for row in model["W1"]:
        vals.extend(row)
    vals.extend(model["b1"])
    for row in model["W2"]:
        vals.extend(row)
    vals.extend(model["b2"])
    return vals


def load_pts(paths):
    pts = []
    for path in paths:
        with open(path) as f:
            pts.extend(json.load(f))
    return pts


with open("global_model.json") as f:
    model = json.load(f)

train_pts = load_pts(sorted(glob("client_data_*.json")))
test_pts = load_pts(["client_test_1.json", "client_test_2.json"])
# Activation ranges are calibrated on client 1 only — the server never sees
# a central unlabeled set, which is the federated constraint.
calib_pts = load_pts(["client_data_1.json"])

W1 = [v for row in model["W1"] for v in row]
b1 = list(model["b1"])
W2 = [v for row in model["W2"] for v in row]
b2 = list(model["b2"])

xs = [v for pt in calib_pts for v in point_x(pt)]
hs = []
for pt in calib_pts:
    _, h, _, _ = forward(model, point_x(pt))
    hs.extend(h)

scales = {
    "W1": qmax_scale(W1),
    "b1": qmax_scale(b1),
    "W2": qmax_scale(W2),
    "b2": qmax_scale(b2),
    "x": qmax_scale(xs, ACT_PERCENTILE),
    "h": qmax_scale(hs, ACT_PERCENTILE),
}

qW1 = quantize(W1, scales["W1"])
qb1 = quantize(b1, scales["b1"])
qW2 = quantize(W2, scales["W2"])
qb2 = quantize(b2, scales["b2"])
q_all = qW1 + qb1 + qW2 + qb2
f_all = W1 + b1 + W2 + b2

os.makedirs(EXPORT, exist_ok=True)

with open(os.path.join(EXPORT, "weights_float32.bin"), "wb") as f:
    f.write(struct.pack("<" + "f" * len(f_all), *f_all))

with open(os.path.join(EXPORT, "weights_int8.bin"), "wb") as f:
    f.write(struct.pack("<" + "b" * len(q_all), *q_all))

with open(os.path.join(EXPORT, "scales.txt"), "w") as f:
    f.write(f"IN {IN}\nH {H}\nOUT {OUT}\n")
    for k, v in scales.items():
        f.write(f"{k}_scale {v}\n")

with open(os.path.join(EXPORT, "float32.json"), "w") as f:
    json.dump(model, f)

with open(os.path.join(EXPORT, "int8.json"), "w") as f:
    json.dump({"qW1": qW1, "qb1": qb1, "qW2": qW2, "qb2": qb2, "scales": scales}, f)

with open(os.path.join(EXPORT, "test_xy.txt"), "w") as f:
    for pt in test_pts:
        f.write(f"{pt['x0']} {pt['x1']} {pt['y']}\n")

n = len(f_all)
print("exported to export/")
print(f"params {n}  float32 {n * 4} bytes  int8 {n} bytes  ({4.0:.2f}x)")
print("scales", {k: round(v, 6) for k, v in scales.items()})
print(f"activation scales: client 1, {ACT_PERCENTILE}th percentile")
