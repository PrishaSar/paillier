import json
import math
import random


def moons(n_per_class, noise, rng):
    pts = []
    for i in range(n_per_class):
        t = math.pi * i / max(n_per_class - 1, 1)
        pts.append({
            "x0": round(math.cos(t) + rng.gauss(0, noise), 4),
            "x1": round(math.sin(t) + rng.gauss(0, noise), 4),
            "y": 0,
        })
        pts.append({
            "x0": round(1 - math.cos(t) + rng.gauss(0, noise), 4),
            "x1": round(0.5 - math.sin(t) + rng.gauss(0, noise), 4),
            "y": 1,
        })
    rng.shuffle(pts)
    return pts


def split_non_iid(pts):
    pts = sorted(pts, key=lambda p: p["x0"])
    n = len(pts)
    return pts[: n // 3], pts[n // 3 : 2 * n // 3], pts[2 * n // 3 :]


rng = random.Random(7)
train = moons(50, noise=0.26, rng=rng)
c1, c2, c3 = split_non_iid(train)

# noisier held-out set — more points sit on the overlapping moons
test_all = moons(100, noise=0.38, rng=random.Random(21))
t1, t2 = test_all[:100], test_all[100:]

for path, data in [
    ("client_data_1.json", c1),
    ("client_data_2.json", c2),
    ("client_data_3.json", c3),
    ("client_test_1.json", t1),
    ("client_test_2.json", t2),
]:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(path, len(data), "class0", sum(p["y"] == 0 for p in data), "class1", sum(p["y"] == 1 for p in data))
