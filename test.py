import json
from model import USE_RELU, predict

with open("global_model.json") as f:
    model = json.load(f)
w, b = model["w"], model["b"]

act = "ReLU(w*x + b)" if USE_RELU else "w*x + b"
print(f"model:  y_hat = {act}  (w={w:.4f}, b={b:.4f})")
print(f"true:   y     = 2 * x + 1")
print()

all_abs = []
all_sq = []

for i in (1, 2):
    with open(f"client_test_{i}.json") as f:
        data = json.load(f)

    abs_err = []
    sq_err = []
    print(f"client_test_{i}")
    print(f"{'x':>8} {'y':>8} {'y_hat':>8} {'error':>8}")
    for pt in data:
        x, y = pt["x"], pt["y"]
        y_hat = predict(w, b, x)
        err = y_hat - y
        abs_err.append(abs(err))
        sq_err.append(err * err)
        print(f" {x:8.4f} {y:8.4f} {y_hat:8.4f} {err:8.4f}")

    mae = sum(abs_err) / len(abs_err)
    mse = sum(sq_err) / len(sq_err)
    all_abs.extend(abs_err)
    all_sq.extend(sq_err)
    print(f"  MAE {mae:.4f}   MSE {mse:.4f}\n")

mae = sum(all_abs) / len(all_abs)
mse = sum(all_sq) / len(all_sq)
print(f"overall  MAE {mae:.4f}   MSE {mse:.4f}")
