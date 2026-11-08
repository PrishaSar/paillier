import json
from model import point_x, predict_class

with open("global_model.json") as f:
    model = json.load(f)

print("float32 MLP  2-8-2  ReLU + softmax")
print()

all_ok = 0
all_n = 0
for i in (1, 2):
    with open(f"client_test_{i}.json") as f:
        data = json.load(f)
    ok = 0
    print(f"client_test_{i}")
    print(f"{'x0':>8} {'x1':>8} {'y':>4} {'pred':>4}")
    for pt in data:
        pred = predict_class(model, point_x(pt))
        y = int(pt["y"])
        ok += pred == y
        print(f"{pt['x0']:8.4f} {pt['x1']:8.4f} {y:4d} {pred:4d}")
    acc = 100.0 * ok / len(data)
    all_ok += ok
    all_n += len(data)
    print(f"  accuracy {acc:.1f}%\n")

print(f"overall  {100.0 * all_ok / all_n:.1f}%")
