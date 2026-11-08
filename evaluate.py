import json
import os
import subprocess
import sys

from model import point_x, predict_class


def load_test():
    pts = []
    for path in ("client_test_1.json", "client_test_2.json"):
        with open(path) as f:
            pts.extend(json.load(f))
    return pts


def accuracy(preds, ys):
    n = len(ys)
    ok = sum(1 for p, y in zip(preds, ys) if int(p) == int(y))
    return 100.0 * ok / n


if not os.path.exists("export/weights_int8.bin"):
    subprocess.run([sys.executable, "quantize.py"], check=True)

subprocess.run(["make", "infer"], check=True)
out = subprocess.check_output(["./infer"], text=True)
int8_preds = [int(line) for line in out.splitlines() if line.strip()]

with open("export/float32.json") as f:
    model = json.load(f)

pts = load_test()
ys = [int(pt["y"]) for pt in pts]
f32_preds = [predict_class(model, point_x(pt)) for pt in pts]

if len(int8_preds) != len(ys):
    raise SystemExit(f"C++ returned {len(int8_preds)} preds, expected {len(ys)}")

acc_f32 = accuracy(f32_preds, ys)
acc_i8 = accuracy(int8_preds, ys)
drop = acc_f32 - acc_i8

size_f32 = os.path.getsize("export/weights_float32.bin")
size_i8 = os.path.getsize("export/weights_int8.bin")

print("predictions (first 8)")
print(f"{'x0':>8} {'x1':>8} {'y':>4} {'f32':>4} {'int8':>4}")
for pt, p32, p8 in list(zip(pts, f32_preds, int8_preds))[:8]:
    print(f"{pt['x0']:8.4f} {pt['x1']:8.4f} {pt['y']:4d} {p32:4d} {p8:4d}")

print()
print(f"float32 model size: {size_f32} bytes")
print(f"int8 model size:    {size_i8} bytes")
print(f"compression:        {size_f32 / size_i8:.2f}x")
print()
print(f"float32 accuracy: {acc_f32:.2f}%")
print(f"int8 accuracy:    {acc_i8:.2f}%")
print(f"accuracy drop:    {drop:.2f}%")
print("(class match on held-out two-moons test points)")
