import json
import subprocess
import sys
from model import point_x, predict_class

ROUNDS = 20


def acc(path, model):
    with open(path) as f:
        data = json.load(f)
    if not data:
        return 0.0
    ok = sum(1 for pt in data if predict_class(model, point_x(pt)) == int(pt["y"]))
    return 100.0 * ok / len(data)


subprocess.run([sys.executable, "reset.py"], check=True)

print(f"{'round':<8}{'train%':<10}{'test%'}")
for r in range(1, ROUNDS + 1):
    for i in (1, 2, 3):
        subprocess.run([sys.executable, "client_update.py", str(i)], check=True)
    subprocess.run([sys.executable, "aggregate.py"], check=True)
    subprocess.run([sys.executable, "update_model.py"], check=True, stdout=subprocess.DEVNULL)
    with open("global_model.json") as f:
        model = json.load(f)
    train_a = (acc("client_data_1.json", model) + acc("client_data_2.json", model) + acc("client_data_3.json", model)) / 3
    test_a = (acc("client_test_1.json", model) + acc("client_test_2.json", model)) / 2
    print(f"{r:<8}{train_a:<10.1f}{test_a:.1f}")
