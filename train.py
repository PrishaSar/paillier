import json
import subprocess
import sys

ROUNDS = 20

subprocess.run([sys.executable, "reset.py"], check=True)

print(f"{'round':<8}{'w':<12}{'b'}")
print(f"{0:<8}{0.0:<12.4f}{0.0:.4f}")

for r in range(1, ROUNDS + 1):
    for i in (1, 2, 3):
        subprocess.run([sys.executable, "client_update.py", str(i)], check=True)
    subprocess.run([sys.executable, "aggregate.py"], check=True)
    subprocess.run([sys.executable, "update_model.py"], check=True, stdout=subprocess.DEVNULL)
    with open("global_model.json") as f:
        m = json.load(f)
    print(f"{r:<8}{m['w']:<12.4f}{m['b']:.4f}")
