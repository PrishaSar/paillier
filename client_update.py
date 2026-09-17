import json
import sys
from paillier import encrypt, encode
from model import predict, activation_grad

client_id = sys.argv[1]

with open("public_key.json") as f:
    pub = json.load(f)
n, g = pub["n"], pub["g"]

with open("global_model.json") as f:
    model = json.load(f)
w, b = model["w"], model["b"]

with open(f"client_data_{client_id}.json") as f:
    data = json.load(f)

N = len(data)
dw = 0.0
db = 0.0
for pt in data:
    x, y = pt["x"], pt["y"]
    z = w * x + b
    err = predict(w, b, x) - y
    dz = err * activation_grad(z)
    dw += dz * x
    db += dz
dw = 2 * dw / N
db = 2 * db / N

enc = {
    "dw": encrypt((n, g), encode(dw, n)),
    "db": encrypt((n, g), encode(db, n)),
}

try:
    with open("update_pool.json") as f:
        pool = json.load(f)
except FileNotFoundError:
    pool = []

pool.append(enc)

with open("update_pool.json", "w") as f:
    json.dump(pool, f)
