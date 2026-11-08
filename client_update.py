import json
import sys
from paillier import encrypt, encode
from model import gradients

client_id = sys.argv[1]

with open("public_key.json") as f:
    pub = json.load(f)
n, g = pub["n"], pub["g"]
key = (n, g)

with open("global_model.json") as f:
    model = json.load(f)

with open(f"client_data_{client_id}.json") as f:
    data = json.load(f)

grads = gradients(model, data)


def enc_mat(M):
    return [[encrypt(key, encode(v, n)) for v in row] for row in M]


def enc_vec(v):
    return [encrypt(key, encode(x, n)) for x in v]


enc = {
    "W1": enc_mat(grads["W1"]),
    "b1": enc_vec(grads["b1"]),
    "W2": enc_mat(grads["W2"]),
    "b2": enc_vec(grads["b2"]),
}

try:
    with open("update_pool.json") as f:
        pool = json.load(f)
except FileNotFoundError:
    pool = []

pool.append(enc)

with open("update_pool.json", "w") as f:
    json.dump(pool, f)
