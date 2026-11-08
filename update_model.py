import json
from paillier import decrypt, decode
from model import IN, H, OUT

N_CLIENTS = 3
LR = 0.25

with open("private_key.json") as f:
    priv = json.load(f)
lam, mu, n = priv["lambda"], priv["mu"], priv["n"]
pk = (lam, mu, n)
pub = (n, n + 1)

with open("encrypted_aggregate.json") as f:
    agg = json.load(f)


def dec_mat(M):
    return [[decode(decrypt(pk, pub, c), n) / N_CLIENTS for c in row] for row in M]


def dec_vec(v):
    return [decode(decrypt(pk, pub, c), n) / N_CLIENTS for c in v]


avg = {
    "W1": dec_mat(agg["W1"]),
    "b1": dec_vec(agg["b1"]),
    "W2": dec_mat(agg["W2"]),
    "b2": dec_vec(agg["b2"]),
}

with open("global_model.json") as f:
    model = json.load(f)

for i in range(H):
    model["b1"][i] -= LR * avg["b1"][i]
    for j in range(IN):
        model["W1"][i][j] -= LR * avg["W1"][i][j]
for k in range(OUT):
    model["b2"][k] -= LR * avg["b2"][k]
    for i in range(H):
        model["W2"][k][i] -= LR * avg["W2"][k][i]

with open("global_model.json", "w") as f:
    json.dump(model, f)

with open("update_pool.json", "w") as f:
    json.dump([], f)

print("updated")
