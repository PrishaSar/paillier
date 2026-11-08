import json
from model import IN, H, OUT

with open("public_key.json") as f:
    n = json.load(f)["n"]
n_sq = n * n

with open("update_pool.json") as f:
    pool = json.load(f)


def hom_mat(rows, cols, key):
    acc = [[1] * cols for _ in range(rows)]
    for upd in pool:
        M = upd[key]
        for i in range(rows):
            for j in range(cols):
                acc[i][j] = (acc[i][j] * M[i][j]) % n_sq
    return acc


def hom_vec(length, key):
    acc = [1] * length
    for upd in pool:
        v = upd[key]
        for i in range(length):
            acc[i] = (acc[i] * v[i]) % n_sq
    return acc


agg = {
    "W1": hom_mat(H, IN, "W1"),
    "b1": hom_vec(H, "b1"),
    "W2": hom_mat(OUT, H, "W2"),
    "b2": hom_vec(OUT, "b2"),
}

with open("encrypted_aggregate.json", "w") as f:
    json.dump(agg, f)
