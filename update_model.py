import json
from paillier import decrypt, decode

N_CLIENTS = 3
LR = 0.1

with open("private_key.json") as f:
    priv = json.load(f)
lam, mu, n = priv["lambda"], priv["mu"], priv["n"]

with open("encrypted_aggregate.json") as f:
    agg = json.load(f)

sum_dw = decode(decrypt((lam, mu, n), (n, n + 1), agg["dw"]), n)
sum_db = decode(decrypt((lam, mu, n), (n, n + 1), agg["db"]), n)
avg_dw = sum_dw / N_CLIENTS
avg_db = sum_db / N_CLIENTS

with open("global_model.json") as f:
    model = json.load(f)

model["w"] -= LR * avg_dw
model["b"] -= LR * avg_db

with open("global_model.json", "w") as f:
    json.dump(model, f)

with open("update_pool.json", "w") as f:
    json.dump([], f)

print(model)
