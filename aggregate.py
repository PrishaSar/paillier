import json

with open("public_key.json") as f:
    n = json.load(f)["n"]
n_sq = n * n

with open("update_pool.json") as f:
    pool = json.load(f)

sum_dw = 1
sum_db = 1
for upd in pool:
    sum_dw = (sum_dw * upd["dw"]) % n_sq
    sum_db = (sum_db * upd["db"]) % n_sq

with open("encrypted_aggregate.json", "w") as f:
    json.dump({"dw": sum_dw, "db": sum_db}, f)
