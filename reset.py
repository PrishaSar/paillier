import json
from model import init_model

with open("global_model.json", "w") as f:
    json.dump(init_model(), f)
with open("update_pool.json", "w") as f:
    json.dump([], f)

print("reset MLP weights")
