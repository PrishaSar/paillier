import json

with open("global_model.json", "w") as f:
    json.dump({"w": 0.0, "b": 0.0}, f)
with open("update_pool.json", "w") as f:
    json.dump([], f)

print("reset w=0, b=0")
