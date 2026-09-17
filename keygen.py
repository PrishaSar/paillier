import json
from paillier import generate_keypair

def runonce():
    (n, g), (lam, mu, n) = generate_keypair()
    with open("public_key.json", "w") as f:
        json.dump({"n": int(n), "g": int(g)}, f)
    with open("private_key.json", "w") as f:
        json.dump({"lambda": int(lam), "mu": int(mu), "n": int(n)}, f)

if __name__ == "__main__":
    runonce()
