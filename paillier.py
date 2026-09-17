import random
from sympy import isprime, nextprime, gcd

def generate_keypair(bits=64):
    # toy prime generation — NOT secure, just illustrative
    p = nextprime(random.getrandbits(bits))
    q = nextprime(random.getrandbits(bits))
    n = p * q
    n_sq = n * n
    g = n + 1  # simplified generator choice
    lam = (p - 1) * (q - 1) // gcd(p - 1, q - 1)  # lcm(p-1, q-1)
    mu = pow(lam, -1, n)  # modular inverse
    return (n, g), (lam, mu, n)

SCALE = 10 ** 6

def encode(x, n):
    return int(round(x * SCALE)) % n

def decode(m, n):
    if m > n // 2:
        m -= n
    return m / SCALE

def encrypt(pub, m):
    n, g = pub
    n_sq = n * n
    while True:
        r = random.randrange(1, n)
        if gcd(r, n) == 1:
            break
    c = (pow(g, m, n_sq) * pow(r, n, n_sq)) % n_sq
    return c

def decrypt(priv, pub, c):
    lam, mu, n = priv
    n_sq = n * n
    x = pow(c, lam, n_sq)
    L = (x - 1) // n
    m = (L * mu) % n
    return m

def add_encrypted(c1, c2, pub):
    n, g = pub
    n_sq = n * n
    return (c1 * c2) % n_sq  # <-- this is the "homomorphic" step

# demo
if __name__ == "__main__":
    pub, priv = generate_keypair()
    a, b = 15, 27
    ca, cb = encrypt(pub, a), encrypt(pub, b)
    c_sum = add_encrypted(ca, cb, pub)
    result = decrypt(priv, pub, c_sum)
    print(result)  # -> 42, without ever decrypting a or b individually