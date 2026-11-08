# Federated two-moons MLP, then on-device int8

Three clients train a tiny **2 → 8 → 2** classifier (`ReLU` hidden, `softmax` out) on noisy two-moons points. Gradients are encrypted with Paillier so the server only ever sees ciphertexts. After training we freeze the weights, quantize them to int8, and run inference in C++ with no Python and no PyTorch.

## Pipeline

```
train  →  quantize  →  export  →  C++ inference
```

**1. Data + train**

```bash
python3 make_dataset.py   # noisy two-moons, split non-IID across 3 clients
python3 keygen.py         # once
python3 train.py          # 20 encrypted rounds → global_model.json
```

**2. Quantize and export**

```bash
python3 quantize.py
```

Writes `export/`:

- `weights_float32.bin` — all weights/biases as float32 (4× larger)
- `weights_int8.bin` — same tensors, symmetric per-tensor int8
- `scales.txt` — dequant scales for `W1,b1,W2,b2` plus input/hidden activations (activations calibrated on client 1 at the 80th percentile — the server never sees a central dataset)
- `test_xy.txt` — held-out `(x0, x1, y)` rows

**3. C++ int8 forward pass** (C stdlib only):

```bash
make
./infer
```

Integer matmul → dequant → ReLU → integer matmul → softmax. Prints predicted class `0` or `1` per test row.

**4. Compare**

```bash
python3 evaluate.py
```

Prints float32 accuracy, int8 accuracy, accuracy drop, and the 4× size change. Accuracy is ordinary classification: predicted class vs label.

Reset with `python3 reset.py` (does not delete keys).
