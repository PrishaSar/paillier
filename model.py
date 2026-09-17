USE_RELU = False  # True if you switch to data with y >= 0

def relu(x):
    return max(0.0, x)

def relu_grad(z):
    return 1.0 if z > 0 else 0.0

def predict(w, b, x):
    z = w * x + b
    if USE_RELU:
        return relu(z)
    return z

def activation_grad(z):
    if USE_RELU:
        return relu_grad(z)
    return 1.0
