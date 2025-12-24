# Alex ANderson, Vu Hung-Nghi 
# File saved in src not AI agents to generate training data for our Neural network code in our other file
# train_nn.py — simple 2-layer sigmoid net using NumPy
import numpy as np
import csv
np.random.seed(0)

CSV = "train_log.csv"
H = 8         # hidden size (tune)
LR = 0.5      # learning rate (tune 0.1–1.0)
EPOCHS = 200  # tune (watch loss)
BATCH = 64

# 1) load data
with open(CSV) as f:
    rdr = csv.reader(f)
    header = next(rdr)
    data = np.array([[float(x) for x in row] for row in rdr], dtype=np.float64)
X, y = data[:, :-1], data[:, -1:].copy()
N, F = X.shape

# 2) init weights
W1 = np.random.uniform(-1, 1, size=(F, H))
b1 = np.zeros((1, H))
W2 = np.random.uniform(-1, 1, size=(H, 1))
b2 = np.zeros((1, 1))

def sigm(z): return 1.0 / (1.0 + np.exp(-z))
def d_sigm(a): return a * (1.0 - a)  # derivative wrt activation

# 3) train
for ep in range(1, EPOCHS+1):
    # shuffle
    idx = np.random.permutation(N)
    Xs, ys = X[idx], y[idx]
    # mini-batches
    for i in range(0, N, BATCH):
        xb = Xs[i:i+BATCH]; yb = ys[i:i+BATCH]
        # forward
        z1 = xb @ W1 + b1        # [B,H]
        a1 = sigm(z1)
        z2 = a1 @ W2 + b2        # [B,1]
        yhat = sigm(z2)

        # loss (MSE)
        err = yhat - yb          # [B,1]
        dL_dyhat = err           # MSE’ wrt yhat up to 2/B factor, absorbed into LR

        # backprop
        dL_dz2 = dL_dyhat * d_sigm(yhat)   # [B,1]
        dL_dW2 = a1.T @ dL_dz2             # [H,1]
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # [1,1]

        dL_da1 = dL_dz2 @ W2.T             # [B,H]
        dL_dz1 = dL_da1 * d_sigm(a1)       # [B,H]
        dL_dW1 = xb.T @ dL_dz1             # [F,H]
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # [1,H]

        # SGD step
        W1 -= LR * dL_dW1 / max(1, len(xb))
        b1 -= LR * dL_db1 / max(1, len(xb))
        W2 -= LR * dL_dW2 / max(1, len(xb))
        b2 -= LR * dL_db2 / max(1, len(xb))

    # monitor
    z1 = X @ W1 + b1; a1 = sigm(z1)
    z2 = a1 @ W2 + b2; yhat = sigm(z2)
    mse = np.mean((yhat - y)**2)
    if ep % 10 == 0:
        print(f"epoch {ep:4d}  mse={mse:.5f}")

# 4) print weights in pasteable Python lists
def as_list(A): return np.round(A.astype(float).tolist(), 6).tolist()
print("\n# Paste these into your agent:")
print(f"W1 = {as_list(W1)}")
print(f"b1 = {as_list(b1.flatten())}")
print(f"W2 = {as_list(W2)}")
print(f"b2 = {as_list(b2.flatten())}")