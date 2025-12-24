# Alex Anderson &n Vu Hung-Nghi
# 11/06/25
# ---------------------------------------------------------------------------
# HOW TO RUN THIS NEURAL NETWORK PROGRAM (STEP-BY-STEP TERMINAL 
# INSTRUCTIONS for user if needed)
# ---------------------------------------------------------------------------
# 1. Make sure Python 3 is installed.
#       In a terminal, type:
#           python3 --version
#       or on Windows:
#           python --version
#
# 2. Install NumPy (required library).
#       If you do NOT have NumPy installed, type ONE of the following:
#           pip install numpy
#       or (Mac/Linux):
#           pip3 install numpy
#
# 3. Save this file to your computer as:
#           nn_train.py
#
# 4. Open a terminal and navigate to the folder where the file is saved.
#       Example commands (you may change path as needed):
#           cd Desktop
#           cd my_project_folder
#
# 5. Run the program by typing:
#           python nn_train.py
#       OR if your system uses python3:
#           python3 nn_train.py
#
# 6. What you will see when it runs:
#       - The terminal will print a number every epoch (the average error).
#       - The number will decrease over time as the network learns.
#       - Training will automatically stop once the error goes below 0.05.
#



import numpy as np
np.random.seed(2)  # Reproducible randomness

# ---------------------------------------------------------------------------
# Given dataset: 4 binary inputs -> 1 binary target.
# Each example is (x, t) where x is a length-4 vector of 0/1, and t is [0] or [1].
# (Basically is unknown boolean function of 4 inputs.)
# ---------------------------------------------------------------------------
examples = np.array([
    ([0,0,0,0],[0]), ([0,0,0,1],[1]), ([0,0,1,0],[0]), ([0,0,1,1],[1]),
    ([0,1,0,0],[0]), ([0,1,0,1],[1]), ([0,1,1,0],[0]), ([0,1,1,1],[1]),
    ([1,0,0,0],[1]), ([1,0,0,1],[1]), ([1,0,1,0],[1]), ([1,0,1,1],[1]),
    ([1,1,0,0],[0]), ([1,1,0,1],[0]), ([1,1,1,0],[0]), ([1,1,1,1],[1])
], dtype=object)

# ---------------------------------------------------------------------------
# Network architecture
# IN:  number of input features
# H:   number of hidden units
# OUT: number of outputs
# We use a bias by augmenting inputs/hidden with a constant 1.0 (bias trick).
# ---------------------------------------------------------------------------
IN, H, OUT = 4, 8, 1

# ---------------------------------------------------------------------------
# Adam optimizer hyperparameters
# lr:  learning rate
# b1:  beta1 (first-moment exponential decay)
# b2:  beta2 (second-moment exponential decay)
# eps: numerical stabilizer to avoid division by 0
# ---------------------------------------------------------------------------
lr = 0.05
b1, b2, eps = 0.9, 0.999, 1e-8

# ---------------------------------------------------------------------------
# Weights
# W1: (H, IN+1) maps 4 inputs + 1 bias -> 8 hidden units
# W2: (OUT, H+1) maps 8 hidden + 1 bias -> 1 output
# Random init in [-1, 1]
# ---------------------------------------------------------------------------
W1 = np.random.uniform(-1, 1, (H, IN+1))
W2 = np.random.uniform(-1, 1, (OUT, H+1))

# Adam state (first moment m*, second moment v*)
m1 = np.zeros_like(W1); v1 = np.zeros_like(W1)
m2 = np.zeros_like(W2); v2 = np.zeros_like(W2)
tstep = 0  # Adam time step for bias correction

# Sigmoid activation for hidden and output layers
sigmoid = lambda z: 1/(1+np.exp(-z))

def train_epoch():
    """
    Runs 10 stochastic updates (mini-steps) and returns the average absolute error.
    Notes:
      - We do forward pass -> compute error -> backprop -> Adam update.
      - Bias is handled by appending a 1.0 to the input/hidden vectors (xb, hb).
      - The gradient signs are consistent with the 'err = t - y' convention used here.
    """
    global W1, W2, m1, v1, m2, v2, tstep
    errs = []

    for _ in range(10):
        # -----------------------------
        # Sample one training pair (x, t)
        # -----------------------------
        x, t = examples[np.random.randint(len(examples))]
        x, t = np.array(x, float), np.array(t, float)  # ensure float arrays

        # -----------------------------
        # Forward pass
        # -----------------------------
        xb = np.append(x, 1.0)      # input + bias -> shape (IN+1,)
        h  = sigmoid(W1 @ xb)       # hidden pre-activation -> sigmoid -> shape (H,)
        hb = np.append(h, 1.0)      # hidden + bias -> shape (H+1,)
        y  = sigmoid(W2 @ hb)       # output -> shape (OUT,)

        # -----------------------------
        # Error and output delta
        # err is target - prediction (scalar here)
        # For squared error L = 0.5*(t - y)^2, derivative w.r.t. output pre-activation
        # is proportional to (t - y) * y*(1 - y) under this sign convention.
        # -----------------------------
        err = t - y
        delta_o = err * y*(1-y)     # shape (OUT,)
        errs.append(abs(err)[0])    # track |error| for reporting

        # -----------------------------
        # Backprop to hidden layer
        # - W2[:, :H] are the weights from hidden units (no bias column) to output.
        # - Elementwise multiply by h*(1-h) for sigmoid derivative at hidden.
        # -----------------------------
        delta_h = (W2[:, :H].T @ delta_o) * h*(1-h)  # shape (H,)

        # -----------------------------
        # Gradients for weights (outer products)
        # g2: dL/dW2 has shape (OUT, H+1)
        # g1: dL/dW1 has shape (H, IN+1)
        # -----------------------------
        g2 = delta_o.reshape(OUT,1) @ hb.reshape(1,H+1)
        g1 = delta_h.reshape(H,1)   @ xb.reshape(1,IN+1)

        # -----------------------------
        # Adam updates (with bias correction)
        # Note: This code uses W += ... which is consistent with the sign used
        # in delta terms above. If you switch to the (y - t) convention, you'd
        # typically use W -= lr * update.
        # -----------------------------
        tstep += 1

        # Adam for W2
        m2 = b1*m2 + (1-b1)*g2          # first moment
        v2 = b2*v2 + (1-b2)*(g2*g2)     # second moment
        m2h = m2/(1-b1**tstep)          # bias-corrected first moment
        v2h = v2/(1-b2**tstep)          # bias-corrected second moment
        W2 += lr * m2h / (np.sqrt(v2h)+eps)

        # Adam for W1
        m1 = b1*m1 + (1-b1)*g1
        v1 = b2*v1 + (1-b2)*(g1*g1)
        m1h = m1/(1-b1**tstep)
        v1h = v1/(1-b2**tstep)
        W1 += lr * m1h / (np.sqrt(v1h)+eps)

    # Average absolute error over these 10 mini-steps
    return float(np.mean(errs))

# ---------------------------------------------------------------------------
# Training loop: keep running epochs until average |error| < 0.05
# Prints the running avg error so you can see convergence.
# ---------------------------------------------------------------------------
while True:
    avg_err = train_epoch()
    print(f"{avg_err:.6f}")
    if avg_err < 0.05:
        break




    