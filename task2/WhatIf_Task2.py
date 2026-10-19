# CODE MAP of WhatIf_Task2.py
# Settings        libraries, file paths, SGD and ADAM constants, number of epochs, random seed, 0.001 threshold
# Load data       read RegressionData.csv and InitialWeights.npz; build xtest
# Network         forward() as in Task2.py; backward() with a switch that replaces the tanh slope by 1
# Training error  mean of (d - y)^2 over the 41 samples; first epoch below the threshold
# Train SGD       SGD from the given weights, normal or with the forgotten derivative
# Train ADAM      ADAM from the given weights (m, v and t start at zero), normal or with the forgotten derivative
# Runs            the four runs: SGD and ADAM, each normal and with the forgotten derivative (same seed)
# Results         print the final training error and the first epoch below 0.001 of every run
# Plots           one figure: fit on xtest (left) and learning curves (right) of the four runs
# AI use          Claude Opus 5.5 (claude-opus-5-5), High
#
# How to run:   python WhatIf_Task2.py      (paths are relative to this file)
#
# WHAT-IF: THE FORGOTTEN DERIVATIVE
# In backpropagation the delta of a hidden node is the error passed back from the output layer,
# multiplied by the slope of tanh at that node, 1 - a^2 (Eq. (8) and (9) of the derivation).
# This file is Task2.py with ONE change: backward() can replace that factor by 1, as if it had
# been forgotten. Everything else (data, initial weights, settings, seed, shuffling) is identical,
# so the "normal" runs below reproduce the numbers printed by Task2.py exactly.
#
# Notation is the same as in Task2.py (see the header there):
#   v1, a1 = induced local fields and outputs of the 3 hidden tanh nodes,  y = network output.
# "Eq. (n)" in the comments = equation n of the derivation (equation sheet in FINALISE.md).


# %% Settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    HERE = Path(__file__).resolve().parent
except NameError:                   # running cell by cell in an interactive window
    HERE = Path.cwd()
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# The same fixed settings as Task2.py.
ETA = 0.1           # SGD learning rate (eta)
ALPHA = 0.005       # ADAM step size (alpha)
BETA1 = 0.9         # ADAM decay rate of the gradient average m
BETA2 = 0.999       # ADAM decay rate of the squared-gradient average v
EPS = 1e-8          # ADAM epsilon
N_EPOCHS = 3000
SEED = 1
THRESHOLD = 0.001


# %% Load data
data = pd.read_csv(DATA_DIR / "RegressionData.csv")
X = data["x"].to_numpy().reshape(1, -1)     # inputs,          shape (1, 41)
D = data["d"].to_numpy().reshape(1, -1)     # desired outputs, shape (1, 41)
N = X.shape[1]                              # 41

init = np.load(DATA_DIR / "InitialWeights.npz")
W1_INIT = init["W1r"]   # (3, 1)
B1_INIT = init["b1r"]   # (3, 1)
W2_INIT = init["W2r"]   # (1, 3)
B2_INIT = init["b2r"]   # (1, 1)

xtest = np.linspace(-0.97, 0.93, 20).reshape(1, -1)    # shape (1, 20)


# %% Network
def forward(x, W1, b1, W2, b2):
    """Forward pass, identical to Task2.py.
    In : x (1, n); weights.   Out: a1 (3, n) hidden outputs, y (1, n) network output."""
    v1 = W1 @ x + b1    # Eq. (1)
    a1 = np.tanh(v1)    # Eq. (2)
    v2 = W2 @ a1 + b2   # Eq. (3)
    y = v2              # Eq. (4) linear output
    return a1, y


def backward(x, d, a1, y, W2, forget_derivative):
    """Backpropagation for ONE sample with E = 1/2 (d - y)^2.

    In : x, d, a1, y, W2 as in Task2.py;
         forget_derivative - False: normal backpropagation (as in Task2.py),
                             True : the tanh slope 1 - a1^2 is replaced by 1 (the What-if).
    Out: the gradients dE/dW1 (3, 1), dE/db1 (3, 1), dE/dW2 (1, 3), dE/db2 (1, 1).
    """
    e = d - y                               # Eq. (5)

    # ---- Output layer: unchanged ----
    phi2_slope = 1.0
    delta2 = e * phi2_slope                 # Eq. (6)
    dE_dW2 = -delta2 @ a1.T                 # Eq. (7)
    dE_db2 = -delta2                        # Eq. (7)

    # ---- Hidden layer: THE ONLY CHANGE FROM Task2.py ----
    if forget_derivative:
        tanh_slope = np.ones_like(a1)       # WHAT-IF: the factor 1 - a1^2 is "forgotten" (replaced by 1)
    else:
        tanh_slope = 1 - a1 ** 2            # Eq. (8)  normal: slope of tanh, 1 - a^2
    delta1 = (W2.T @ delta2) * tanh_slope   # Eq. (9)
    dE_dW1 = -delta1 @ x.T                  # Eq. (10)
    dE_db1 = -delta1                        # Eq. (10)

    return dE_dW1, dE_db1, dE_dW2, dE_db2


# %% Training error
def training_error(W1, b1, W2, b2):
    """Eq. (12): mean of (d - y)^2 over all 41 training samples, using the current weights."""
    _, Y = forward(X, W1, b1, W2, b2)
    return np.mean((D - Y) ** 2)


def first_epoch_below(errors, threshold):
    """Out: the first epoch (counting from 1) whose error is below the threshold, or None."""
    below = np.flatnonzero(errors < threshold)
    return int(below[0]) + 1 if below.size > 0 else None


# %% Train SGD
def train_sgd(forget_derivative):
    """SGD from the given weights, as in Task2.py.
    In : forget_derivative - passed on to backward().
    Out: the trained weights (W1, b1, W2, b2) and the learning curve (one error per epoch)."""
    W1, b1, W2, b2 = W1_INIT.copy(), B1_INIT.copy(), W2_INIT.copy(), B2_INIT.copy()
    rng = np.random.default_rng(SEED)       # same seed -> same shuffles as Task2.py
    errors = np.zeros(N_EPOCHS)

    for epoch in range(N_EPOCHS):
        order = rng.permutation(N)
        for k in order:
            x = X[:, [k]]
            d = D[:, [k]]
            a1, y = forward(x, W1, b1, W2, b2)
            dE_dW1, dE_db1, dE_dW2, dE_db2 = backward(x, d, a1, y, W2, forget_derivative)
            W2 = W2 - ETA * dE_dW2          # Eq. (11)
            b2 = b2 - ETA * dE_db2
            W1 = W1 - ETA * dE_dW1
            b1 = b1 - ETA * dE_db1
        errors[epoch] = training_error(W1, b1, W2, b2)

    return (W1, b1, W2, b2), errors


# %% Train ADAM
def adam_update(w, grad, m, v, t):
    """One ADAM step for one weight array, as in Task2.py.
    In : w, grad, m, v (same shape), t >= 1.   Out: new w, m, v."""
    m = BETA1 * m + (1 - BETA1) * grad          # Eq. (13)
    v = BETA2 * v + (1 - BETA2) * grad ** 2     # Eq. (14)
    m_hat = m / (1 - BETA1 ** t)                # Eq. (15)
    v_hat = v / (1 - BETA2 ** t)                # Eq. (15)
    w = w - ALPHA * m_hat / np.sqrt(v_hat + EPS)    # Eq. (16)
    return w, m, v


def train_adam(forget_derivative):
    """ADAM from the given weights (m, v and t start at zero), as in Task2.py.
    In : forget_derivative - passed on to backward().
    Out: the trained weights (W1, b1, W2, b2) and the learning curve (one error per epoch)."""
    W1, b1, W2, b2 = W1_INIT.copy(), B1_INIT.copy(), W2_INIT.copy(), B2_INIT.copy()
    m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    t = 0
    rng = np.random.default_rng(SEED)
    errors = np.zeros(N_EPOCHS)

    for epoch in range(N_EPOCHS):
        order = rng.permutation(N)
        for k in order:
            x = X[:, [k]]
            d = D[:, [k]]
            a1, y = forward(x, W1, b1, W2, b2)
            dE_dW1, dE_db1, dE_dW2, dE_db2 = backward(x, d, a1, y, W2, forget_derivative)
            t += 1
            W2, m_W2, v_W2 = adam_update(W2, dE_dW2, m_W2, v_W2, t)
            b2, m_b2, v_b2 = adam_update(b2, dE_db2, m_b2, v_b2, t)
            W1, m_W1, v_W1 = adam_update(W1, dE_dW1, m_W1, v_W1, t)
            b1, m_b1, v_b1 = adam_update(b1, dE_db1, m_b1, v_b1, t)
        errors[epoch] = training_error(W1, b1, W2, b2)

    return (W1, b1, W2, b2), errors


# %% Runs
# Each entry: (label, optimiser, forget_derivative). Every run starts from the given weights
# with the same seed, so the only difference between "normal" and "forgotten" is Eq. (8).
RUNS = [("SGD, normal", "SGD", False),
        ("SGD, derivative forgotten", "SGD", True),
        ("ADAM, normal", "ADAM", False),
        ("ADAM, derivative forgotten", "ADAM", True)]

results = {}    # label -> (trained weights, learning curve)
for label, optimiser, forget in RUNS:
    print(f"Training: {label} (3000 epochs) ...", flush=True)
    train = train_sgd if optimiser == "SGD" else train_adam
    results[label] = train(forget)


# %% Results
def epoch_text(epoch):
    """In : an epoch number or None.  Out: 'epoch 123' or 'not reached'."""
    return f"epoch {epoch}" if epoch is not None else "not reached"


print()
print("What-if (Task 2): the forgotten derivative - in the hidden delta, the tanh slope "
      "1 - a1^2 is replaced by 1")
print(f"{N_EPOCHS} epochs, seed = {SEED}, SGD eta = {ETA}, ADAM alpha = {ALPHA}")
print(f"{'Run':<28}{'Final training error':<24}First epoch below {THRESHOLD}")
for label, _, _ in RUNS:
    errors = results[label][1]
    print(f"{label:<28}{errors[-1]:<24.6f}{epoch_text(first_epoch_below(errors, THRESHOLD))}")

for optimiser in ("SGD", "ADAM"):
    normal = results[f"{optimiser}, normal"][1][-1]
    forgotten = results[f"{optimiser}, derivative forgotten"][1][-1]
    print(f"{optimiser}: final error with the derivative forgotten / normal = "
          f"{forgotten / normal:.2f}")
print("(The normal runs use the same code, data and seed as Task2.py, so they print the "
      "same numbers.)")


# %% Plots
FIG_DIR.mkdir(exist_ok=True)

# Colour = optimiser (same colours as Task2.py); line style = normal (solid) or forgotten (dashed).
STYLE = {"SGD, normal": ("#2a78d6", "-"),
         "SGD, derivative forgotten": ("#2a78d6", "--"),
         "ADAM, normal": ("#eb6834", "-"),
         "ADAM, derivative forgotten": ("#eb6834", "--")}
DATA_COLOUR = "#52514e"

fig, (ax_fit, ax_curve) = plt.subplots(1, 2, figsize=(12, 5.4))

# Left: network output on xtest for the four runs, with the training data.
ax_fit.scatter(X.ravel(), D.ravel(), s=22, color=DATA_COLOUR, zorder=3,
               label="Training data (41 samples)")
for label, _, _ in RUNS:
    colour, linestyle = STYLE[label]
    _, y_test = forward(xtest, *results[label][0])
    ax_fit.plot(xtest.ravel(), y_test.ravel(), linestyle=linestyle, marker="o", markersize=4,
                color=colour, linewidth=2, label=label)
ax_fit.set_xlabel("Input x")
ax_fit.set_ylabel("Output (d for data, y for network)")
ax_fit.set_title("Fit on xtest after 3000 epochs")
ax_fit.grid(True, color="#e3e2de", linewidth=0.8)

# Right: learning curves of the four runs (log scales on both axes, as in Task2.py, right panel).
# ADAM is drawn first and SGD on top, so that both stay visible.
epochs = np.arange(1, N_EPOCHS + 1)
for label in ["ADAM, normal", "ADAM, derivative forgotten",
              "SGD, normal", "SGD, derivative forgotten"]:
    colour, linestyle = STYLE[label]
    ax_curve.plot(epochs, results[label][1], linestyle=linestyle, color=colour, linewidth=1.0)
threshold_line = ax_curve.axhline(THRESHOLD, color=DATA_COLOUR, linestyle=":", linewidth=1.5,
                                  label=f"Threshold {THRESHOLD}")
ax_curve.set_xscale("log")
ax_curve.set_yscale("log")
ax_curve.set_xlabel("Epoch (log scale)")
ax_curve.set_ylabel("Training error, mean of (d - y)$^2$ (log scale)")
ax_curve.set_title("Learning curves")
ax_curve.grid(True, which="both", color="#e3e2de", linewidth=0.8)

# One legend for both panels, below them, so that it covers no data.
handles, labels = ax_fit.get_legend_handles_labels()
handles.append(threshold_line)
labels.append(threshold_line.get_label())
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

fig.suptitle("What-if: tanh slope 1 - a$^2$ replaced by 1 in backpropagation (WhatIf_Task2.py)")
fig.tight_layout(rect=(0, 0.1, 1, 1))     # leave the bottom 10% free for the legend
fig.savefig(FIG_DIR / "WhatIf_Task2.png", dpi=200)

print(f"Figure saved in {FIG_DIR}")
plt.show()
