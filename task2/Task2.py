# CODE MAP of Task2.py
# Settings        libraries, file paths, SGD and ADAM constants, number of epochs, random seed, 0.001 threshold
# Load data       read RegressionData.csv and InitialWeights.npz; build xtest
# Network         forward(): hidden outputs a1 and network output y; backward(): deltas, then gradients of E
# Training error  mean of (d - y)^2 over the 41 samples; first epoch below the threshold
# Train SGD       train from the given weights with SGD (update after every sample), record the learning curve
# Train ADAM      train from the given weights with ADAM (m, v and t start at zero), record the learning curve
# Results         print the final training errors and the epoch at which each first goes below 0.001
# Plots           network output on xtest with the training data; both learning curves on one figure
# AI use          Claude Opus 5.5 (claude-opus-5-5), High
#
# How to run:   python Task2.py      (paths are relative to this file, so it runs from any folder)
#
# Notation (the same as the Week 3 lecture and the derivation in the appendix):
#   layer (1) = hidden layer, 3 nodes with tanh activation
#   layer (2) = output layer, 1 node with linear activation
#   W1[j, 0] = w_1j^(1)   weight from the input x to hidden node j        (j = 1, 2, 3)
#   b1[j, 0] = w_0j^(1)   bias of hidden node j (a weight on a constant input of 1)
#   W2[0, j] = w_j1^(2)   weight from hidden node j to the output node
#   b2[0, 0] = w_01^(2)   bias of the output node
#   v1 = v^(1)            induced local fields of the hidden nodes (adder output)
#   a1 = x_out^(1)        hidden outputs, a1 = tanh(v1)
#   v2 = v^(2)            induced local field of the output node
#   y  = x_out^(2)        network output, y = v2 (linear)
# "Eq. (n)" in the comments = equation n of the derivation (equation sheet in FINALISE.md).


# %% Settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Folder of this file. Everything is found relative to it, so the code runs from any working
# directory. (When the cells are run one by one in an interactive window, __file__ does not
# exist; then we fall back to the current working directory.)
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
DATA_DIR = HERE / "data"        # RegressionData.csv and InitialWeights.npz
FIG_DIR = HERE / "figures"      # the figures are saved here for the report

# Fixed settings from the assignment, so that results can be compared between teams.
ETA = 0.1           # SGD learning rate (eta)
ALPHA = 0.005       # ADAM step size (alpha; written eta on the Week 4 ADAM slides)
BETA1 = 0.9         # ADAM decay rate of the gradient average m      (beta_m in the slides)
BETA2 = 0.999       # ADAM decay rate of the squared-gradient average v (beta_v in the slides)
EPS = 1e-8          # ADAM epsilon: keeps the step finite when v_hat is ~0
N_EPOCHS = 3000     # one epoch = every one of the 41 samples presented once
SEED = 1            # the same seed is set at the start of every training run
THRESHOLD = 0.001   # we report the first epoch at which the training error goes below this


# %% Load data
# RegressionData.csv has two columns: x (input) and d (desired output), 41 rows.
# We store the samples as ROW vectors of shape (1, 41): one column per sample. This matches
# the column-vector convention of the network (one input = one column), so the same forward()
# works for a single sample (1, 1) and for all samples at once (1, 41).
data = pd.read_csv(DATA_DIR / "RegressionData.csv")
X = data["x"].to_numpy().reshape(1, -1)     # inputs,          shape (1, 41)
D = data["d"].to_numpy().reshape(1, -1)     # desired outputs, shape (1, 41)
N = X.shape[1]                              # number of training samples = 41

# The given initial weights. Every training run starts from these values.
init = np.load(DATA_DIR / "InitialWeights.npz")
W1_INIT = init["W1r"]   # (3, 1) input  -> hidden weights
B1_INIT = init["b1r"]   # (3, 1) hidden biases
W2_INIT = init["W2r"]   # (1, 3) hidden -> output weights
B2_INIT = init["b2r"]   # (1, 1) output bias

# Test inputs: 20 equally spaced values from -0.97 to 0.93 (they lie between the training x
# values, so they show whether the network has learnt a smooth curve, not just the 41 points).
xtest = np.linspace(-0.97, 0.93, 20).reshape(1, -1)    # shape (1, 20)


# %% Network
def forward(x, W1, b1, W2, b2):
    """Forward pass of the 1-3-1 network.

    In : x of shape (1, n) - one input per column (n = 1 during training, 41 or 20 otherwise),
         and the current weights W1 (3, 1), b1 (3, 1), W2 (1, 3), b2 (1, 1).
    Out: a1 of shape (3, n) - the outputs of the 3 hidden nodes,
         y  of shape (1, n) - the network output.
    """
    v1 = W1 @ x + b1    # Eq. (1)  v_j^(1) = w_1j^(1) x + w_0j^(1); (3,1)@(1,n) + (3,1) -> (3,n)
    a1 = np.tanh(v1)    # Eq. (2)  x_out,j^(1) = tanh(v_j^(1))
    v2 = W2 @ a1 + b2   # Eq. (3)  v^(2) = sum_j w_j1^(2) x_out,j^(1) + w_01^(2); (1,3)@(3,n) -> (1,n)
    y = v2              # Eq. (4)  linear output node: y = phi(v^(2)) = v^(2)
    return a1, y


def backward(x, d, a1, y, W2):
    """Backpropagation for ONE sample with E = 1/2 (d - y)^2.

    In : x (1, 1) the input, d (1, 1) the desired output,
         a1 (3, 1) and y (1, 1) from forward() for this sample,
         W2 (1, 3) the current output weights (needed to pass the error back).
    Out: the gradients dE/dW1 (3, 1), dE/db1 (3, 1), dE/dW2 (1, 3), dE/db2 (1, 1).
    All gradients are computed BEFORE any weight changes, so the hidden delta uses the
    same W2 that produced y.
    """
    e = d - y                               # Eq. (5)  error of the output node, e = d - y

    # ---- Output layer (layer 2) ----
    phi2_slope = 1.0                        # derivative of the linear activation: d(v)/dv = 1
    delta2 = e * phi2_slope                 # Eq. (6)  local gradient delta^(2) = e * phi'(v^(2)) = e
    dE_dW2 = -delta2 @ a1.T                 # Eq. (7)  dE/dw_j1^(2) = -delta^(2) x_out,j^(1); (1,1)@(1,3)
    dE_db2 = -delta2                        # Eq. (7)  bias: its input is the constant 1

    # ---- Hidden layer (layer 1) ----
    tanh_slope = 1 - a1 ** 2                # Eq. (8)  slope of tanh at each hidden node: 1 - a^2
    delta1 = (W2.T @ delta2) * tanh_slope   # Eq. (9)  delta_j^(1) = delta^(2) w_j1^(2) (1 - a_j^2); (3,1)
    dE_dW1 = -delta1 @ x.T                  # Eq. (10) dE/dw_1j^(1) = -delta_j^(1) x; (3,1)@(1,1) -> (3,1)
    dE_db1 = -delta1                        # Eq. (10) bias: its input is the constant 1

    return dE_dW1, dE_db1, dE_dW2, dE_db2


# %% Training error
def training_error(W1, b1, W2, b2):
    """Eq. (12): mean of (d - y)^2 over all 41 training samples, using the current weights.
    No 1/2 here: the brief defines the training error as the plain mean squared error."""
    _, Y = forward(X, W1, b1, W2, b2)       # all 41 samples at once, Y has shape (1, 41)
    return np.mean((D - Y) ** 2)


def first_epoch_below(errors, threshold):
    """In : errors - learning curve, one training error per epoch; threshold - e.g. 0.001.
    Out: the first epoch (counting from 1) whose error is below the threshold, or None."""
    below = np.flatnonzero(errors < threshold)      # indices of all epochs below the threshold
    return int(below[0]) + 1 if below.size > 0 else None    # +1: index 0 is epoch 1


# %% Train SGD
def train_sgd():
    """Train the network with stochastic gradient descent from the given weights.

    Out: the trained weights (W1, b1, W2, b2) and the learning curve (one error per epoch).
    """
    # Start from the GIVEN weights. .copy() makes new arrays, so training never changes
    # W1_INIT etc. and the ADAM run below also starts from the given values.
    W1, b1, W2, b2 = W1_INIT.copy(), B1_INIT.copy(), W2_INIT.copy(), B2_INIT.copy()

    # Same seed at the start of every run: SGD and ADAM (and the What-if file) see exactly
    # the same shuffled order of samples, so a run gives the same result in any file.
    rng = np.random.default_rng(SEED)
    errors = np.zeros(N_EPOCHS)

    for epoch in range(N_EPOCHS):
        order = rng.permutation(N)          # shuffle every epoch: a random order of 0..40
        for k in order:
            x = X[:, [k]]                   # one input,          shape (1, 1)
            d = D[:, [k]]                   # its desired output, shape (1, 1)

            a1, y = forward(x, W1, b1, W2, b2)
            dE_dW1, dE_db1, dE_dW2, dE_db2 = backward(x, d, a1, y, W2)

            # Eq. (11): steepest descent, w_new = w_old - eta dE/dw.
            # Because dE/dw = -delta * x_out, this is the lecture's w_new = w_old + eta delta x_out.
            W2 = W2 - ETA * dE_dW2
            b2 = b2 - ETA * dE_db2
            W1 = W1 - ETA * dE_dW1
            b1 = b1 - ETA * dE_db1

        errors[epoch] = training_error(W1, b1, W2, b2)  # error at the END of this epoch

    return (W1, b1, W2, b2), errors


print("Training with SGD for 3000 epochs (this takes a few seconds) ...", flush=True)
sgd_weights, sgd_errors = train_sgd()


# %% Train ADAM
def adam_update(w, grad, m, v, t):
    """One ADAM step for one weight array (Week 4, Adam (3) slide).

    In : w - current weights; grad - dE/dw for this sample (same shape as w);
         m, v - running averages of the gradient and of the squared gradient for w;
         t - number of updates so far, including this one (t >= 1).
    Out: the new w, m and v.
    Every operation is element-wise, so each of the 10 weights gets its own step size.
    """
    m = BETA1 * m + (1 - BETA1) * grad          # Eq. (13) average of past gradients (momentum)
    v = BETA2 * v + (1 - BETA2) * grad ** 2     # Eq. (14) average of past squared gradients (RMSProp)
    m_hat = m / (1 - BETA1 ** t)                # Eq. (15) bias correction: m and v start at 0,
    v_hat = v / (1 - BETA2 ** t)                #          so early averages are too small
    # Eq. (16) as on the lecture slide, with eps inside the square root. (The original ADAM
    # paper writes sqrt(v_hat) + eps; with eps = 1e-8 the difference is negligible here.)
    w = w - ALPHA * m_hat / np.sqrt(v_hat + EPS)
    return w, m, v


def train_adam():
    """Train the network with ADAM from the given weights (m, v and t start at zero).

    Out: the trained weights (W1, b1, W2, b2) and the learning curve (one error per epoch).
    """
    W1, b1, W2, b2 = W1_INIT.copy(), B1_INIT.copy(), W2_INIT.copy(), B2_INIT.copy()

    # One m and one v per weight, same shape as the weight array, all starting at zero.
    m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    t = 0                                   # step count: +1 at every weight update (every sample)

    rng = np.random.default_rng(SEED)       # same seed -> same shuffles as the SGD run
    errors = np.zeros(N_EPOCHS)

    for epoch in range(N_EPOCHS):
        order = rng.permutation(N)
        for k in order:
            x = X[:, [k]]
            d = D[:, [k]]

            # Forward and backward passes are exactly the same as for SGD ...
            a1, y = forward(x, W1, b1, W2, b2)
            dE_dW1, dE_db1, dE_dW2, dE_db2 = backward(x, d, a1, y, W2)

            # ... only the update rule is different.
            t += 1      # after the last sample of the last epoch, t = 3000 * 41 = 123000
            W2, m_W2, v_W2 = adam_update(W2, dE_dW2, m_W2, v_W2, t)
            b2, m_b2, v_b2 = adam_update(b2, dE_db2, m_b2, v_b2, t)
            W1, m_W1, v_W1 = adam_update(W1, dE_dW1, m_W1, v_W1, t)
            b1, m_b1, v_b1 = adam_update(b1, dE_db1, m_b1, v_b1, t)

        errors[epoch] = training_error(W1, b1, W2, b2)

    return (W1, b1, W2, b2), errors


print("Training with ADAM for 3000 epochs ...", flush=True)
adam_weights, adam_errors = train_adam()


# %% Results
sgd_epoch = first_epoch_below(sgd_errors, THRESHOLD)
adam_epoch = first_epoch_below(adam_errors, THRESHOLD)
initial_error = training_error(W1_INIT, B1_INIT, W2_INIT, B2_INIT)  # before any training


def epoch_text(epoch):
    """In : an epoch number or None.  Out: 'epoch 123' or 'not reached'."""
    return f"epoch {epoch}" if epoch is not None else "not reached"


print()
print(f"Task 2: 1-3-1 network (3 tanh hidden nodes, linear output), "
      f"{N_EPOCHS} epochs, seed = {SEED}")
print(f"Training error with the given initial weights : {initial_error:.6f}")
print("Final training error (mean of (d - y)^2 over the 41 samples, after the last epoch):")
print(f"  SGD  (eta = {ETA})      : {sgd_errors[-1]:.6f}  ({sgd_errors[-1]:.3e})")
print(f"  ADAM (alpha = {ALPHA})  : {adam_errors[-1]:.6f}  ({adam_errors[-1]:.3e})")
print(f"First epoch at which the training error goes below {THRESHOLD}:")
print(f"  SGD  : {epoch_text(sgd_epoch)}")
print(f"  ADAM : {epoch_text(adam_epoch)}")


# %% Plots
FIG_DIR.mkdir(exist_ok=True)

# Colours: one fixed colour per optimiser, used in both figures; the data are neutral grey.
SGD_COLOUR = "#2a78d6"      # blue
ADAM_COLOUR = "#eb6834"     # orange
DATA_COLOUR = "#52514e"     # grey

# Network outputs on the 20 test inputs, with the trained weights of each run.
_, y_sgd = forward(xtest, *sgd_weights)
_, y_adam = forward(xtest, *adam_weights)

# Figure 1: the fit on xtest, with the training data.
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.scatter(X.ravel(), D.ravel(), s=22, color=DATA_COLOUR, zorder=3,
            label="Training data (41 samples)")
ax1.plot(xtest.ravel(), y_sgd.ravel(), "-o", color=SGD_COLOUR, linewidth=2, markersize=5,
         label="SGD: network output on xtest")
ax1.plot(xtest.ravel(), y_adam.ravel(), "--s", color=ADAM_COLOUR, linewidth=2, markersize=5,
         label="ADAM: network output on xtest")
ax1.set_xlabel("Input x")
ax1.set_ylabel("Output (d for data, y for network)")
ax1.set_title(f"1-3-1 network fit after {N_EPOCHS} epochs (Task2.py)")
ax1.grid(True, color="#e3e2de", linewidth=0.8)
ax1.legend()
fig1.tight_layout()
fig1.savefig(FIG_DIR / "Task2_fit.png", dpi=200)

# Figure 2: both learning curves on one figure, drawn twice with the same data:
#   left  - linear epoch axis: all 3000 epochs, shows that both curves come down and flatten;
#   right - log epoch axis: stretches out the first ~100 epochs, where SGD and ADAM differ most.
# The error axis is logarithmic in both panels, because the error falls by about three orders
# of magnitude. ADAM is drawn first and SGD on top, so both curves stay visible.
epochs = np.arange(1, N_EPOCHS + 1)
fig2, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax in (ax_lin, ax_log):
    ax.plot(epochs, adam_errors, color=ADAM_COLOUR, linewidth=1.0, label=f"ADAM (alpha = {ALPHA})")
    ax.plot(epochs, sgd_errors, color=SGD_COLOUR, linewidth=1.0, label=f"SGD (eta = {ETA})")
    ax.axhline(THRESHOLD, color=DATA_COLOUR, linestyle=":", linewidth=1.5,
               label=f"Threshold {THRESHOLD}")
    # Mark where each curve first goes below the threshold (if it does).
    for epoch, curve, colour, name in [(sgd_epoch, sgd_errors, SGD_COLOUR, "SGD"),
                                       (adam_epoch, adam_errors, ADAM_COLOUR, "ADAM")]:
        if epoch is not None:
            ax.plot(epoch, curve[epoch - 1], "o", color=colour, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=4,
                    label=f"{name} first below {THRESHOLD}: epoch {epoch}")
    ax.set_yscale("log")
    ax.grid(True, which="both", color="#e3e2de", linewidth=0.8)
ax_log.set_xscale("log")
ax_lin.set_xlabel("Epoch")
ax_log.set_xlabel("Epoch (log scale)")
ax_lin.set_ylabel("Training error, mean of (d - y)$^2$ (log scale)")
ax_lin.set_title("All epochs")
ax_log.set_title("Same curves, log epoch axis")
ax_lin.legend(loc="upper right")
fig2.suptitle("Learning curves of SGD and ADAM (Task2.py)")
fig2.tight_layout()
fig2.savefig(FIG_DIR / "Task2_learning_curves.png", dpi=200)

print(f"Figures saved in {FIG_DIR}")
plt.show()
