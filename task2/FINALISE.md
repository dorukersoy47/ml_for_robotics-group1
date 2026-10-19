# Task 2: how to finalise it

This guide takes Task 2 (the 1-3-1 MLP trained with SGD and ADAM) from "code written" to "ready to submit and defend at the viva". Work through it top to bottom.

| Item | Status | Who |
|---|---|---|
| `Task2.py` | Written and tested (results below) | done |
| `WhatIf_Task2.py` | Written; tested only to check that it runs. **Nobody runs it until the team has predicted** (Section 5) | done |
| `data/` | Copies of `RegressionData.csv` and `InitialWeights.npz`; the originals stay in `coursework/Assignment_Files_Draft37/` | done |
| `requirements.txt` | NumPy, pandas, Matplotlib | done |
| Virtual environment, first run, git | Sections 1 and 2 | **you** |
| Derivation appendix (handwritten is fine, at most 2 pages) | Section 4 | **you** |
| What-if analysis (one page in the report) | Section 5 | **whole team** |
| `documentation.md` (section-by-section walkthrough for the viva) | Written. Its What-if explanation (Section 6.1) is left until after the team meeting | done |

---

## 1. What is in `task2/`

```
task2/
├── Task2.py            the file you run at the viva (SGD + ADAM, prints and plots)
├── WhatIf_Task2.py     the What-if file: Task2.py with the tanh slope replaceable by 1
├── data/
│   ├── RegressionData.csv    41 samples, columns x and d
│   └── InitialWeights.npz    W1r (3x1), b1r (3x1), W2r (1x3), b2r (1x1)
├── figures/            created by the scripts the first time they run
├── requirements.txt
└── FINALISE.md         this file
```

Both scripts find `data/` and `figures/` relative to their own location, so they run from any folder (the brief asks for relative paths). The `.mat` weights are not copied because the team uses Python.

---

## 2. Set up the venv and run

From the repo root (`ml_for_robotics-group1/`), on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r task2/requirements.txt
python task2/Task2.py
```

Notes:
- `.venv` is already in the repo's `.gitignore`, so it will not be committed. `__pycache__/` is ignored too.
- `figures/` is created on the first run. Committing the PNGs is your choice: they can always be regenerated.
- The run takes about 5 to 20 seconds, depending on the laptop. The two `Training ...` lines show it has not hung. Close the figure windows to end the script.
- You can also run it cell by cell in VS Code (the `# %%` lines are cells). In that case the script uses the current working directory instead of the file's folder, which VS Code sets to the file's folder by default.

---

## 3. What you should see when you run `Task2.py`

Printed output (checked on NumPy 2.2.6 and 2.5.3, both identical):

```
Training with SGD for 3000 epochs (this takes a few seconds) ...
Training with ADAM for 3000 epochs ...

Task 2: 1-3-1 network (3 tanh hidden nodes, linear output), 3000 epochs, seed = 1
Training error with the given initial weights : 0.516987
Final training error (mean of (d - y)^2 over the 41 samples, after the last epoch):
  SGD  (eta = 0.1)      : 0.000475  (4.750e-04)
  ADAM (alpha = 0.005)  : 0.000372  (3.725e-04)
First epoch at which the training error goes below 0.001:
  SGD  : epoch 188
  ADAM : epoch 151
```

Two figure windows, also saved in `task2/figures/`:
1. **`Task2_fit.png`**: the 41 training points (grey), with the SGD output on `xtest` (blue, solid) and the ADAM output on `xtest` (orange, dashed). Both follow the S-shaped cubic trend closely.
2. **`Task2_learning_curves.png`**: the two learning curves, drawn twice. On the left, all 3000 epochs on a linear epoch axis: both come down and flatten. On the right, the same curves on a log epoch axis, which stretches out the first 100 epochs where SGD and ADAM differ. The dotted line is 0.001, and the dots mark the first epoch below it.

The assessor looks at three things, and our output answers each:

| Assessor checks | Where it is |
|---|---|
| Does the fit follow the data? | Figure 1 |
| Do both learning curves come down and flatten? | Figure 2, left panel |
| The printed final error, plus the epoch below 0.001 or "not reached" | The printed block |

### Understand these results (likely viva material)

1. **Why the error settles at about 3–5 × 10⁻⁴ and not at 0.** The data are a cubic plus Gaussian noise with standard deviation 0.02, so the noise variance is 0.02² = 4 × 10⁻⁴. No smooth curve can predict random noise. As a benchmark I fitted the best possible cubic by least squares, outside the submission code. It leaves an error of 3.08 × 10⁻⁴. The network's typical late error (median over the last 500 epochs: SGD 3.2 × 10⁻⁴, ADAM 4.0 × 10⁻⁴) is at that floor. The network has learnt the underlying curve, and what is left is noise. Pushing much lower would mean fitting the noise, which is overfitting (Week 4).
2. **Why the curves are jagged.** We update after every single sample (sequential, or stochastic, mode, Week 3). Each update reduces the error on one sample and slightly disturbs the others. The error over all 41 samples therefore jitters from epoch to epoch instead of falling smoothly, as it would with batch updates.
3. **Why ADAM is jaggier near the end.** SGD's step is η × gradient, so its steps shrink as the errors shrink. ADAM's step for each weight is about α · m̂ / √v̂. That ratio is a *normalised* gradient, so the step stays around α even when the gradient is tiny. Near the minimum, ADAM keeps moving the weights by up to about 0.005 per update and jitters around the floor.
4. **Why ADAM starts slower.** After epoch 1, the error is 0.234 for ADAM against 0.035 for SGD (both start from 0.517). At the start the errors, and so the gradients, are large. SGD's steps scale with them and are large. ADAM's steps are limited to roughly α = 0.005 per update whatever the gradient size, so in 41 updates a weight can move only about 0.2.
5. **Why ADAM goes below 0.001 at a different epoch (151 against 188).** Later on the gradients are small. SGD's steps shrink with them, while ADAM's normalisation keeps its steps near α. On the right panel, ADAM's curve falls more steeply between epochs 40 and 150, and it overtakes SGD. The ordering is not luck: in scratch runs with seeds 1–4, ADAM first went below 0.001 at epochs 143–155 and SGD at 184–194. It is a close race, though. At epoch 150 the errors are 0.00141 (SGD) and 0.00139 (ADAM). Both curves then hover around 0.001 for a while: after first going below it, SGD is back above it for 46 of the next ~210 epochs, and ADAM for 35 of the next ~250. So the *exact* "first epoch below" also depends on the jitter.
6. **Do not over-read the final numbers.** SGD's last epoch (4.75 × 10⁻⁴) happens to be above its typical level (3.2 × 10⁻⁴), and ADAM's (3.7 × 10⁻⁴) slightly below its typical level (4.0 × 10⁻⁴). Which optimiser "ends lower" depends on where the jitter happens to be at epoch 3000. The honest statement is that both reach the noise floor.

---

## 4. The derivation appendix (you write this by hand)

The brief asks you to derive backpropagation for the 1-3-1 network with E = ½(d − y)²:
- Show the **full calculation for one weight in each layer**.
- Then give the **general formula** for the other weights of that layer.
- Put it in an appendix: at most two pages, and a scan is fine.

A viva question may ask *"which equation of your derivation is this line?"*. So **number your equations exactly as below**: the code comments use these numbers (`# Eq. (9)` etc.).

The notation follows the Week 3 slides:
- Layer (1) is the hidden layer and layer (2) the output layer.
- `w_ij^(s)` is the weight from node i of the previous layer to node j of layer s. The bias is `w_0j^(s)`, with a constant input of 1.
- `v` is an induced local field (the adder output), and `x_out` a node output.

### 4.1 Equation sheet

| Eq. | Meaning | Formula |
|---|---|---|
| (1) | Hidden adder, j = 1, 2, 3 | v_j^(1) = w_1j^(1) · x + w_0j^(1) |
| (2) | Hidden output | x_out,j^(1) = φ^(1)(v_j^(1)) = tanh(v_j^(1)) |
| (3) | Output adder | v^(2) = Σ_j w_j1^(2) · x_out,j^(1) + w_01^(2) |
| (4) | Output (linear) | y = φ^(2)(v^(2)) = v^(2) |
| (5) | Error and loss | e = d − y,  E = ½ e² |
| (6) | Output local gradient | δ^(2) = (d − y) · φ^(2)′(v^(2)) = e · 1 = e |
| (7) | Output-layer gradients | ∂E/∂w_j1^(2) = −δ^(2) · x_out,j^(1),  ∂E/∂w_01^(2) = −δ^(2) |
| (8) | Slope of tanh | φ^(1)′(v_j^(1)) = 1 − tanh²(v_j^(1)) = 1 − (x_out,j^(1))² |
| (9) | Hidden local gradient | δ_j^(1) = δ^(2) · w_j1^(2) · (1 − (x_out,j^(1))²) |
| (10) | Hidden-layer gradients | ∂E/∂w_1j^(1) = −δ_j^(1) · x,  ∂E/∂w_0j^(1) = −δ_j^(1) |
| (11) | SGD update (every weight) | w_new = w_old − η · ∂E/∂w = w_old + η · δ · x_out |
| (12) | Training error | (1/41) Σ_k (d_k − y_k)² |
| (13) | ADAM gradient average | m(t) = β₁ m(t−1) + (1 − β₁) g, where g = ∂E/∂w |
| (14) | ADAM squared-gradient average | v(t) = β₂ v(t−1) + (1 − β₂) g² |
| (15) | ADAM bias correction | m̂ = m(t) / (1 − β₁ᵗ),  v̂ = v(t) / (1 − β₂ᵗ) |
| (16) | ADAM update | w ← w − α · m̂ / √(v̂ + ε) |

Watch out for two things:
- **Two different v's.** The `v` in (1), (3) is an induced local field. The `v` in (14)–(16) is ADAM's squared-gradient average. The lecture uses the same letter for both. In the code they are `v1`, `v2` (fields) and `v_W1`, `v_b1`, ... (ADAM).
- **The time index.** The slides write ADAM as m(t+1) with t starting at 0. The code adds 1 to t first and then uses β^t. That is the same thing, written one step shifted.

Equations (13)–(16) are not part of the backpropagation derivation the brief asks for. Include them only if you want the appendix to cover ADAM too.

### 4.2 Output layer: full calculation for w_21^(2) (hidden node 2 → output)

Write E in terms of the weight:

E = ½ (d − y)²,  y = v^(2),  v^(2) = w_11^(2) x_out,1^(1) + **w_21^(2)** x_out,2^(1) + w_31^(2) x_out,3^(1) + w_01^(2)

Chain rule:

∂E/∂w_21^(2) = ∂E/∂y · ∂y/∂v^(2) · ∂v^(2)/∂w_21^(2)

- ∂E/∂y = −(d − y)
- ∂y/∂v^(2) = 1, because the output is linear
- ∂v^(2)/∂w_21^(2) = x_out,2^(1)

⇒ ∂E/∂w_21^(2) = −(d − y) · 1 · x_out,2^(1) = −δ^(2) · x_out,2^(1), with δ^(2) = (d − y) · 1 (Eq. 6).

Steepest descent: w_21,new^(2) = w_21,old^(2) − η ∂E/∂w_21^(2) = w_21,old^(2) + η · δ^(2) · x_out,2^(1)

**General formula for this layer:**
- w_j1,new^(2) = w_j1,old^(2) + η · δ^(2) · x_out,j^(1), for j = 1, 2, 3.
- Bias (input 1): w_01,new^(2) = w_01,old^(2) + η · δ^(2).

In the code, `W2[0, 1]` is w_21^(2). The line `dE_dW2 = -delta2 @ a1.T` computes all three weight gradients at once.

### 4.3 Hidden layer: full calculation for w_12^(1) (input x → hidden node 2)

Follow the weight forward through the network:
- w_12^(1) appears only in v_2^(1) = **w_12^(1)** x + w_02^(1).
- v_2^(1) appears only in x_out,2^(1) = tanh(v_2^(1)).
- x_out,2^(1) appears only in v^(2), since there is a single output node. The sum over output nodes in the lecture's general hidden-layer formula therefore has one term here.
- v^(2) gives y, and y gives E.

Chain rule:

∂E/∂w_12^(1) = ∂E/∂y · ∂y/∂v^(2) · ∂v^(2)/∂x_out,2^(1) · ∂x_out,2^(1)/∂v_2^(1) · ∂v_2^(1)/∂w_12^(1)

- ∂E/∂y = −(d − y)
- ∂y/∂v^(2) = 1
- ∂v^(2)/∂x_out,2^(1) = w_21^(2)
- ∂x_out,2^(1)/∂v_2^(1) = 1 − tanh²(v_2^(1)) = 1 − (x_out,2^(1))². This is Eq. 8, derived by the quotient rule on Week 2 slide 89.
- ∂v_2^(1)/∂w_12^(1) = x

⇒ ∂E/∂w_12^(1) = −(d − y) · 1 · w_21^(2) · (1 − (x_out,2^(1))²) · x = −δ_2^(1) · x,

with δ_2^(1) = δ^(2) · w_21^(2) · (1 − (x_out,2^(1))²) (Eq. 9).

Steepest descent: w_12,new^(1) = w_12,old^(1) + η · δ_2^(1) · x

**General formula for this layer:**
- δ_j^(1) = δ^(2) · w_j1^(2) · (1 − (x_out,j^(1))²)
- w_1j,new^(1) = w_1j,old^(1) + η · δ_j^(1) · x
- Bias: w_0j,new^(1) = w_0j,old^(1) + η · δ_j^(1), for j = 1, 2, 3

Add one sentence to the appendix: **δ^(1) uses the *old* w_j1^(2)**, the value that produced y. In the code, `backward()` computes every gradient before any weight is changed.

### 4.4 Which line implements which equation

| Eq. | Line in the code | Function |
|---|---|---|
| (1) | `v1 = W1 @ x + b1` | `forward` |
| (2) | `a1 = np.tanh(v1)` | `forward` |
| (3) | `v2 = W2 @ a1 + b2` | `forward` |
| (4) | `y = v2` | `forward` |
| (5) | `e = d - y` | `backward` |
| (6) | `delta2 = e * phi2_slope` | `backward` |
| (7) | `dE_dW2 = -delta2 @ a1.T`, `dE_db2 = -delta2` | `backward` |
| (8) | `tanh_slope = 1 - a1 ** 2` | `backward` (the What-if line) |
| (9) | `delta1 = (W2.T @ delta2) * tanh_slope` | `backward` |
| (10) | `dE_dW1 = -delta1 @ x.T`, `dE_db1 = -delta1` | `backward` |
| (11) | `W2 = W2 - ETA * dE_dW2` (and the three lines below it) | `train_sgd` |
| (12) | `np.mean((D - Y) ** 2)` | `training_error` |
| (13) | `m = BETA1 * m + (1 - BETA1) * grad` | `adam_update` |
| (14) | `v = BETA2 * v + (1 - BETA2) * grad ** 2` | `adam_update` |
| (15) | `m_hat = ...`, `v_hat = ...` | `adam_update` |
| (16) | `w = w - ALPHA * m_hat / np.sqrt(v_hat + EPS)` | `adam_update` |

---

## 5. What-if analysis: the forgotten derivative (whole team, 2 marks)

**Do not run `WhatIf_Task2.py` and do not open `figures/WhatIf_Task2.png` until Step 1 is done.** The PNG does not exist yet; it is created the first time the file runs. I ran the file once in my own workspace, only to check that it works and that its normal runs reproduce `Task2.py`. None of its results are in this guide or in the repo.

What the file does: it is `Task2.py` with one change in `backward()`. When `forget_derivative=True`, the factor `1 - a1 ** 2` (Eq. 8) is replaced by 1. It runs four trainings with the same seed:
- SGD, normal
- SGD, derivative forgotten
- ADAM, normal
- ADAM, derivative forgotten

It prints a table (final training error and first epoch below 0.001 for each run, plus the ratio forgotten / normal). It also saves one figure with two panels: the fit on xtest on the left and the learning curves on the right.

### Step 1: Predict (team meeting, before anyone runs it)

Every member writes their own prediction **with a reason**. The predictions do not have to agree, and a wrong one costs nothing. An empty reason ("it will get worse") costs a mark.

| Member | (1) What will the fit look like? Why? | (2) How will the final training error compare with the normal run? Why? |
|---|---|---|
| Name 1 | | |
| Name 2 | | |
| Name 3 | | |
| Name 4 | | |

Useful facts for thinking (these are about the method, not the result):
- From Section 3, the normal final errors are about 4.7 × 10⁻⁴ (SGD) and 3.7 × 10⁻⁴ (ADAM).
- The output-layer equations (6)–(7) do not use the factor at all.
- 1 − a² lies between 0 and 1. It is 1 when a = 0 and close to 0 when the node is saturated (a ≈ ±1).

### Step 2: Run (together)

```bash
python task2/WhatIf_Task2.py
```

Put three things in the report:
- The file name `WhatIf_Task2.py`.
- The numbers from the printed table.
- The figure `figures/WhatIf_Task2.png`, with a caption that names the file. For example: *"Figure 2: Fit on xtest (left) and learning curves (right) for SGD and ADAM, normal (solid) and with the tanh slope 1 − a² replaced by 1 (dashed). Produced by WhatIf_Task2.py."*

The figure already has axis labels and a legend, which the brief requires.

### Step 3: Explain (only after running; don't read this before the prediction meeting)

Discuss these questions and agree on one explanation. Then say why each wrong prediction was wrong.
- With the factor set to 1, is −δ_j^(1) · x still the derivative of E with respect to w_1j^(1)? If not, what are the hidden weights now following?
- Which weights still receive their correct gradient?
- What is the factor 1 − a² normally doing for a hidden node whose output is near ±1? What happens without it?
- Do SGD and ADAM react the same way? Remember that ADAM's step size depends on how consistent the gradient's sign is, not on its size.

Send me the printed table and the figure after the meeting, and we'll write the agreed explanation together.

### Write-up (at most one page, scored out of 2)

Write it in this order:
1. **Predict**: one line per member, with name, prediction and reason.
2. **Run**: the file name, the numbers and the figure.
3. **Explain**: the agreed explanation, and why each wrong prediction was wrong.

For 2 marks you need: every member's prediction with a reason, a result we can reproduce from the separate file, and a correct explanation that addresses the wrong predictions. It must fit within one page, with captions, axis labels and legends on the figures. Use Arial 11 pt and single spacing, as for the whole report.

---

## 6. Code map and AI line

Both files start with the required code map. Its names are the same as the `# %%` section headings:
- `Task2.py`: Settings, Load data, Network, Training error, Train SGD, Train ADAM, Results, Plots.
- `WhatIf_Task2.py`: the same, plus Runs.

The last line is `# AI use          Claude Opus 5.5 (claude-opus-5-5), High`. If you change the code yourself, keep the map in step with the section names.

---

## 7. Viva: Task 2 (about 6 minutes: run the file, then one question per student)

**Before the viva:**
- Activate the venv, open `Task2.py` (and `WhatIf_Task2.py`), and run `Task2.py` once to check it works on that laptop.
- Every member must be able to answer questions about any line. `documentation.md`, written next, will go through the code section by section.
- No AI or lookup is allowed during the viva.

**Quick answers to the most likely questions** (expand on these in `documentation.md`):

| Question | Short answer |
|---|---|
| What does `delta1` hold, and where is it used? | The local gradients δ_j^(1) of the 3 hidden nodes, shape (3, 1), from Eq. 9. It is used straight away for the hidden-layer gradients `dE_dW1` and `dE_db1` (Eq. 10). |
| Why is the output delta just `e`? | The output node is linear, so φ′ = 1 and δ^(2) = e · 1 (Eq. 6). |
| Where is the tanh derivative, and why is it `1 - a1**2`? | `tanh_slope`, Eq. 8. d tanh(v)/dv = 1 − tanh²(v), and tanh(v) is already stored as `a1`. |
| What goes into `backward()`, and what comes out? | In: one sample `x`, `d`, the `a1` and `y` from `forward()`, and the current `W2`. Out: the four gradient arrays of E = ½(d − y)². |
| Why are the gradients negative, while the lecture adds η δ x? | ∂E/∂w = −δ · x_out, so w − η ∂E/∂w = w + η δ x_out (Eq. 11). It is the same update. We keep the gradient form because ADAM needs ∂E/∂w. |
| Why `.copy()` of the initial weights? | So that training can never change `W1_INIT` etc. Our update lines (`W = W - ...`) make new arrays anyway. But if anyone wrote an in-place `W -= ...` without the copy, the given weights would be overwritten, and ADAM would start from SGD's trained weights. |
| Why set the seed at the start of every run? | So that SGD, ADAM and the What-if file shuffle the samples in exactly the same order. A run then gives the same result in any file, as the brief requires. |
| What is `t` at the end of ADAM training? | 3000 × 41 = 123 000. It counts updates (one per sample), not epochs. |
| Why the bias correction in ADAM? | m and v start at 0, so their early averages are too small. Dividing by 1 − βᵗ scales them up. Without it, the first steps would be wrongly sized: v is far too small, so the division by √v blows up the step. |
| Why is the training error not halved? | The brief defines it as the plain mean of (d − y)². The ½ is only in the per-sample loss E, where it cancels the 2 from differentiating. |
| Why is ADAM's curve noisier near the end? | ADAM's normalised step stays about α even when gradients are tiny. SGD's step shrinks with the gradient (Section 3, point 3). |
| Why does ADAM go below 0.001 at a different epoch from SGD? | Section 3, points 4–5: ADAM is slower at first (steps limited to about α) and faster later (steps don't shrink), and the exact epoch also depends on the jitter around 0.001. |
| Why does the error not go to 0? | The noise variance is about 4 × 10⁻⁴, and the best possible cubic already leaves 3.1 × 10⁻⁴ (Section 3, point 1). |
| Why `X[:, [k]]` and not `X[:, k]`? | `[k]` keeps the result 2-D, with shape (1, 1). The matrix products in `forward` and `backward` then have consistent shapes. |

---

## 8. Checks already done

- **Gradient check.** `backward()` was compared with central finite differences, (E(w + h) − E(w − h)) / 2h with h = 10⁻⁶, for all 10 weights at a random point. The largest difference was 2 × 10⁻¹⁰, so the backpropagation equations are implemented correctly.
- **Reproducibility.** `Task2.py` gives identical output on NumPy 2.2.6 and 2.5.3. The two normal runs in `WhatIf_Task2.py` print exactly the same numbers as `Task2.py`.
- **Relative paths.** Both scripts were run from a different working directory and still found `data/`.

---

## 9. Before submission (Task 2 part)

- [ ] `Task2.py` and `WhatIf_Task2.py` run from a fresh terminal in the venv and print the numbers in the report.
- [ ] `data/` is inside `task2/`, so the zip runs on the assessor's machine with no path changes.
- [ ] The derivation appendix uses the equation numbers of Section 4 and fits in at most 2 pages (it does not count towards the 6-page limit).
- [ ] The What-if page has every member's prediction with a reason, the file name, the numbers and figure, and the agreed explanation, all within one page.
- [ ] Every figure caption in the report names the code file that produced it.
- [ ] Any code or text taken unchanged from an AI tool is credited in the references (IEEE style), as the brief requires.
- [ ] Git (your side): commit `task2/` (both `.py` files, `data/`, `requirements.txt`, `FINALISE.md`). Committing `figures/` is optional, and `.venv` is ignored automatically.
