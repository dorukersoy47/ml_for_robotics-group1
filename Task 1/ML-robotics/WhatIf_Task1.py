# CODE MAP of WhatIf_Task1.py
# Settings      dark factors to test (1.0 down to 0.1, step 0.05), mirror
# Reference     run Task1 at darkFactor 1.0; its boxes are "what is visible" of each fruit
# Sweep         run Task1 unchanged at every dark factor; measure how much of the
#               reference box the new box covers; lost = no box or cover < 0.5
#               (second sweep with brightness normalisation off, for the explanation only)
# Print         table of cover fractions and the dark factor where each fruit is first lost
# Plots         cover fraction against dark factor; boxes at a few dark factors
# AI use        Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Run:  python WhatIf_Task1.py      (Task1.py and Fruits.jpg in the same folder)
# The functions and colour rules are imported from Task1.py, so the code and
# settings are exactly the ones submitted; nothing is retuned per dark factor.

# %% Settings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Task1 import load_image, find_fruits, FRUIT_RULES, image_file

dark_factors = np.round(np.arange(1.0, 0.099, -0.05), 2)   # 1.0, 0.95, ..., 0.1  (19 values)
mirror = False             # mirroring does not change any colour, so it cannot change the result
fruits = list(FRUIT_RULES.keys())


# %% Reference
def box_area(box):
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min + 1) * (y_max - y_min + 1)


def overlap_area(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right < x_left or y_bottom < y_top:
        return 0                                          # the boxes do not touch
    return (x_right - x_left + 1) * (y_bottom - y_top + 1)


reference = find_fruits(load_image(image_file, 1.0, mirror))   # checked by eye: one box per fruit


# %% Sweep
def sweep(normalise):
    cover = {name: [] for name in fruits}         # fraction of the reference box covered, per dark factor
    size_ratio = {name: [] for name in fruits}    # new box area / reference box area
    all_results = {}
    for k in dark_factors:
        results = find_fruits(load_image(image_file, k, mirror), normalise)
        all_results[k] = results
        for name in fruits:
            r = results[name]
            if r is None:                                 # no box at all -> lost
                cover[name].append(0.0)
                size_ratio[name].append(0.0)
            else:
                ref_box = reference[name]["box"]
                cover[name].append(overlap_area(r["box"], ref_box) / box_area(ref_box))
                size_ratio[name].append(box_area(r["box"]) / box_area(ref_box))
    return cover, size_ratio, all_results


# The What-if result: Task1 exactly as submitted.
cover, size_ratio, all_results = sweep(normalise=True)
# For the explanation only: the same code with the brightness normalisation switched off.
cover_raw, _, _ = sweep(normalise=False)


# %% Print
print("Cover = part of the visible fruit (reference box at darkFactor 1.0) inside the new box.")
print("Size  = new box area / reference box area (a value far above 1 means the box grew onto other things).")
print("darkFactor | " + " | ".join(f"{n:>9s} cover  size" for n in fruits))
for i, k in enumerate(dark_factors):
    row = " | ".join(f"{cover[n][i]:15.2f} {size_ratio[n][i]:5.2f}" for n in fruits)
    print(f"{k:10.2f} | {row}")

def first_lost(cover_list):
    lost = [k for k, c in zip(dark_factors, cover_list) if c < 0.5]
    if len(lost) == 0:
        return "never lost (still found at 0.10)"
    return f"first lost at darkFactor = {lost[0]:.2f}"


print()
print("RESULT (Task1.py as submitted):")
for name in fruits:
    print(f"  {name:9s}: {first_lost(cover[name])}")
print("Blob area in pixels (darkFactor 1.0 -> 0.1): " + ", ".join(
    f"{n} {reference[n]['area']} -> {all_results[0.1][n]['area'] if all_results[0.1][n] else 0}" for n in fruits))
print("FOR THE EXPLANATION ONLY (same code, brightness normalisation switched off):")
for name in fruits:
    print(f"  {name:9s}: {first_lost(cover_raw[name])}")


# %% Plots
plt.figure(figsize=(8, 4.8))
markers = {"apple": "o", "capsicum": "s", "lemon": "^"}
for name in fruits:
    plt.plot(dark_factors, cover[name], marker=markers[name], color=FRUIT_RULES[name]["colour"],
             markeredgecolor="k", label=f"{name} (as submitted)")
for name in fruits:
    plt.plot(dark_factors, cover_raw[name], linestyle=":", marker=markers[name], markersize=4,
             color=FRUIT_RULES[name]["colour"], markeredgecolor="k", alpha=0.8,
             label=f"{name} (no normalisation)")
plt.axhline(0.5, color="k", linestyle="--", linewidth=1, label="lost below 0.5")
plt.gca().invert_xaxis()                                  # read left to right: getting darker
plt.xlabel("dark factor (darkFactor)")
plt.ylabel("fraction of visible fruit inside box")
plt.title("WhatIf_Task1.py: darkening the image")
plt.ylim(-0.05, 1.1)
plt.grid(alpha=0.3)
plt.legend(loc="center left", fontsize=8, bbox_to_anchor=(1.01, 0.5))
plt.tight_layout()
plt.savefig("WhatIf_Task1_cover.png", dpi=150)

show_factors = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for ax, k in zip(axes.flat, show_factors):
    img = load_image(image_file, k, mirror)
    ax.imshow(img)
    for name, r in all_results[k].items():
        if r is None:
            continue
        x_min, y_min, x_max, y_max = r["box"]
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               fill=False, edgecolor=FRUIT_RULES[name]["colour"], linewidth=2))
    ax.set_title(f"darkFactor = {k}")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
plt.suptitle("WhatIf_Task1.py: boxes found (cyan apple, green capsicum, magenta lemon)")
plt.tight_layout()
plt.savefig("WhatIf_Task1_boxes.png", dpi=120)
plt.show()
