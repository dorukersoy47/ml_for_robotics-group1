# CODE MAP of Task1.py
# Settings          darkFactor, mirror, image file, colour rule of each fruit
# Load image        read Fruits.jpg, darken it, mirror it
# Colour            convert every pixel from RGB to hue, saturation, value (HSV)
# Mask              keep the pixels whose colour fits one fruit's rule
# Clean             majority (binary median) filter, then keep the largest blob
# Box and centroids bounding box, box-midpoint centroid, moment centroid
# Find fruits       Colour -> Mask -> Clean -> Box and centroids, for each fruit
# Run               print both centroids and their distance, draw boxes and centroids
# AI use            Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Run:  python Task1.py   (Fruits.jpg in the same folder; needs numpy, matplotlib)

# %% Settings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

darkFactor = 0.8
mirror = True
image_file = "Fruits.jpg"

# hue in degrees (0-360); sat and val from 0 to 1; (320, 6) means 320..360 OR 0..6
FRUIT_RULES = {
    "apple":    {"hue": (320, 6), "sat_min": 0.40, "val_min": 0.60, "colour": "cyan"},
    "capsicum": {"hue": (5, 35),  "sat_min": 0.90, "val_min": 0.90, "colour": "lime"},
    "lemon":    {"hue": (38, 62), "sat_min": 0.45, "val_min": 0.75, "colour": "magenta"},
}


# %% Load image
def load_image(file_name, dark_factor, flip):
    img = plt.imread(file_name)[:, :, :3].astype(float)
    img = np.clip(np.round(img * dark_factor), 0, 255).astype(np.uint8)
    if flip:
        img = np.fliplr(img)
    return img


# %% Colour
def rgb_to_hsv(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    val = img.max(axis=2)
    delta = val - img.min(axis=2)
    d = np.where(delta == 0, 1, delta)
    hue = 60 * (R - G) / d + 240
    hue = np.where(val == G, 60 * (B - R) / d + 120, hue)
    hue = np.where(val == R, (60 * (G - B) / d) % 360, hue)
    sat = delta / np.where(val == 0, 1, val)
    return hue, sat, val


# %% Mask
def colour_mask(hue, sat, val, rule):
    low, high = rule["hue"]
    if low <= high:
        hue_ok = (hue >= low) & (hue <= high)
    else:
        hue_ok = (hue >= low) | (hue <= high)
    return hue_ok & (sat >= rule["sat_min"]) & (val >= rule["val_min"])


# %% Clean
def majority_filter(mask, size):
    rows, cols = mask.shape
    padded = np.pad(mask.astype(float), size // 2)
    total = np.zeros((rows, cols))
    for dy in range(size):
        for dx in range(size):
            total += padded[dy:dy + rows, dx:dx + cols]
    return total > size * size / 2


def largest_blob(mask):
    rows, cols = mask.shape
    visited = np.zeros((rows, cols), dtype=bool)
    best = []
    for y0, x0 in np.argwhere(mask):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        blob = []
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            blob.append((y, x))
            for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(blob) > len(best):
            best = blob
    result = np.zeros((rows, cols), dtype=bool)
    for y, x in best:
        result[y, x] = True
    return result


# %% Box and centroids
def box_and_centroids(blob):
    ys, xs = np.nonzero(blob)
    if len(xs) == 0:
        return None
    x_min, x_max, y_min, y_max = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    box_c = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    m00, m10, m01 = len(xs), xs.sum(), ys.sum()
    mom_c = (m10 / m00, m01 / m00)
    distance = np.hypot(box_c[0] - mom_c[0], box_c[1] - mom_c[1])
    return {"box": (x_min, y_min, x_max, y_max), "box_centroid": box_c,
            "moment_centroid": mom_c, "distance": distance, "area": m00}


# %% Find fruits
def find_fruits(img, normalise=True):
    scale = img.max() if normalise else 255
    hue, sat, val = rgb_to_hsv(img / scale)
    size = 2 * round(img.shape[1] / 150) + 1
    results = {}
    for name, rule in FRUIT_RULES.items():
        mask = majority_filter(colour_mask(hue, sat, val, rule), size)
        results[name] = box_and_centroids(largest_blob(mask))
    return results


# %% Run
if __name__ == "__main__":
    img = load_image(image_file, darkFactor, mirror)
    results = find_fruits(img)

    print(f"darkFactor = {darkFactor}, mirror = {mirror}, image size = {img.shape[1]} x {img.shape[0]}")
    print("(x, y) in pixels: x = column, y = row, (0, 0) = top-left corner")
    plt.figure(figsize=(9, 7))
    plt.imshow(img)
    for name, r in results.items():
        if r is None:
            print(f"{name:9s}: NOT FOUND")
            continue
        (bx, by), (mx, my) = r["box_centroid"], r["moment_centroid"]
        print(f"{name:9s}: box centroid = ({bx:6.1f}, {by:6.1f})   "
              f"moment centroid = ({mx:6.1f}, {my:6.1f})   distance = {r['distance']:5.2f} px")
        colour = FRUIT_RULES[name]["colour"]
        x_min, y_min, x_max, y_max = r["box"]
        plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      fill=False, edgecolor=colour, linewidth=2))
        plt.text(x_min, y_min - 5, name, color=colour, fontweight="bold")
        plt.plot(bx, by, "+", color=colour, markersize=14, markeredgewidth=2.5)
        plt.plot(mx, my, "x", color=colour, markersize=10, markeredgewidth=2.5)
    plt.plot([], [], "k+", label="a) bounding-box midpoint")
    plt.plot([], [], "kx", label="b) moment centroid")
    plt.legend(loc="lower right")
    plt.title(f"Task 1: darkFactor = {darkFactor}, mirror = {mirror}")
    plt.xlabel("x (pixel column)")
    plt.ylabel("y (pixel row)")
    plt.tight_layout()
    plt.savefig("Task1_result.png", dpi=150)
    plt.show()
