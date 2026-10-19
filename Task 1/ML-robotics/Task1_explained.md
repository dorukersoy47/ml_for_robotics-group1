# Task1.py explained, line by line

Line numbers below (**L15**, **L46**, …) match `Task1.py` exactly. If you change the file, the numbers move.

**Coordinates.** `x` = pixel **column** (0 = left edge), `y` = pixel **row** (0 = top edge). An image array is indexed `img[y, x]`: row first, then column. `Fruits.jpg` is 1024 columns × 783 rows.

---

## The whole method in 6 sentences (memorise this first)

1. **Load** the image, multiply every pixel by `darkFactor`, round to 8 bits, and mirror it if `mirror` is True.
2. **Undo the darkening**: divide by the brightest pixel, so the image always goes from 0 to 1.
3. **Convert RGB → HSV**: hue = *which* colour, saturation = *how strong*, value = *how bright*. Hue and saturation are ratios, so darkening doesn't change them.
4. **Mask**: for each fruit, keep the pixels whose hue, saturation and value are inside that fruit's rule.
5. **Clean**: a majority (binary median) filter removes specks and thin lines, then a flood fill keeps only the **largest connected blob**.
6. **Measure**: bounding box = min/max x and y of the blob; centroid a) = middle of the box; centroid b) = moment method (m10/m00, m01/m00); then print both centroids and their distance and draw them.

---

## L1–L12 · Code map

```python
# CODE MAP of Task1.py
# Settings          darkFactor, mirror, image file, colour rule of each fruit
...
# AI use            Claude Opus 5.5 (claude-opus-5-5), default setting
# Run:  python Task1.py   (Fruits.jpg in the same folder; needs numpy, matplotlib)
```
The assignment requires it. Each line names one `# %%` section of the file and says what it does. The last map line names the AI model. The run command and libraries are the "documented commands" the assignment asks for.

---

## L14–L28 · `# %% Settings`

**L14** `# %% Settings`
A section heading. `# %%` makes VS Code treat everything up to the next `# %%` as one "cell". The code map uses the same names.

**L15** `import numpy as np`
NumPy: arrays and maths on the whole image at once, without a loop over every pixel.

**L16** `import matplotlib.pyplot as plt`
Matplotlib: `plt.imread` reads the JPG; `imshow`, `plot` and `text` draw the result.

**L17** `from matplotlib.patches import Rectangle`
The rectangle shape used to draw a bounding box.

**L19** `darkFactor = 0.8`
Every pixel is multiplied by this. The task says the code must work from 0.6 to 1.0. The viva uses 0.8.

**L20** `mirror = True`
If True, the image is flipped left-to-right before searching.

**L21** `image_file = "Fruits.jpg"`
Relative path: the image must sit in the same folder as the code.

**L23** comment: hue is in degrees (0–360), saturation and value go from 0 to 1, and `(320, 6)` means "320 to 360 **or** 0 to 6".

**L24–L28** `FRUIT_RULES = { ... }`
A dictionary with one rule per fruit. A pixel belongs to a fruit if:
- its hue is inside `"hue"`, **and**
- its saturation ≥ `"sat_min"`, **and**
- its value (brightness) ≥ `"val_min"`.

`"colour"` is only the colour used to draw that fruit's box.

**How the numbers were chosen.** I measured H, S, V of each fruit and of the things most likely to be confused with it, then put each limit in the gap between them:

| Fruit | Rule | What it must avoid | Why this limit |
|---|---|---|---|
| Apple | hue 320–360 or 0–6, S ≥ 0.40, V ≥ 0.60 | dark-red wall shadow and table (hue 8–13°, V 0.3–0.66) | upper hue limit 6° and V ≥ 0.60 keep the shadow out |
| Capsicum | hue 5–35, S ≥ 0.90, V ≥ 0.90 | the wooden wall: **same hue** (25–32°) but S ≈ 0.87 and V ≈ 0.85 | the capsicum is more saturated (≈0.99) and brighter (≈0.98), so S ≥ 0.90 and V ≥ 0.90 separate them |
| Lemon | hue 38–62, S ≥ 0.45, V ≥ 0.75 | the wall next to it (hue ≈ 29–35°), the banana (same hue but S ≈ 0.39), the lime (hue ≈ 100°) | hue starts at 38° (above the wall), S ≥ 0.45 (above the banana), hue stops at 62° (below the lime) |

Red is at **both ends** of the hue circle (350° and 3° are both red). That's why the apple's range "wraps round" zero.

---

## L31–L37 · `# %% Load image`

**L32** `def load_image(file_name, dark_factor, flip):`
**In:** file name, dark factor, mirror flag. **Out:** the image as an array of shape (rows, columns, 3) with whole numbers 0–255.

**L33** `img = plt.imread(file_name)[:, :, :3].astype(float)`
- `plt.imread` reads the JPG into an array (783 × 1024 × 3), values 0–255.
- `[:, :, :3]` keeps only R, G, B (it drops a 4th "alpha" layer if a file has one).
- `.astype(float)` turns the values into decimals so multiplying by 0.8 isn't cut off.

**L34** `img = np.clip(np.round(img * dark_factor), 0, 255).astype(np.uint8)`
The darkening exactly as the task defines it, read from the inside out:
1. `img * dark_factor` multiplies every pixel;
2. `np.round` rounds to whole numbers;
3. `np.clip(…, 0, 255)` keeps the values inside the 8-bit range;
4. `.astype(np.uint8)` stores them as 8-bit integers ("rounded to 8 bits").

**L35–L36** `if flip: img = np.fliplr(img)`
`fliplr` = "flip left-right": the columns are reversed, so a pixel at column x moves to column (1023 − x). Colours don't change.

**L37** `return img`

---

## L40–L50 · `# %% Colour`: RGB → HSV, written from scratch

**L41** `def rgb_to_hsv(img):`
**In:** image with values 0–1. **Out:** three arrays the size of the image: `hue` (0–360°), `sat` (0–1), `val` (0–1).

**L42** `R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]`
Splits the image into its red, green and blue layers (each 783 × 1024).

**L43** `val = img.max(axis=2)`
**Value** = the largest of R, G, B at each pixel = brightness. `axis=2` means "across the 3 colour layers".

**L44** `delta = val - img.min(axis=2)`
Δ = max − min of R, G, B: how much the strongest and weakest channel differ. Δ = 0 means grey (no colour).

**L45** `d = np.where(delta == 0, 1, delta)`
A copy of Δ with 0 replaced by 1, **only to avoid dividing by zero** on grey pixels. `np.where(condition, a, b)` = "a where the condition is True, otherwise b".

**L46** `hue = 60 * (R - G) / d + 240`
Start by assuming **blue** is the largest channel. Blue sits at 240° on the hue circle, and the fraction (R − G)/Δ moves the hue a bit towards red or green.

**L47** `hue = np.where(val == G, 60 * (B - R) / d + 120, hue)`
Where **green** is the largest channel, use the green formula instead (green sits at 120°).

**L48** `hue = np.where(val == R, (60 * (G - B) / d) % 360, hue)`
Where **red** is the largest channel, use the red formula (red sits at 0°). `% 360` turns a negative angle into a positive one, e.g. −15° → 345°. Red is done last, so if two channels tie for largest, red wins.
*Grey pixel:* R = G = B, so `val == R` and hue = 0 · … = 0.

**Why darkening doesn't change hue:** every hue formula is (difference of two channels) ÷ (difference of two channels). Multiplying R, G, B by k multiplies top and bottom by k, so k cancels.

**L49** `sat = delta / np.where(val == 0, 1, val)`
**Saturation** = Δ / max: 0 = grey, 1 = pure colour. It's also a ratio, so darkening doesn't change it. The `np.where` avoids dividing by zero on black pixels (there Δ = 0 too, so sat = 0).

**L50** `return hue, sat, val`

---

## L53–L60 · `# %% Mask`

**L54** `def colour_mask(hue, sat, val, rule):`
**In:** the H, S, V images and one fruit's rule. **Out:** a True/False image (a *mask*), True where the pixel has that fruit's colour.

**L55** `low, high = rule["hue"]`
Unpacks the hue range, e.g. low = 38, high = 62.

**L56–L57** `if low <= high: hue_ok = (hue >= low) & (hue <= high)`
Normal range: the hue must be between low **and** high. `&` is element-wise AND.

**L58–L59** `else: hue_ok = (hue >= low) | (hue <= high)`
Range that wraps round 0° (the apple: 320 → 6). The hue must be ≥ 320 **or** ≤ 6. `|` is element-wise OR.

**L60** `return hue_ok & (sat >= rule["sat_min"]) & (val >= rule["val_min"])`
A pixel passes only if all three tests are True.

---

## L63–L96 · `# %% Clean`

### Majority filter (L64–L71)

**L64** `def majority_filter(mask, size):`
**In:** a mask and an odd window size (15 for this image). **Out:** the cleaned mask.
**Idea:** a pixel stays True only if **more than half** of the size × size window around it is True. On a True/False image this is exactly a **median filter** (`medfilt2`): the median of 0s and 1s is 1 when more than half are 1. It removes specks and thin lines (the bowl rim, the wall shadow) and fills tiny holes.

**L65** `rows, cols = mask.shape`
Image height and width.

**L66** `padded = np.pad(mask.astype(float), size // 2)`
Turns True/False into 1/0 and adds a border of zeros `size // 2` pixels wide (7 for size 15), so windows at the edge still fit. `//` = whole-number division.

**L67** `total = np.zeros((rows, cols))`
An empty image that will hold the count of True pixels in each window.

**L68–L70** the double loop
```python
for dy in range(size):
    for dx in range(size):
        total += padded[dy:dy + rows, dx:dx + cols]
```
For every offset (dy, dx) inside the window, take a shifted copy of the padded mask and add it. After all size × size shifts, `total[y, x]` = the number of True pixels in the window around (x, y). This is the same as a **2-D convolution with a size × size kernel of ones** (`conv2(mask, ones(size))`).

**L71** `return total > size * size / 2`
Keep the pixel if more than half the window (more than 112.5 of 225 pixels for size 15) is True.

### Largest blob: flood fill (L74–L96)

We can't use `bwlabel`/`regionprops`, so we find connected regions ourselves.

**L74** `def largest_blob(mask):`
**In:** the cleaned mask. **Out:** a mask containing only the biggest connected group of True pixels.

**L75** `rows, cols = mask.shape`

**L76** `visited = np.zeros((rows, cols), dtype=bool)`
Remembers which pixels already belong to a blob we've filled, so no pixel is counted twice.

**L77** `best = []`
The list of (y, x) pixels of the biggest blob found so far.

**L78** `for y0, x0 in np.argwhere(mask):`
`np.argwhere` lists the (row, column) of every True pixel, like `find` in MATLAB. Each one could be the start of a new blob.

**L79–L80** `if visited[y0, x0]: continue`
Already part of a blob we filled → skip to the next pixel.

**L81** `visited[y0, x0] = True`
Mark the start pixel.

**L82** `blob = []`
The pixels of the new blob.

**L83** `stack = [(y0, x0)]`
A to-do list of pixels whose neighbours still have to be checked.

**L84** `while stack:`
Repeat while the to-do list isn't empty.

**L85** `y, x = stack.pop()`
Take one pixel off the to-do list.

**L86** `blob.append((y, x))`
Add it to the blob.

**L87** `for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:`
Its 4 neighbours: up, down, left, right (**4-connectivity**).

**L88** `if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not visited[ny, nx]:`
The neighbour must be inside the image, True in the mask, and not visited yet.

**L89–L90** `visited[ny, nx] = True` / `stack.append((ny, nx))`
Mark it and add it to the to-do list. When the to-do list is empty, the whole connected blob has been found.

**L91–L92** `if len(blob) > len(best): best = blob`
Keep the blob with the most pixels.

**L93** `result = np.zeros((rows, cols), dtype=bool)`
An empty mask.

**L94–L95** `for y, x in best: result[y, x] = True`
Paint the pixels of the biggest blob into it.

**L96** `return result`

**Why the largest blob?** The fruit is by far the biggest patch of its colour. Keeping only the largest blob guarantees exactly **one box per fruit** and that **nothing else is boxed**, which is what the viva checks.

---

## L99–L110 · `# %% Box and centroids`

**L100** `def box_and_centroids(blob):`
**In:** the mask of one fruit. **Out:** a dictionary with the box, both centroids, their distance and the area, or `None` if nothing was found.

**L101** `ys, xs = np.nonzero(blob)`
The rows (`ys`) and columns (`xs`) of every fruit pixel, like `[y, x] = find(blob)` in MATLAB.

**L102–L103** `if len(xs) == 0: return None`
No pixels → fruit not found.

**L104** `x_min, x_max, y_min, y_max = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())`
The **bounding box** is the smallest rectangle around all fruit pixels: leftmost, rightmost, top and bottom pixel. `int()` makes them plain whole numbers.

**L105** `box_c = ((x_min + x_max) / 2, (y_min + y_max) / 2)`
**Centroid a)**: the midpoint of the bounding box.

**L106** `m00, m10, m01 = len(xs), xs.sum(), ys.sum()`
**Moments** of the binary image, m_pq = Σ xᵖ · yᵠ over the fruit pixels:
- m00 = Σ 1 = number of pixels = **area** (zeroth moment)
- m10 = Σ x (first moment in x)
- m01 = Σ y (first moment in y)

**L107** `mom_c = (m10 / m00, m01 / m00)`
**Centroid b)**, the moment method: x̄ = m10/m00, ȳ = m01/m00. This is the average position of all fruit pixels, the "centre of mass".

**L108** `distance = np.hypot(box_c[0] - mom_c[0], box_c[1] - mom_c[1])`
Distance between the two centroids: `np.hypot(a, b)` = √(a² + b²) (Pythagoras).

**L109–L110** `return {...}`
Returns everything in one dictionary. `"area"` is used by `WhatIf_Task1.py`.

---

## L113–L122 · `# %% Find fruits`

**L114** `def find_fruits(img, normalise=True):`
**In:** an 8-bit image. **Out:** `{"apple": result, "capsicum": result, "lemon": result}`. `normalise` is always True in `Task1.py`. Only `WhatIf_Task1.py` switches it off once, to show what would happen without it.

**L115** `scale = img.max() if normalise else 255`
The number to divide by: the **brightest pixel** in the image (normal), or a fixed 255 (no correction). After darkening by 0.8 the brightest pixel is 204, so dividing by 204 gives back the original 0–1 image. This **cancels the dark factor**, so the `val_min` thresholds work for any darkFactor.

**L116** `hue, sat, val = rgb_to_hsv(img / scale)`
Scale to 0–1, then convert to HSV once. All three fruits use the same H, S, V images.

**L117** `size = 2 * round(img.shape[1] / 150) + 1`
Window size of the majority filter, scaled to the image width (`shape[1]` = number of columns) so it covers the same part of the scene at any resolution. `2 × … + 1` makes it odd, so the window has a centre pixel. For 1024 columns: 2 × round(6.83) + 1 = 2 × 7 + 1 = **15**.

**L118** `results = {}`
An empty dictionary for the answers.

**L119** `for name, rule in FRUIT_RULES.items():`
Loop over the three fruits: `name` = "apple", …; `rule` = its dictionary.

**L120** `mask = majority_filter(colour_mask(hue, sat, val, rule), size)`
Threshold the colours, then clean the result.

**L121** `results[name] = box_and_centroids(largest_blob(mask))`
Keep the largest blob, then measure its box and centroids.

**L122** `return results`

---

## L125–L156 · `# %% Run`

**L126** `if __name__ == "__main__":`
The code below runs only when you run `python Task1.py` yourself. When `WhatIf_Task1.py` does `from Task1 import …`, only the functions and settings are loaded. That way the What-if uses exactly the submitted code.

**L127** `img = load_image(image_file, darkFactor, mirror)`
Load, darken and mirror.

**L128** `results = find_fruits(img)`
Find all three fruits.

**L130–L131** two `print` lines
Print the settings and image size, and remind the reader which way x and y go.

**L132** `plt.figure(figsize=(9, 7))`
Open a new figure window, 9 × 7 inches.

**L133** `plt.imshow(img)`
Show the darkened, mirrored image: the one the fruits were found in. `imshow` puts y = 0 at the top, like the array.

**L134** `for name, r in results.items():`
One fruit at a time; `r` is its result dictionary.

**L135–L137** `if r is None: print(... NOT FOUND); continue`
If the fruit wasn't found, say so and skip the drawing.

**L138** `(bx, by), (mx, my) = r["box_centroid"], r["moment_centroid"]`
Unpack the two centroids into four numbers.

**L139–L140** `print(...)`
Print both centroids and the distance between them, as the task asks. `{name:9s}` pads the name to 9 characters so the columns line up; `{bx:6.1f}` prints a number 6 characters wide with 1 decimal.

**L141** `colour = FRUIT_RULES[name]["colour"]`
That fruit's drawing colour.

**L142** `x_min, y_min, x_max, y_max = r["box"]`
Unpack the box.

**L143–L144** `plt.gca().add_patch(Rectangle(...))`
`gca()` = "get current axes". Adds a rectangle with its corner at (x_min, y_min), width x_max − x_min and height y_max − y_min, not filled, 2 points thick.

**L145** `plt.text(x_min, y_min - 5, name, ...)`
Writes the fruit's name 5 pixels above its box.

**L146** `plt.plot(bx, by, "+", ...)`
**+** = centroid a), the box midpoint.

**L147** `plt.plot(mx, my, "x", ...)`
**x** = centroid b), the moment centroid.

**L148–L149** `plt.plot([], [], "k+", label=...)`, `plt.plot([], [], "kx", label=...)`
Two empty "plots" in black (`k`). They draw nothing; they only create the two legend entries that explain + and x.

**L150** `plt.legend(loc="lower right")`
Shows the legend.

**L151–L153** title and axis labels
The assignment wants every figure to have them.

**L154** `plt.tight_layout()`
Trims extra white space.

**L155** `plt.savefig("Task1_result.png", dpi=150)`
Saves the figure for the report (150 dots per inch).

**L156** `plt.show()`
Opens the window on screen.

---

## Result at the viva setting (darkFactor 0.8, mirror on)

```
apple    : box centroid = ( 666.5,  499.0)   moment centroid = ( 659.4,  505.1)   distance =  9.35 px
capsicum : box centroid = ( 869.0,  426.0)   moment centroid = ( 862.1,  435.5)   distance = 11.71 px
lemon    : box centroid = ( 380.5,  250.0)   moment centroid = ( 383.1,  253.6)   distance =  4.44 px
```

---

## Viva practice: answer out loud without notes

**What does this variable hold / where is it used?**
- `hue, sat, val`: three 783 × 1024 arrays made in L116 by `rgb_to_hsv`, used in L120 by `colour_mask`.
- `scale`: the brightest pixel value (204 at darkFactor 0.8). Used in L116 to divide the image to 0–1.
- `size`: 15, the majority-filter window width (L117). Used in L120.
- `mask` (L120): True/False image of the fruit's colour after cleaning. Used in L121.
- `visited` (L76): which pixels are already in a blob. Used in L79 and L88.
- `best` (L77): the pixels of the biggest blob so far. Used in L91 and L94.
- `m00, m10, m01` (L106): area, Σx, Σy. Used in L107.
- `results`: one dictionary per fruit, filled in L121 and used by the Run block.

**What goes in / comes out of a function?**
- `load_image`: file, dark factor, flip flag → 8-bit image.
- `rgb_to_hsv`: 0–1 image → hue, sat, val.
- `colour_mask`: H, S, V, one rule → mask.
- `majority_filter`: mask, size → cleaned mask.
- `largest_blob`: mask → mask of the largest blob only.
- `box_and_centroids`: blob → box, two centroids, distance, area.
- `find_fruits`: image → the result for all 3 fruits.

**Why HSV and not RGB?**
Darkening multiplies R, G, B by the same number. Hue and saturation are ratios, so the number cancels; in RGB every threshold would move with darkFactor.

**Why divide by `img.max()`?**
Value V = max(R, G, B) does change with darkening. Dividing by the brightest pixel cancels the dark factor, so `val_min` works for any darkFactor.

**How did you find the thresholds?**
Measured H, S, V of each fruit and of its look-alikes, then put each limit in the gap (table above). Example: the capsicum and the wall have the same hue, but the capsicum is more saturated (0.99 vs 0.87), so `sat_min = 0.90`.

**Why is the apple range `(320, 6)`?**
Red is at both ends of the hue circle. It means 320–360 OR 0–6, which is why L59 uses `|`.

**What does `% 360` do in L48?**
Turns negative hues into positive ones (−15 → 345), so all hues are 0–360.

**What does the majority filter do, and why?**
Keeps a pixel if more than half of its 15 × 15 window is True. It's a median filter for binary images (= convolution with a kernel of ones, then threshold at half). It removes specks and thin lines so they can't join the fruit or make extra blobs.

**What if `size` were 3, or 41?**
I tested both at darkFactor 0.8 mirrored. At 3 the cleaning is weak: the boxes grow by a few pixels (the lemon from x 286–475 to 278–476), because more ragged edge pixels survive, though the largest-blob step still throws away the separate specks. At 41 the cleaning is strong: it eats into the fruit edges and corners, so the boxes shrink (the capsicum's top edge drops from y 339 to 358, losing the stem area). All three fruits are still found in both cases. 15 is a middle value: it removes thin lines and specks without eating the fruit.

**Why the largest blob? Why 4 neighbours?**
`bwlabel` is not allowed, so we flood-fill. The largest blob gives exactly one box per fruit and ignores small leftovers. With 4-connectivity, two regions that touch only at a corner stay separate; 8-connectivity (adding diagonals) would also work here.

**What are m00, m10, m01?**
Area, Σx, Σy. Moment centroid = (m10/m00, m01/m00).

**Why are the two centroids different? Which is better?**
The box midpoint depends only on the 4 extreme pixels; the moment centroid uses every pixel. They differ when the shape isn't symmetric or has a hole. The capsicum has the biggest gap (11.7 px): its green stem isn't orange, so there's a hole near the top and the mass shifts down. The lemon is the most symmetric visible shape (4.4 px). The moment centroid is more robust: one stray edge pixel moves the box a lot but the average only a little.

**What changes with mirroring?**
x becomes 1023 − x; y stays the same; colours are unchanged, so the same fruits are found.

**What does L34 do?**
The task's darkening: multiply, round, keep within 0–255, store as 8-bit.

**Why `if __name__ == "__main__":`?**
So `WhatIf_Task1.py` can import the functions without running the plot. The What-if then uses exactly the submitted code.

**Which allowed MATLAB functions does your code match?**
`majority_filter` = `medfilt2` / `conv2` with ones; `np.argwhere` and `np.nonzero` = `find`; moment sums = `sum`. None of `regionprops`, `imfill`, `bwlabel`, `bwperim`, `bwboundaries`, `edge` is used.

**What happens if a fruit isn't found?**
`box_and_centroids` returns None and the Run block prints `NOT FOUND` and skips drawing that fruit.
