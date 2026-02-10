# Commit 1 — Project Setup & Baseline SSD Matching

## What is Content-Based Image Retrieval?

Content-Based Image Retrieval (CBIR) is the problem of finding images in a
database that are *visually similar* to a given query image — without relying
on text tags or manual annotations. The core idea is simple:

> Represent each image as a compact numeric **feature vector**, then rank
> database images by how *close* their feature vectors are to the query's.

Everything in this project is a variation on that theme. The methods differ in
*what* they measure (raw pixels, color distributions, texture, deep activations)
and *how* they measure closeness (SSD, histogram intersection, cosine distance).

---

## Project Architecture

The codebase is split into three layers:

| File | Role |
|------|------|
| `features.cpp / .h` | Feature extraction — one function per method |
| `distances.cpp / .h` | Distance metrics — one function per metric |
| `cbir.cpp` | Pipeline: read target → scan DB → sort → print top N |
| `CMakeLists.txt` | CMake build, links against OpenCV 4 |

This separation means adding a new feature or metric in later commits only
touches the relevant file; `cbir.cpp` just dispatches by name.

The database (1106 JPEG images in `olympus/`) is scanned at query time for
commit 1. Later commits will add an offline feature-storage step so repeated
queries don't re-read every pixel.

---

## Task 1 — Baseline: 7×7 Center Patch + SSD

### Feature Vector

The simplest possible descriptor: read the 7×7 square of pixels at the
exact center of the image and flatten all channel values into a vector.

```
patch_size = 7
cx = image_width  / 2
cy = image_height / 2

for r in [cy-3 .. cy+3]:
    for c in [cx-3 .. cx+3]:
        append B, G, R values of pixel (r, c)
```

For a standard 3-channel BGR image this gives **7 × 7 × 3 = 147 floats**.
No normalization is applied because SSD works in the same raw-pixel units.

### Distance Metric — Sum of Squared Differences

Given two feature vectors **a** and **b** of length n:

```
SSD(a, b) = Σ (a[i] - b[i])²   for i = 0 … n-1
```

Properties worth noting:
- **SSD = 0** iff the two patches are pixel-identical (self-match is 0 by
  construction — no numerical issues).
- Quadratic penalty: a single channel difference of 20 contributes 400 to
  the score, so the metric is sensitive to individual pixel outliers.
- Not normalized by patch size, but that's fine here because all feature
  vectors have the same length and we only compare same-method features.

### Required Result

Query: `pic.1016.jpg`  — a dark indoor scene with a specific color palette.

| Rank | File | Distance |
|------|------|----------|
| 1 | pic.0986.jpg | 14 049 |
| 2 | pic.0641.jpg | 21 756 |
| 3 | pic.0547.jpg | 49 703 |
| 4 | pic.1013.jpg | 51 539 |

These match the instructor's expected results exactly. The low distance for
`pic.0986` suggests its center patch has nearly identical color values to
`pic.1016` — both appear to be similarly lit and colored interior shots.

### Limitations of the Baseline

The 7×7 patch captures less than 0.1 % of a typical 640×480 image. It is
highly sensitive to framing (a slightly off-center subject completely changes
the descriptor) and ignores global color, texture, and layout entirely.
These limitations motivate the richer feature methods developed in later commits.

---

## How to Build and Run

```bash
mkdir build && cd build
cmake ..
make
```

```bash
# General usage
./build/cbir <target_image> <image_dir> baseline <N>

# Reproduce Task 1 required result
./build/cbir olympus/pic.1016.jpg olympus baseline 4
```
