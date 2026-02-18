# Content-Based Image Retrieval

**CS 5330 - Computer Vision | Project 2**
Ashish Dasu · Northeastern University

---

## Environment

- **Compiler:** Apple Clang (C++14)
- **Dependencies:** OpenCV 4.x, CMake 3.10+

---

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

Produces two executables in `build/`:

| Executable | Purpose |
|---|---|
| `cbir` | Live query: compute features on the fly and rank results |
| `compute_features` | Offline batch extraction: write features to CSV |

---

## Running the Programs

### Live query

```bash
./build/cbir <target_image> <image_dir> <feature_type> <N>
```

```bash
# Example: top 5 matches using RGB histogram
./build/cbir olympus/pic.0274.jpg olympus rgb_hist 5

# Example: top 3 using baseline SSD
./build/cbir olympus/pic.1016.jpg olympus baseline 3
```

### DNN query (uses precomputed embedding CSV)

```bash
./build/cbir <target_image> <embeddings.csv> dnn <N>

# Example
./build/cbir olympus/pic.0893.jpg ResNet18_olym.csv dnn 3
```

### Offline batch feature extraction

```bash
./build/compute_features <image_dir> <feature_type> <output.csv>

# Example
./build/compute_features olympus gabor features_gabor.csv
```

---

## Feature Types

| Feature | Flag | Vector size | Distance | Description |
|---|---|---|---|---|
| Baseline | `baseline` | 147 | SSD | Raw 7×7 center patch |
| RG Chromaticity | `rg_hist` | 256 | Intersection | 2D illumination invariant color histogram |
| RGB Histogram | `rgb_hist` | 512 | Intersection | 3D full-color histogram |
| Spatial Histogram | `multi_hist` | 1024 | Weighted intersection | Top/bottom RGB histograms concatenated |
| Texture + Color | `texture_color` | 528 | Weighted intersection | RGB histogram + Sobel magnitude histogram |
| GLCM | `cooccurrence` | 5 | SSD | Haralick statistics from gray co-occurrence matrix |
| Gabor | `gabor` | 64 | Per-filter intersection | Filter bank at 4 orientations × 2 scales |
| DNN | `dnn` | 512 | Cosine | ResNet18 global average pooling embeddings |
| Banana detector | `banana` | 4 | Weighted SSD | Yellow HSV blob: fraction + spatial coherence |
| Banana + DNN | `banana_dnn` | — | Combined | 0.5 × cosine + 0.5 × banana weighted SSD |
| Trash can detector | `trash_can` | 4 | Weighted SSD | Blue HSV blob: fraction + spatial coherence |

---

## Testing Extensions

### GLCM co-occurrence
```bash
./build/cbir olympus/pic.0274.jpg olympus cooccurrence 4
```

### Gabor texture
```bash
./build/cbir olympus/pic.0535.jpg olympus gabor 5
```

### Banana detector (combined DNN + yellow blob)
```bash
./build/cbir olympus/pic.0343.jpg olympus banana_dnn 5 ResNet18_olym.csv
```

### Blue trash can detector
```bash
./build/cbir olympus/pic.0287.jpg olympus trash_can 5
```

---

## File Structure

```
cbir.cpp              main query pipeline
compute_features.cpp  offline batch feature extractor
features.h / .cpp     all feature extraction implementations
distances.h / .cpp    SSD, histogram intersection, cosine, weighted SSD
CMakeLists.txt        build configuration
report.tex / .pdf     project report
figures/              images used in the report
```

---
