/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Main query program. Given a target image, an image database directory,
  a feature type, and N, returns the N closest images ranked by distance.

  Usage:
    cbir <target_image> <image_dir> <feature_type> <N>

  Supported feature types:
    baseline      - 7x7 center patch, SSD distance
    rg_hist       - 2D RG chromaticity histogram (16 bins), histogram intersection
    rgb_hist      - 3D RGB histogram (8 bins), histogram intersection
    multi_hist    - top/bottom RGB histograms (8 bins each), weighted intersection
    texture_color - RGB histogram + Sobel magnitude histogram, weighted intersection
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "features.h"
#include "distances.h"

// ---------------------------------------------------------------------------
// Directory scanning
// ---------------------------------------------------------------------------

/*
  Populates `files` with the full path of every image file in `dirpath`.
  Recognises .jpg, .png, .ppm, and .tif extensions.
  Results are sorted alphabetically for deterministic ordering.
*/
static int collectImages(const char *dirpath, std::vector<std::string> &files) {
    DIR *dirp = opendir(dirpath);
    if (!dirp) {
        fprintf(stderr, "Cannot open directory: %s\n", dirpath);
        return -1;
    }

    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        const char *name = dp->d_name;
        if (strstr(name, ".jpg") || strstr(name, ".png") ||
            strstr(name, ".ppm") || strstr(name, ".tif")) {
            files.push_back(std::string(dirpath) + "/" + name);
        }
    }
    closedir(dirp);
    std::sort(files.begin(), files.end());
    return 0;
}

// Extract just the filename from a full path
static std::string basename(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

// ---------------------------------------------------------------------------
// Feature dispatch helpers
// ---------------------------------------------------------------------------

static int computeFeature(cv::Mat &img, const char *feat_type,
                           std::vector<float> &fvec) {
    if (strcmp(feat_type, "baseline") == 0)
        return baselineFeature(img, fvec);
    if (strcmp(feat_type, "rg_hist") == 0)
        return rgChromaHistogram(img, fvec);
    if (strcmp(feat_type, "rgb_hist") == 0)
        return rgbHistogram(img, fvec);
    if (strcmp(feat_type, "multi_hist") == 0)
        return multiHistogram(img, fvec);
    if (strcmp(feat_type, "texture_color") == 0)
        return textureColorFeature(img, fvec);

    fprintf(stderr, "Unknown feature type: %s\n", feat_type);
    return -1;
}

static float computeDistance(const std::vector<float> &a,
                              const std::vector<float> &b,
                              const char *feat_type) {
    if (strcmp(feat_type, "baseline") == 0)
        return ssd(a, b);
    if (strcmp(feat_type, "rg_hist") == 0 || strcmp(feat_type, "rgb_hist") == 0)
        return histIntersection(a, b);
    if (strcmp(feat_type, "multi_hist") == 0)
        return multiHistDistance(a, b, 8 * 8 * 8);
    if (strcmp(feat_type, "texture_color") == 0)
        return multiHistDistance(a, b, 8 * 8 * 8); // split: 512 color + 16 texture

    return ssd(a, b);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <target> <image_dir> <feature_type> <N>\n\n", argv[0]);
        printf("  feature_type: baseline | rg_hist | rgb_hist | multi_hist | texture_color\n");
        printf("  N          : number of top matches to display\n");
        return 1;
    }

    const char *target_path = argv[1];
    const char *image_dir   = argv[2];
    const char *feat_type   = argv[3];
    int N = atoi(argv[4]);

    // ------------------------------------------------------------------
    // Step 1: compute features for the target image
    // ------------------------------------------------------------------
    cv::Mat target = cv::imread(target_path);
    if (target.empty()) {
        fprintf(stderr, "Could not read target image: %s\n", target_path);
        return -1;
    }

    std::vector<float> target_feat;
    if (computeFeature(target, feat_type, target_feat) != 0) return -1;

    // ------------------------------------------------------------------
    // Step 2: collect all database images
    // ------------------------------------------------------------------
    std::vector<std::string> db_paths;
    if (collectImages(image_dir, db_paths) != 0) return -1;

    printf("Database size: %zu images\n", db_paths.size());

    std::string target_base = basename(std::string(target_path));

    // ------------------------------------------------------------------
    // Step 3: compute features + distance for every database image
    // ------------------------------------------------------------------
    std::vector<std::pair<float, std::string>> results;
    results.reserve(db_paths.size());

    for (const auto &path : db_paths) {
        // skip the query image itself
        if (basename(path) == target_base) continue;

        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            fprintf(stderr, "  Warning: skipping unreadable file %s\n",
                    path.c_str());
            continue;
        }

        std::vector<float> feat;
        computeFeature(img, feat_type, feat);

        float dist = computeDistance(target_feat, feat, feat_type);
        results.push_back({dist, path});
    }

    // ------------------------------------------------------------------
    // Step 4: sort ascending and return top N
    // ------------------------------------------------------------------
    std::sort(results.begin(), results.end());

    int show = std::min(N, (int)results.size());
    printf("\nTop %d matches for %s  [method: %s]\n", show,
           target_base.c_str(), feat_type);
    printf("%-5s  %-12s  %s\n", "Rank", "Distance", "File");
    printf("%-5s  %-12s  %s\n", "----", "--------", "----");
    for (int i = 0; i < show; i++) {
        printf("%-5d  %-12.2f  %s\n", i + 1, results[i].first,
               basename(results[i].second).c_str());
    }

    return 0;
}
