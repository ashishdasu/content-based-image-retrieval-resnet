/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Offline feature pre-computation. Scans an image directory, computes the
  requested feature vector for every image, and writes the results to a CSV
  file (filename, f0, f1, ..., fN per line). The CSV can then be passed to
  a query program to avoid recomputing features for every query.

  Usage:
    compute_features <image_dir> <feature_type> <output_csv>

  Supported feature types:
    baseline      - 7x7 center patch (147 values)
    rg_hist       - 2D RG chromaticity histogram, 16 bins (256 values)
    rgb_hist      - 3D RGB histogram, 8 bins (512 values)
    multi_hist    - top/bottom RGB histograms, 8 bins (1024 values)
    texture_color - RGB histogram + Sobel magnitude histogram (528 values)
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

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <image_dir> <feature_type> <output_csv>\n\n", argv[0]);
        printf("  feature_type: baseline | rg_hist | rgb_hist\n");
        return 1;
    }

    const char *image_dir = argv[1];
    const char *feat_type = argv[2];
    const char *out_csv   = argv[3];

    std::vector<std::string> paths;
    if (collectImages(image_dir, paths) != 0) return -1;

    FILE *fp = fopen(out_csv, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file: %s\n", out_csv);
        return -1;
    }

    int processed = 0;
    for (const auto &path : paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            fprintf(stderr, "  Skipping: %s\n", path.c_str());
            continue;
        }

        std::vector<float> feat;
        int status = -1;

        if (strcmp(feat_type, "baseline") == 0)
            status = baselineFeature(img, feat);
        else if (strcmp(feat_type, "rg_hist") == 0)
            status = rgChromaHistogram(img, feat);
        else if (strcmp(feat_type, "rgb_hist") == 0)
            status = rgbHistogram(img, feat);
        else if (strcmp(feat_type, "multi_hist") == 0)
            status = multiHistogram(img, feat);
        else if (strcmp(feat_type, "texture_color") == 0)
            status = textureColorFeature(img, feat);
        else if (strcmp(feat_type, "cooccurrence") == 0)
            status = cooccurrenceFeature(img, feat);
        else if (strcmp(feat_type, "banana") == 0)
            status = bananaFeature(img, feat);
        else if (strcmp(feat_type, "trash_can") == 0)
            status = trashCanFeature(img, feat);
        else if (strcmp(feat_type, "gabor") == 0)
            status = gaborFeature(img, feat);
        else {
            fprintf(stderr, "Unknown feature type: %s\n", feat_type);
            fclose(fp);
            return -1;
        }

        if (status != 0) continue;

        // write: path,v0,v1,...,vN
        fprintf(fp, "%s", path.c_str());
        for (float v : feat) fprintf(fp, ",%.6f", v);
        fprintf(fp, "\n");
        processed++;
    }

    fclose(fp);
    printf("Wrote features for %d images to %s\n", processed, out_csv);
    return 0;
}
