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
    dnn           - ResNet18 embeddings from CSV, cosine distance
                    Usage for dnn: cbir <target_name> <csv_path> dnn <N>
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
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
    if (strcmp(feat_type, "cooccurrence") == 0)
        return cooccurrenceFeature(img, fvec);
    if (strcmp(feat_type, "banana") == 0)
        return bananaFeature(img, fvec);

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
    if (strcmp(feat_type, "cooccurrence") == 0)
        return ssd(a, b);
    if (strcmp(feat_type, "banana") == 0) {
        // weight: fraction x4, var_x x1, var_y x1, coherence x2
        std::vector<float> w = {4.0f, 1.0f, 1.0f, 2.0f};
        return weightedSSD(a, b, w);
    }

    return ssd(a, b);
}

// ---------------------------------------------------------------------------
// DNN embedding support
// ---------------------------------------------------------------------------

// Read a CSV where each row is: filename,v0,v1,...,v511
static int readEmbeddingsCSV(const char *csv_path,
                              std::vector<std::string> &names,
                              std::vector<std::vector<float>> &data) {
    FILE *fp = fopen(csv_path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open CSV: %s\n", csv_path);
        return -1;
    }

    char line[16384];
    while (fgets(line, sizeof(line), fp)) {
        char *tok = strtok(line, ",\n");
        if (!tok) continue;
        names.push_back(std::string(tok));

        std::vector<float> row;
        while ((tok = strtok(NULL, ",\n")) != NULL)
            row.push_back((float)atof(tok));
        data.push_back(row);
    }

    fclose(fp);
    return 0;
}

// Full DNN query — reads embeddings from CSV, ranks by cosine distance
static int queryDNN(const char *target_name, const char *csv_path, int N) {
    std::vector<std::string> names;
    std::vector<std::vector<float>> data;

    if (readEmbeddingsCSV(csv_path, names, data) != 0) return -1;
    printf("Loaded %zu embeddings from %s\n", names.size(), csv_path);

    std::string tbase = basename(std::string(target_name));

    int target_idx = -1;
    for (int i = 0; i < (int)names.size(); i++) {
        if (basename(names[i]) == tbase) { target_idx = i; break; }
    }
    if (target_idx < 0) {
        fprintf(stderr, "Target '%s' not found in CSV\n", target_name);
        return -1;
    }

    const std::vector<float> &tvec = data[target_idx];

    std::vector<std::pair<float, std::string>> results;
    for (int i = 0; i < (int)names.size(); i++) {
        if (i == target_idx) continue;
        results.push_back({cosineDistance(tvec, data[i]), names[i]});
    }

    std::sort(results.begin(), results.end());

    int show = std::min(N, (int)results.size());
    printf("\nTop %d matches for %s  [method: dnn / cosine]\n", show, tbase.c_str());
    printf("%-5s  %-12s  %s\n", "Rank", "Distance", "File");
    printf("%-5s  %-12s  %s\n", "----", "--------", "----");
    for (int i = 0; i < show; i++) {
        printf("%-5d  %-12.4f  %s\n", i + 1, results[i].first,
               basename(results[i].second).c_str());
    }
    return 0;
}

// Combined DNN + yellow-blob banana query
// Usage: cbir <target_name> <image_dir> banana_dnn <N> <csv_path>
static int queryBananaDNN(const char *target_name, const char *image_dir,
                           int N, const char *csv_path) {
    // Load DNN embeddings
    std::vector<std::string> emb_names;
    std::vector<std::vector<float>> emb_data;
    if (readEmbeddingsCSV(csv_path, emb_names, emb_data) != 0) return -1;

    // Build lookup: basename -> embedding index
    std::unordered_map<std::string, int> emb_idx;
    for (int i = 0; i < (int)emb_names.size(); i++)
        emb_idx[basename(emb_names[i])] = i;

    std::string tbase = basename(std::string(target_name));
    auto it = emb_idx.find(tbase);
    if (it == emb_idx.end()) {
        fprintf(stderr, "Target '%s' not found in CSV\n", target_name);
        return -1;
    }
    const std::vector<float> &tvec_dnn = emb_data[it->second];

    // Yellow feature for target
    cv::Mat target_img = cv::imread(std::string(image_dir) + "/" + tbase);
    if (target_img.empty()) {
        fprintf(stderr, "Cannot read target image\n"); return -1;
    }
    std::vector<float> tvec_yellow;
    bananaFeature(target_img, tvec_yellow);

    // Collect images
    std::vector<std::string> db_paths;
    collectImages(image_dir, db_paths);

    const std::vector<float> yellow_weights = {4.0f, 1.0f, 1.0f, 2.0f};

    std::vector<std::pair<float, std::string>> results;
    for (const auto &path : db_paths) {
        std::string bname = basename(path);
        if (bname == tbase) continue;

        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        std::vector<float> yvec;
        bananaFeature(img, yvec);

        float d_yellow = weightedSSD(tvec_yellow, yvec, yellow_weights);

        float d_dnn = 1.0f; // default high if not in CSV
        auto jt = emb_idx.find(bname);
        if (jt != emb_idx.end())
            d_dnn = cosineDistance(tvec_dnn, emb_data[jt->second]);

        // Equal weighting of the two signals
        float dist = 0.5f * d_dnn + 0.5f * d_yellow;
        results.push_back({dist, path});
    }

    std::sort(results.begin(), results.end());

    int show = std::min(N, (int)results.size());
    printf("\nTop %d matches for %s  [method: banana_dnn]\n", show, tbase.c_str());
    printf("%-5s  %-14s  %s\n", "Rank", "Distance", "File");
    printf("%-5s  %-14s  %s\n", "----", "--------", "----");
    for (int i = 0; i < show; i++) {
        printf("%-5d  %-14.6f  %s\n", i + 1, results[i].first,
               basename(results[i].second).c_str());
    }
    return 0;
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

    // DNN path: argv[2] is the CSV
    if (strcmp(feat_type, "dnn") == 0) {
        return queryDNN(target_path, image_dir, N);
    }
    // banana_dnn: needs image_dir and CSV; CSV passed as argv[5]
    if (strcmp(feat_type, "banana_dnn") == 0) {
        if (argc < 6) {
            fprintf(stderr, "banana_dnn requires: cbir <target_name> <image_dir> banana_dnn <N> <csv_path>\n");
            return 1;
        }
        return queryBananaDNN(target_path, image_dir, N, argv[5]);
    }

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
    printf("%-5s  %-14s  %s\n", "Rank", "Distance", "File");
    printf("%-5s  %-14s  %s\n", "----", "--------", "----");
    for (int i = 0; i < show; i++) {
        printf("%-5d  %-14.6f  %s\n", i + 1, results[i].first,
               basename(results[i].second).c_str());
    }

    return 0;
}
