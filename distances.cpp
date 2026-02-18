/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Distance metric implementations.
*/

#include "distances.h"
#include <cmath>

// Sum of squared differences. Returns 0 when the two vectors are identical.
float ssd(const std::vector<float> &a, const std::vector<float> &b) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

// 1 - sum(min(a[i], b[i])). Returns 0 for identical normalized histograms.
float histIntersection(const std::vector<float> &a, const std::vector<float> &b) {
    float intersection = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        intersection += std::min(a[i], b[i]);
    }
    return 1.0f - intersection;
}

// Splits two concatenated histograms at histSize and returns a weighted
// average of their intersection distances.
float multiHistDistance(const std::vector<float> &a, const std::vector<float> &b,
                        int histSize, float w1, float w2) {
    std::vector<float> a1(a.begin(), a.begin() + histSize);
    std::vector<float> a2(a.begin() + histSize, a.end());
    std::vector<float> b1(b.begin(), b.begin() + histSize);
    std::vector<float> b2(b.begin() + histSize, b.end());

    return w1 * histIntersection(a1, b1) + w2 * histIntersection(a2, b2);
}

// 1 - cosine_similarity(a, b). Range [0, 2]; 0 means identical direction.
float cosineDistance(const std::vector<float> &a, const std::vector<float> &b) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-8f || nb < 1e-8f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

// SSD with per-element weights.
float weightedSSD(const std::vector<float> &a, const std::vector<float> &b,
                  const std::vector<float> &weights) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = a[i] - b[i];
        dist += weights[i] * d * d;
    }
    return dist;
}
