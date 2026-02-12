/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Distance metric implementations.
*/

#include "distances.h"
#include <cmath>

/*
  ssd

  Computes the sum-of-squared differences between two equal-length feature
  vectors. This is one of the simplest and most interpretable distance
  metrics: it penalizes large individual channel differences quadratically,
  making it sensitive to bright outlier pixels.

  Time complexity is O(n) in the length of the vectors.

  Returns 0.0 when the two vectors are identical (self-comparison).
*/
float ssd(const std::vector<float> &a, const std::vector<float> &b) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

/*
  histIntersection

  Measures dissimilarity between two normalized histograms as
  1 - sum_i( min(a[i], b[i]) ).

  The intersection sum is a similarity in [0, 1] for normalized inputs,
  so subtracting from 1 gives a proper distance where 0 = identical.
*/
float histIntersection(const std::vector<float> &a, const std::vector<float> &b) {
    float intersection = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        intersection += std::min(a[i], b[i]);
    }
    return 1.0f - intersection;
}

/*
  multiHistDistance

  Splits two concatenated spatial histograms at histSize and computes
  a weighted sum of the intersection distance for each region.
  Default weights are equal (0.5 / 0.5).
*/
float multiHistDistance(const std::vector<float> &a, const std::vector<float> &b,
                        int histSize, float w1, float w2) {
    std::vector<float> a1(a.begin(), a.begin() + histSize);
    std::vector<float> a2(a.begin() + histSize, a.end());
    std::vector<float> b1(b.begin(), b.begin() + histSize);
    std::vector<float> b2(b.begin() + histSize, b.end());

    return w1 * histIntersection(a1, b1) + w2 * histIntersection(a2, b2);
}

/*
  cosineDistance

  1 - cosine_similarity(a, b), where cosine similarity is the dot product
  of the L2-normalized vectors. Range is [0, 2]; 0 means identical direction.

  Normalizing first then dot-producting avoids a separate magnitude computation
  and handles the high-dimensional case (512-d embeddings) cleanly.
*/
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
