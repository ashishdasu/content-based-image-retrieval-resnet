/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Distance metric implementations.
*/

#include "distances.h"

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
