/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Distance / similarity metric declarations for comparing feature vectors.
*/

#ifndef DISTANCES_H
#define DISTANCES_H

#include <vector>

/*
  Sum of squared differences. Both vectors must be the same length.
  Returns 0 when the two vectors are identical.
*/
float ssd(const std::vector<float> &a, const std::vector<float> &b);

// Histogram intersection distance: 1 - sum(min(a[i], b[i])).
// Both vectors must already be normalized (sum to 1).
float histIntersection(const std::vector<float> &a, const std::vector<float> &b);

// Weighted combination of histogram intersection over two spatial sub-histograms.
// histSize is the number of bins in each half (total vector length = 2 * histSize).
float multiHistDistance(const std::vector<float> &a, const std::vector<float> &b,
                        int histSize, float w1 = 0.5f, float w2 = 0.5f);

// Cosine distance: 1 - cos(theta). Computed via L2-normalized dot product.
float cosineDistance(const std::vector<float> &a, const std::vector<float> &b);

#endif // DISTANCES_H
