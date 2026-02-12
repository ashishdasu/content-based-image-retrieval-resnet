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

#endif // DISTANCES_H
