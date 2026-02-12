/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Feature extraction functions. Each function takes a source image and fills
  a float vector with the computed feature descriptor.
*/

#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include "opencv2/opencv.hpp"

/*
  Baseline feature: flatten the 7x7 center patch into a raw pixel vector.
  For a 3-channel image this gives 147 values (7 * 7 * 3).
  Channel order follows OpenCV's default BGR.
*/
int baselineFeature(cv::Mat &src, std::vector<float> &fvec);

// Normalized 2D RG chromaticity histogram. bins^2 values.
int rgChromaHistogram(cv::Mat &src, std::vector<float> &fvec, int bins = 16);

// Normalized 3D RGB histogram. bins^3 values.
int rgbHistogram(cv::Mat &src, std::vector<float> &fvec, int bins = 8);

// Top-half + bottom-half RGB histograms concatenated. 2 * bins^3 values.
int multiHistogram(cv::Mat &src, std::vector<float> &fvec, int bins = 8);

#endif // FEATURES_H
