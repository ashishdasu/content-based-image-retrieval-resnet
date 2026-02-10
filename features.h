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

#endif // FEATURES_H
