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

// Whole-image RGB histogram + Sobel gradient magnitude histogram concatenated.
// colorBins^3 + textureBins values total.
int textureColorFeature(cv::Mat &src, std::vector<float> &fvec,
                        int colorBins = 8, int textureBins = 16);

// Gray-level co-occurrence matrix (GLCM) texture features.
// Computes 5 Haralick statistics averaged over 4 directions, normalized to [0,1].
// levels: number of gray levels to quantize to (default 8).
int cooccurrenceFeature(cv::Mat &src, std::vector<float> &fvec, int levels = 8);

// Custom banana feature: yellow pixel fraction + spatial variance of yellow pixels.
// Designed to find images containing bananas or similarly-colored yellow blobs.
// Returns a 4-element vector: [fraction, var_x, var_y, coherence].
int bananaFeature(cv::Mat &src, std::vector<float> &fvec);

// Blue blob feature for trash can detection.
// Same structure as bananaFeature but tuned to blue HSV range.
// Returns a 4-element vector: [fraction, var_x, var_y, coherence].
int trashCanFeature(cv::Mat &src, std::vector<float> &fvec);

// Gabor texture feature: histograms of filter responses across multiple
// orientations and scales. textureBins bins per filter, concatenated.
// orientations: number of angle steps in [0, pi)
// scales: number of wavelength values to try
// Total vector length: orientations * scales * textureBins
int gaborFeature(cv::Mat &src, std::vector<float> &fvec,
                 int orientations = 4, int scales = 2, int textureBins = 8);

#endif // FEATURES_H
