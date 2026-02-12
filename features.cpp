/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Implementations of image feature extraction routines.
*/

#include "features.h"
#include <cstdio>

/*
  baselineFeature

  Extracts the 7x7 center patch of the image as a flat feature vector.
  Pixels are read in row-major order with interleaved BGR channels, so the
  resulting vector has 7 * 7 * 3 = 147 elements.

  This is intentionally simple: the center crop carries the dominant color
  and rough structure of the subject without any invariance or compression.
  Two images with identical center patches will have SSD distance = 0.

  Returns 0 on success, -1 if the image is smaller than 7x7.
*/
int baselineFeature(cv::Mat &src, std::vector<float> &fvec) {
    fvec.clear();

    if (src.rows < 7 || src.cols < 7) {
        fprintf(stderr, "baselineFeature: image %dx%d is too small\n",
                src.cols, src.rows);
        return -1;
    }

    int cx = src.cols / 2;
    int cy = src.rows / 2;

    for (int r = cy - 3; r <= cy + 3; r++) {
        for (int c = cx - 3; c <= cx + 3; c++) {
            cv::Vec3b px = src.at<cv::Vec3b>(r, c);
            fvec.push_back((float)px[0]); // B
            fvec.push_back((float)px[1]); // G
            fvec.push_back((float)px[2]); // R
        }
    }

    return 0;
}

/*
  rgChromaHistogram

  Computes a normalized 2D histogram in RG chromaticity space.
  r = R/(R+G+B),  g = G/(R+G+B).  Pixels where R+G+B < 1 (pure black)
  are skipped to avoid division by zero.

  Stored row-major: index = r_bin * bins + g_bin.
  Values are normalized by the number of non-black pixels, so each bin
  holds the fraction of pixels falling there.
*/
int rgChromaHistogram(cv::Mat &src, std::vector<float> &fvec, int bins) {
    fvec.assign(bins * bins, 0.0f);

    int counted = 0;
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            cv::Vec3b px = src.at<cv::Vec3b>(r, c);
            float B = px[0], G = px[1], R = px[2];
            float total = B + G + R;
            if (total < 1.0f) continue;

            float rn = R / total;
            float gn = G / total;

            int ri = std::min((int)(rn * bins), bins - 1);
            int gi = std::min((int)(gn * bins), bins - 1);
            fvec[ri * bins + gi] += 1.0f;
            counted++;
        }
    }

    if (counted > 0) {
        for (auto &v : fvec) v /= counted;
    }

    return 0;
}

/*
  rgbHistogram

  Computes a normalized 3D RGB histogram by binning each channel independently.
  Index = r_bin * bins^2 + g_bin * bins + b_bin.
  Histogram is normalized by total pixel count.
*/
int rgbHistogram(cv::Mat &src, std::vector<float> &fvec, int bins) {
    fvec.assign(bins * bins * bins, 0.0f);

    float scale = bins / 256.0f;
    int total   = src.rows * src.cols;

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            cv::Vec3b px = src.at<cv::Vec3b>(r, c);
            int bi = std::min((int)(px[0] * scale), bins - 1); // B
            int gi = std::min((int)(px[1] * scale), bins - 1); // G
            int ri = std::min((int)(px[2] * scale), bins - 1); // R
            fvec[ri * bins * bins + gi * bins + bi] += 1.0f;
        }
    }

    for (auto &v : fvec) v /= total;

    return 0;
}
