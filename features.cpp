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
