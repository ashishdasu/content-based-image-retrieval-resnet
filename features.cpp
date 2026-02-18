/*
  CS 5330 Computer Vision
  Project 2 - Content-Based Image Retrieval
  Ashish Dasu

  Implementations of image feature extraction routines.
*/

#include "features.h"
#include <cstdio>
#include <cmath>

// Flattens the 7x7 center patch into a BGR pixel vector (147 values).
// Returns -1 if the image is smaller than 7x7.
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

// Normalized 2D RG chromaticity histogram. Pure-black pixels skipped.
// Stored row-major: index = r_bin * bins + g_bin.
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

// Normalized 3D RGB histogram. Index = r_bin * bins^2 + g_bin * bins + b_bin.
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

// RGB histograms for the top and bottom halves, concatenated (2 * bins^3 values).
int multiHistogram(cv::Mat &src, std::vector<float> &fvec, int bins) {
    fvec.clear();

    int mid = src.rows / 2;

    cv::Mat top    = src(cv::Rect(0, 0,   src.cols, mid));
    cv::Mat bottom = src(cv::Rect(0, mid, src.cols, src.rows - mid));

    std::vector<float> top_hist, bot_hist;
    rgbHistogram(top,    top_hist, bins);
    rgbHistogram(bottom, bot_hist, bins);

    fvec.insert(fvec.end(), top_hist.begin(), top_hist.end());
    fvec.insert(fvec.end(), bot_hist.begin(), bot_hist.end());

    return 0;
}

// GLCM texture descriptor. Builds co-occurrence matrices across 4 directions,
// extracts 5 Haralick statistics (energy, entropy, contrast, homogeneity,
// correlation), averages across directions, and normalizes each to [0, 1].
int cooccurrenceFeature(cv::Mat &src, std::vector<float> &fvec, int levels) {
    fvec.clear();

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 4 offsets: (dx, dy)
    const int dx[] = {1, 0, 1,  1};
    const int dy[] = {0, 1, 1, -1};
    const int ndirs = 4;

    float sum_energy = 0, sum_entropy = 0, sum_contrast = 0;
    float sum_homogeneity = 0, sum_correlation = 0;

    for (int d = 0; d < ndirs; d++) {
        // Build and normalize GLCM
        std::vector<std::vector<float>> glcm(levels, std::vector<float>(levels, 0.0f));
        float count = 0.0f;

        int r0 = (dy[d] < 0) ? -dy[d] : 0;
        int r1 = src.rows - ((dy[d] > 0) ? dy[d] : 0);
        int c0 = (dx[d] < 0) ? -dx[d] : 0;
        int c1 = src.cols - ((dx[d] > 0) ? dx[d] : 0);

        for (int r = r0; r < r1; r++) {
            for (int c = c0; c < c1; c++) {
                int i = gray.at<uchar>(r, c) * levels / 256;
                int j = gray.at<uchar>(r + dy[d], c + dx[d]) * levels / 256;
                i = std::min(i, levels - 1);
                j = std::min(j, levels - 1);
                glcm[i][j] += 1.0f;
                glcm[j][i] += 1.0f;
                count += 2.0f;
            }
        }

        if (count < 1.0f) continue;
        for (int i = 0; i < levels; i++)
            for (int j = 0; j < levels; j++)
                glcm[i][j] /= count;

        // Mean and std for correlation (symmetric GLCM: mu_i == mu_j)
        float mu = 0.0f;
        for (int i = 0; i < levels; i++) {
            float row = 0.0f;
            for (int j = 0; j < levels; j++) row += glcm[i][j];
            mu += i * row;
        }
        float sigma2 = 0.0f;
        for (int i = 0; i < levels; i++) {
            float row = 0.0f;
            for (int j = 0; j < levels; j++) row += glcm[i][j];
            sigma2 += (i - mu) * (i - mu) * row;
        }

        float energy = 0, entropy = 0, contrast = 0, homogeneity = 0, correlation = 0;
        for (int i = 0; i < levels; i++) {
            for (int j = 0; j < levels; j++) {
                float p = glcm[i][j];
                if (p < 1e-12f) continue;
                energy      += p * p;
                entropy     -= p * std::log(p + 1e-12f);
                contrast    += (float)(i - j) * (i - j) * p;
                homogeneity += p / (1.0f + (i - j) * (i - j));
                if (sigma2 > 1e-8f)
                    correlation += (i - mu) * (j - mu) * p / sigma2;
            }
        }

        sum_energy      += energy;
        sum_entropy     += entropy;
        sum_contrast    += contrast;
        sum_homogeneity += homogeneity;
        sum_correlation += correlation;
    }

    // Average across directions, then normalize to [0, 1]
    float max_entropy = std::log((float)(levels * levels));
    float max_contrast = (float)(levels - 1) * (levels - 1);

    fvec.push_back(sum_energy      / ndirs);                        // [0, 1]
    fvec.push_back(sum_entropy     / ndirs / max_entropy);          // [0, 1]
    fvec.push_back(sum_contrast    / ndirs / max_contrast);         // [0, 1]
    fvec.push_back(sum_homogeneity / ndirs);                        // [0, 1]
    fvec.push_back((sum_correlation / ndirs + 1.0f) / 2.0f);       // [-1,1] -> [0,1]

    return 0;
}

// RGB histogram (colorBins^3) concatenated with a Sobel gradient magnitude
// histogram (textureBins values). Magnitudes capped at 1000 before binning.
int textureColorFeature(cv::Mat &src, std::vector<float> &fvec,
                        int colorBins, int textureBins) {
    fvec.clear();

    std::vector<float> color_hist;
    rgbHistogram(src, color_hist, colorBins);

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat sx, sy;
    cv::Sobel(gray, sx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sy, CV_32F, 0, 1, 3);

    cv::Mat mag;
    cv::magnitude(sx, sy, mag);

    const float max_mag = 1000.0f;
    float scale = textureBins / max_mag;
    int total = mag.rows * mag.cols;

    std::vector<float> texture_hist(textureBins, 0.0f);
    for (int r = 0; r < mag.rows; r++) {
        for (int c = 0; c < mag.cols; c++) {
            int bin = std::min((int)(mag.at<float>(r, c) * scale), textureBins - 1);
            texture_hist[bin] += 1.0f;
        }
    }
    for (auto &v : texture_hist) v /= total;

    fvec.insert(fvec.end(), color_hist.begin(), color_hist.end());
    fvec.insert(fvec.end(), texture_hist.begin(), texture_hist.end());

    return 0;
}

// Yellow HSV blob descriptor: [fraction, var_x, var_y, coherence].
// HSV range H:[15,40] S:[80,255] V:[80,255]. Coherence = 1/(1+var_x+var_y).
int bananaFeature(cv::Mat &src, std::vector<float> &fvec) {
    fvec.clear();

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(15, 80, 80), cv::Scalar(40, 255, 255), mask);

    int total = src.rows * src.cols;

    std::vector<float> xs, ys;
    xs.reserve(total / 4);
    ys.reserve(total / 4);

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            if (mask.at<uchar>(r, c) > 0) {
                xs.push_back((float)c / src.cols);
                ys.push_back((float)r / src.rows);
            }
        }
    }

    float fraction = (float)xs.size() / total;

    float var_x = 0.0f, var_y = 0.0f;
    if (xs.size() > 1) {
        float mx = 0.0f, my = 0.0f;
        for (float x : xs) mx += x;
        for (float y : ys) my += y;
        mx /= xs.size();
        my /= ys.size();

        for (float x : xs) var_x += (x - mx) * (x - mx);
        for (float y : ys) var_y += (y - my) * (y - my);
        var_x /= xs.size();
        var_y /= ys.size();
    }

    float coherence = 1.0f / (1.0f + var_x + var_y);

    fvec.push_back(fraction);
    fvec.push_back(var_x);
    fvec.push_back(var_y);
    fvec.push_back(coherence);

    return 0;
}

// Blue HSV blob descriptor for trash can detection. Same structure as bananaFeature.
// HSV range H:[100,130] S:[80,255] V:[40,255]. Lower V allows bins in shadow.
int trashCanFeature(cv::Mat &src, std::vector<float> &fvec) {
    fvec.clear();

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(100, 80, 40), cv::Scalar(130, 255, 255), mask);

    int total = src.rows * src.cols;

    std::vector<float> xs, ys;
    xs.reserve(total / 4);
    ys.reserve(total / 4);

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            if (mask.at<uchar>(r, c) > 0) {
                xs.push_back((float)c / src.cols);
                ys.push_back((float)r / src.rows);
            }
        }
    }

    float fraction = (float)xs.size() / total;
    float var_x = 0.0f, var_y = 0.0f;

    if (xs.size() > 1) {
        float mx = 0.0f, my = 0.0f;
        for (float x : xs) mx += x;
        for (float y : ys) my += y;
        mx /= xs.size();
        my /= ys.size();

        for (float x : xs) var_x += (x - mx) * (x - mx);
        for (float y : ys) var_y += (y - my) * (y - my);
        var_x /= xs.size();
        var_y /= ys.size();
    }

    float coherence = 1.0f / (1.0f + var_x + var_y);

    fvec.push_back(fraction);
    fvec.push_back(var_x);
    fvec.push_back(var_y);
    fvec.push_back(coherence);

    return 0;
}

// Gabor filter bank texture descriptor. Applies filters at orientations x scales,
// histograms each response magnitude, and concatenates. Total: orientations * scales * textureBins.
int gaborFeature(cv::Mat &src, std::vector<float> &fvec,
                 int orientations, int scales, int textureBins) {
    fvec.clear();

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray32;
    gray.convertTo(gray32, CV_32F);

    // wavelengths for each scale (fine to coarse)
    std::vector<double> lambdas = {4.0, 8.0};
    while ((int)lambdas.size() < scales)
        lambdas.push_back(lambdas.back() * 2.0);
    lambdas.resize(scales);

    const float max_response = 255.0f;
    const float scale_factor = textureBins / max_response;
    int total = src.rows * src.cols;

    for (int s = 0; s < scales; s++) {
        double lambda = lambdas[s];
        double sigma  = lambda / CV_PI;

        for (int o = 0; o < orientations; o++) {
            double theta = o * CV_PI / orientations;

            cv::Mat kernel = cv::getGaborKernel(
                cv::Size(15, 15), sigma, theta, lambda, 0.5, 0, CV_32F);

            cv::Mat response;
            cv::filter2D(gray32, response, CV_32F, kernel);

            cv::Mat mag;
            cv::magnitude(response, cv::Mat::zeros(response.size(), CV_32F), mag);

            std::vector<float> hist(textureBins, 0.0f);
            for (int r = 0; r < mag.rows; r++) {
                for (int c = 0; c < mag.cols; c++) {
                    int bin = std::min((int)(mag.at<float>(r, c) * scale_factor),
                                      textureBins - 1);
                    hist[bin] += 1.0f;
                }
            }
            for (auto &v : hist) v /= total;

            fvec.insert(fvec.end(), hist.begin(), hist.end());
        }
    }

    return 0;
}
