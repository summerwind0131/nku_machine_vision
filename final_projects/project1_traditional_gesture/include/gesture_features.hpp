#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace gesture{
constexpr int NORM_SIZE = 128;

//将输入的图像同意转换为BGR三通道图像
inline cv::Mat toBGR(const cv::Mat &src){
    if(src.empty()){
        return cv::Mat();
    }
    if(src.depth()!=CV_8U){
        cv::Mat tmp8;
        cv::normalize(src,tmp8,0,255,cv::NORM_MINMAX);
        tmp8.convertTo(tmp8, CV_8U);
        return toBGR(tmp8);
    }
    cv::Mat bgr;
    if (src.channels() == 1) {
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    } else if (src.channels() == 3) {
        bgr = src.clone();
    } else if (src.channels() == 4) {
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    } else {
        bgr = src.clone();
    }
    return bgr;
}

//将输入的图像转为二值图像
inline cv::Mat forceBinary(const cv::Mat& in) {
    if (in.empty()) return cv::Mat();

    cv::Mat gray;
    if (in.channels() == 1) {
        gray = in;
    } else {
        cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U);
    return bin;
}

//计算二值图像边界区域中前景像素的比例。
inline double borderWhiteRatio(const cv::Mat& binInput) {
    cv::Mat bin = forceBinary(binInput);
    if (bin.empty()) return 1.0;

    int rows = bin.rows;
    int cols = bin.cols;

    int t = std::max(1, std::min(rows, cols) / 20);
    t = std::min(t, std::max(1, std::min(rows, cols) / 2));

    cv::Mat border = cv::Mat::zeros(rows, cols, CV_8U);
    border.rowRange(0, t).setTo(255);
    border.rowRange(rows - t, rows).setTo(255);
    border.colRange(0, t).setTo(255);
    border.colRange(cols - t, cols).setTo(255);

    cv::Mat inter;
    cv::bitwise_and(bin, border, inter);

    double denom = static_cast<double>(cv::countNonZero(border));
    if (denom <= 0) return 1.0;

    return cv::countNonZero(inter) / denom;
}

//提取最大连通域
inline cv::Mat largestComponent(const cv::Mat& binInput) {
    cv::Mat bin = forceBinary(binInput);
    if (bin.empty()) return cv::Mat();

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(
        bin, labels, stats, centroids, 8, CV_32S
    );

    if (n <= 1) {
        return cv::Mat::zeros(bin.size(), CV_8U);
    }

    int bestLabel = 1;
    int bestArea = stats.at<int>(1, cv::CC_STAT_AREA);

    for (int i = 2; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea) {
            bestArea = area;
            bestLabel = i;
        }
    }

    cv::Mat out;
    cv::compare(labels, bestLabel, out, cv::CMP_EQ);
    return out;
}

//填充手部区域内部的孔洞。
inline cv::Mat fillHoles(const cv::Mat& binInput) {
    cv::Mat bin = forceBinary(binInput);
    if (bin.empty()) return cv::Mat();

    cv::Mat flood = bin.clone();

    // 在图像四周加一圈黑边，方便从外部背景开始 floodFill。
    cv::copyMakeBorder(flood, flood, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));

    // 从左上角外部背景开始填充。
    cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));

    cv::Mat floodROI = flood(cv::Rect(1, 1, bin.cols, bin.rows));

    // floodROI 取反后，白色区域就是内部孔洞。
    cv::Mat holes;
    cv::bitwise_not(floodROI, holes);

    cv::Mat filled;
    cv::bitwise_or(bin, holes, filled);
    return filled;
}

//对初始分割 mask 进行后处理。
inline cv::Mat postProcessMask(const cv::Mat& rawMask) {
    cv::Mat m = forceBinary(rawMask);
    if (m.empty()) return cv::Mat();

    int k = std::max(3, std::min(m.rows, m.cols) / 80);
    if (k % 2 == 0) ++k;

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(k, k)
    );

    cv::morphologyEx(m, m, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(m, m, cv::MORPH_CLOSE, kernel);

    m = largestComponent(m);
    m = fillHoles(m);

    cv::morphologyEx(m, m, cv::MORPH_CLOSE, kernel);
    m = largestComponent(m);

    return m;
}

//判断 mask 是否有效。
inline bool validMask(const cv::Mat& maskInput) {
    cv::Mat mask = forceBinary(maskInput);
    if (mask.empty()) return false;

    double area = cv::countNonZero(mask);
    double total = static_cast<double>(mask.rows * mask.cols);
    if (total <= 0) return false;

    double ratio = area / total;

    if (ratio < 0.005 || ratio > 0.85) {
        return false;
    }

    if (borderWhiteRatio(mask) > 0.35) {
        return false;
    }

    return true;
}

inline cv::Mat alphaMaskIfAvailable(const cv::Mat& src) {
    if (src.empty() || src.channels() != 4) return cv::Mat();

    std::vector<cv::Mat> ch;
    cv::split(src, ch);

    cv::Mat alpha = ch[3];

    double minv = 0, maxv = 0;
    cv::minMaxLoc(alpha, &minv, &maxv);

    if (maxv - minv < 5 || maxv < 10) {
        return cv::Mat();
    }

    cv::Mat mask;
    cv::threshold(alpha, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    double ratio = cv::countNonZero(mask) / static_cast<double>(mask.rows * mask.cols);
    if (ratio < 0.005 || ratio > 0.98) {
        return cv::Mat();
    }

    return mask;
}

//基于背景颜色差异进行手部分割。
inline cv::Mat segmentByBorderDistance(const cv::Mat& bgr) {
    if (bgr.empty()) return cv::Mat();

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32F);

    int rows = lab.rows;
    int cols = lab.cols;

    int t = std::max(2, std::min(rows, cols) / 20);
    t = std::min(t, std::max(1, std::min(rows, cols) / 2));

    cv::Mat borderMask = cv::Mat::zeros(rows, cols, CV_8U);
    borderMask.rowRange(0, t).setTo(255);
    borderMask.rowRange(rows - t, rows).setTo(255);
    borderMask.colRange(0, t).setTo(255);
    borderMask.colRange(cols - t, cols).setTo(255);

    cv::Scalar mean, stddev;
    cv::meanStdDev(lab, mean, stddev, borderMask);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    cv::Mat dist = cv::Mat::zeros(rows, cols, CV_32F);

    for (int i = 0; i < 3; ++i) {
        cv::Mat d;
        cv::subtract(channels[i], cv::Scalar(mean[i]), d);

        // 防止标准差过小导致除零。
        double denom = std::max(3.0, stddev[i]);
        d = d / denom;
        dist += d.mul(d);
    }

    cv::sqrt(dist, dist);

    double minv = 0, maxv = 0;
    cv::minMaxLoc(dist, &minv, &maxv);

    if (maxv - minv < 1e-6) {
        return cv::Mat();
    }

    cv::Mat dist8;
    dist.convertTo(dist8, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));

    cv::GaussianBlur(dist8, dist8, cv::Size(5, 5), 0);

    cv::Mat bin;
    cv::threshold(dist8, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 如果边界区域大部分为白色，说明前景背景反了，需要取反。
    if (borderWhiteRatio(bin) > 0.5) {
        cv::bitwise_not(bin, bin);
    }

    return postProcessMask(bin);
}

//基于灰度 Otsu 阈值的分割方法。
inline cv::Mat segmentByGray(const cv::Mat& bgr) {
    if (bgr.empty()) return cv::Mat();

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    if (borderWhiteRatio(bin) > 0.5) {
        cv::bitwise_not(bin, bin);
    }

    return postProcessMask(bin);
}

//基于肤色模型的分割方法。
inline cv::Mat segmentBySkin(const cv::Mat& bgr) {
    if (bgr.empty()) return cv::Mat();

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat hmask1, hmask2, hmask;
    cv::inRange(hsv, cv::Scalar(0, 25, 40), cv::Scalar(25, 255, 255), hmask1);
    cv::inRange(hsv, cv::Scalar(160, 25, 40), cv::Scalar(180, 255, 255), hmask2);
    cv::bitwise_or(hmask1, hmask2, hmask);

    cv::Mat ycrcb;
    cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);

    cv::Mat ymask;
    cv::inRange(ycrcb, cv::Scalar(0, 133, 70), cv::Scalar(255, 180, 140), ymask);

    cv::Mat mask;
    cv::bitwise_and(hmask, ymask, mask);

    // 如果 HSV 和 YCrCb 交集太小，则退化为只使用 YCrCb 结果。
    double ratio = cv::countNonZero(mask) / static_cast<double>(mask.rows * mask.cols);
    if (ratio < 0.005) {
        mask = ymask;
    }

    return postProcessMask(mask);
}

inline double maskCandidateScore(const cv::Mat& maskInput, double sourceBias = 0.0) {
    cv::Mat mask = forceBinary(maskInput);
    if (!validMask(mask)) {
        return -std::numeric_limits<double>::infinity();
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return -std::numeric_limits<double>::infinity();
    }

    int best = 0;
    double bestArea = cv::contourArea(contours[0]);
    for (int i = 1; i < static_cast<int>(contours.size()); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            best = i;
        }
    }

    const std::vector<cv::Point>& c = contours[best];
    double area = std::max(1.0, cv::contourArea(c));
    double total = static_cast<double>(mask.rows * mask.cols);
    cv::Rect box = cv::boundingRect(c);

    std::vector<cv::Point> hullPts;
    cv::convexHull(c, hullPts);

    double hullArea = std::max(1.0, cv::contourArea(hullPts));
    double perim = std::max(1.0, cv::arcLength(c, true));

    double areaRatio = area / total;
    double borderRatio = borderWhiteRatio(mask);
    double widthCover = box.width / static_cast<double>(std::max(1, mask.cols));
    double heightCover = box.height / static_cast<double>(std::max(1, mask.rows));
    double boxRatio = box.area() / total;
    double extent = area / std::max(1.0, static_cast<double>(box.area()));
    double solidity = area / hullArea;
    double compactness = 4.0 * CV_PI * area / (perim * perim);

    double score = 100.0 + sourceBias;

    score -= borderRatio * 180.0;

    if (areaRatio < 0.015) {
        score -= (0.015 - areaRatio) * 5000.0;
    } else if (areaRatio < 0.04) {
        score -= (0.04 - areaRatio) * 900.0;
    }

    if (areaRatio > 0.55) {
        score -= (areaRatio - 0.55) * 450.0;
    } else if (areaRatio > 0.38) {
        score -= (areaRatio - 0.38) * 150.0;
    }

    if (widthCover > 0.92) {
        score -= (widthCover - 0.92) * 250.0;
    }
    if (heightCover > 0.98) {
        score -= (heightCover - 0.98) * 250.0;
    }

    if (boxRatio > 0.70 && extent > 0.82) {
        score -= (boxRatio - 0.70) * 130.0 + (extent - 0.82) * 90.0;
    }

    if (areaRatio > 0.30 && solidity > 0.97) {
        score -= (solidity - 0.97) * 400.0;
    }

    if (areaRatio > 0.20 && compactness > 0.86) {
        score -= (compactness - 0.86) * 120.0;
    }

    double preferredArea = 0.16;
    score += std::max(0.0, 18.0 * (1.0 - std::abs(areaRatio - preferredArea) / preferredArea));

    return score;
}

inline cv::Mat segmentHand(const cv::Mat& input) {
    if (input.empty()) return cv::Mat();

    struct Candidate {
        cv::Mat mask;
        double score;
    };

    std::vector<Candidate> candidates;

    auto addCandidate = [&](const cv::Mat& mask, double sourceBias) {
        if (mask.empty()) {
            return;
        }

        double score = maskCandidateScore(mask, sourceBias);
        if (std::isfinite(score)) {
            candidates.push_back({mask, score});
        }
    };

    cv::Mat alpha = alphaMaskIfAvailable(input);
    if (!alpha.empty()) {
        cv::Mat m = postProcessMask(alpha);
        addCandidate(m, 35.0);
    }

    cv::Mat bgr = toBGR(input);

    addCandidate(segmentByBorderDistance(bgr), 0.0);
    addCandidate(segmentBySkin(bgr), 10.0);
    addCandidate(segmentByGray(bgr), -8.0);

    if (!candidates.empty()) {
        auto best = std::max_element(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.score < b.score;
            }
        );

        return best->mask;
    }

    cv::Mat fallback = segmentByGray(bgr);
    if (!fallback.empty()) {
        return fallback;
    }

    fallback = segmentByBorderDistance(bgr);
    if (!fallback.empty()) {
        return fallback;
    }

    return segmentBySkin(bgr);
}

//将手部 mask 归一化到固定尺寸。
inline cv::Mat normalizeMask(const cv::Mat& maskInput) {
    cv::Mat mask = postProcessMask(maskInput);
    if (mask.empty()) return cv::Mat();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return cv::Mat();

    int best = 0;
    double bestArea = cv::contourArea(contours[0]);

    for (int i = 1; i < static_cast<int>(contours.size()); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            best = i;
        }
    }

    cv::Rect box = cv::boundingRect(contours[best]);

    int side = std::max(box.width, box.height);
    int margin = std::max(4, static_cast<int>(side * 0.12));

    cv::Mat crop = mask(box).clone();

    cv::Mat canvas = cv::Mat::zeros(side + 2 * margin, side + 2 * margin, CV_8U);

    int x = margin + (side - box.width) / 2;
    int y = margin + (side - box.height) / 2;

    crop.copyTo(canvas(cv::Rect(x, y, box.width, box.height)));

    cv::Mat norm;
    cv::resize(canvas, norm, cv::Size(NORM_SIZE, NORM_SIZE), 0, 0, cv::INTER_NEAREST);
    cv::threshold(norm, norm, 127, 255, cv::THRESH_BINARY);

    return norm;
}

//从输入图像中提取传统手势特征。
inline bool extractFeature(const cv::Mat& input, cv::Mat& feature, cv::Mat* debugNormMask = nullptr) {
    feature.release();

    if (input.empty()) return false;

    // 1. 分割手部区域。
    cv::Mat mask = segmentHand(input);

    // 2. 将手部区域归一化到固定尺寸。
    cv::Mat normMask = normalizeMask(mask);

    if (normMask.empty()) return false;

    if (debugNormMask) {
        *debugNormMask = normMask.clone();
    }

    /*
    HOG 特征：
    winSize = 128×128；
    blockSize = 32×32；
    blockStride = 16×16；
    cellSize = 16×16；
    bins = 9。
    */
    static cv::HOGDescriptor hog(
        cv::Size(NORM_SIZE, NORM_SIZE),
        cv::Size(32, 32),
        cv::Size(16, 16),
        cv::Size(16, 16),
        9
    );

    std::vector<float> hogDesc;
    hog.compute(normMask, hogDesc, cv::Size(0, 0), cv::Size(0, 0));

    // 提取最大轮廓，用于计算形状特征。
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(normMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return false;

    int best = 0;
    double bestArea = cv::contourArea(contours[0]);

    for (int i = 1; i < static_cast<int>(contours.size()); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            best = i;
        }
    }

    const std::vector<cv::Point>& c = contours[best];

    double area = cv::contourArea(c);
    double perim = cv::arcLength(c, true);
    cv::Rect box = cv::boundingRect(c);

    // 计算凸包，用于凸性、凹陷程度等特征。
    std::vector<cv::Point> hullPts;
    cv::convexHull(c, hullPts);

    double hullArea = std::max(1.0, cv::contourArea(hullPts));
    double hullPerim = std::max(1.0, cv::arcLength(hullPts, true));

    /*
    凸缺陷特征：
    手指之间的凹陷可以通过凸缺陷体现。
    Five 一般凸缺陷较多，V 一般有明显指间凹陷，A 通常较少。
    */
    std::vector<int> hullIdx;
    cv::convexHull(c, hullIdx, false, false);

    int defectCount = 0;
    double depthSum = 0.0;
    double maxDepth = 0.0;
    std::vector<cv::Point> defectFarPoints;
    std::vector<double> defectDepths;

    if (hullIdx.size() >= 4) {
        std::vector<cv::Vec4i> defects;
        try {
            cv::convexityDefects(c, hullIdx, defects);
        } catch (...) {
            defects.clear();
        }

        for (const auto& d : defects) {
            cv::Point s = c[d[0]];
            cv::Point e = c[d[1]];
            cv::Point f = c[d[2]];

            double depth = d[3] / 256.0;

            double a = cv::norm(s - f);
            double b = cv::norm(e - f);
            double cc = cv::norm(s - e);

            if (a < 1e-6 || b < 1e-6) continue;

            double cosv = (a * a + b * b - cc * cc) / (2.0 * a * b);
            cosv = std::max(-1.0, std::min(1.0, cosv));

            double angle = std::acos(cosv) * 180.0 / CV_PI;

            // 过滤过浅或角度过大的缺陷，减少噪声影响。
            if (depth > 7.0 && angle < 100.0) {
                ++defectCount;
                depthSum += depth;
                maxDepth = std::max(maxDepth, depth);
                defectFarPoints.push_back(f);
                defectDepths.push_back(depth);
            }
        }
    }

    // 计算图像矩和 Hu 不变矩。
    cv::Moments mom = cv::moments(c);
    if (std::abs(mom.m00) < 1e-6) return false;

    double cx = mom.m10 / mom.m00 / NORM_SIZE;
    double cy = mom.m01 / mom.m00 / NORM_SIZE;

    double mu20 = mom.mu20 / mom.m00;
    double mu02 = mom.mu02 / mom.m00;
    double mu11 = mom.mu11 / mom.m00;

    double trace = mu20 + mu02;
    double det = mu20 * mu02 - mu11 * mu11;
    double disc = std::sqrt(std::max(0.0, trace * trace / 4.0 - det));

    double lambda1 = trace / 2.0 + disc;
    double lambda2 = trace / 2.0 - disc;

    double eccentricity = 0.0;
    if (lambda1 > 1e-6) {
        eccentricity = std::sqrt(std::max(0.0, 1.0 - lambda2 / lambda1));
    }

    double centerX = cx * NORM_SIZE;
    double centerY = cy * NORM_SIZE;

    // Fingertip candidates are convex-hull points that protrude away from the palm.
    std::vector<cv::Point> tipCandidates;
    double minTipDistance = std::max(10.0, box.height * 0.25);
    double upperLimit = box.y + box.height * 0.82;

    for (const cv::Point& p : hullPts) {
        double dx = p.x - centerX;
        double dy = p.y - centerY;
        double dist = std::sqrt(dx * dx + dy * dy);

        bool upperPeak = p.y < centerY + box.height * 0.18;
        bool sidePeak = dist > box.height * 0.35 && p.y < upperLimit;

        if ((upperPeak || sidePeak) && dist > minTipDistance) {
            tipCandidates.push_back(p);
        }
    }

    std::sort(
        tipCandidates.begin(),
        tipCandidates.end(),
        [](const cv::Point& a, const cv::Point& b) {
            return a.x == b.x ? a.y < b.y : a.x < b.x;
        }
    );

    std::vector<cv::Point> fingertips;
    int mergeDist = std::max(6, box.width / 12);
    cv::Point centerPoint(static_cast<int>(centerX), static_cast<int>(centerY));

    for (const cv::Point& p : tipCandidates) {
        if (fingertips.empty() || std::abs(p.x - fingertips.back().x) > mergeDist) {
            fingertips.push_back(p);
            continue;
        }

        double oldDist = cv::norm(fingertips.back() - centerPoint);
        double newDist = cv::norm(p - centerPoint);
        if (newDist > oldDist || (std::abs(newDist - oldDist) < 1e-6 && p.y < fingertips.back().y)) {
            fingertips.back() = p;
        }
    }

    int fingertipCount = static_cast<int>(fingertips.size());
    double fingertipSpread = 0.0;
    double fingertipHeightMean = 0.0;

    if (!fingertips.empty()) {
        int minX = fingertips[0].x;
        int maxX = fingertips[0].x;
        double heightSum = 0.0;

        for (const cv::Point& p : fingertips) {
            minX = std::min(minX, p.x);
            maxX = std::max(maxX, p.x);
            heightSum += (centerY - p.y) / NORM_SIZE;
        }

        fingertipSpread = (maxX - minX) / static_cast<double>(NORM_SIZE);
        fingertipHeightMean = heightSum / fingertips.size();
    }

    int valleyCount = 0;
    double valleyDepthMean = 0.0;

    for (size_t i = 0; i < defectFarPoints.size(); ++i) {
        const cv::Point& p = defectFarPoints[i];
        double depth = defectDepths[i];

        if (p.y < box.y + box.height * 0.82 && depth > std::max(7.0, box.height * 0.06)) {
            ++valleyCount;
            valleyDepthMean += depth;
        }
    }

    if (valleyCount > 0) {
        valleyDepthMean /= valleyCount;
    }

    int upperProjectionPeaks = 0;
    {
        int y0 = std::max(0, box.y);
        int y1 = std::min(NORM_SIZE, box.y + static_cast<int>(box.height * 0.62));
        if (y1 <= y0) {
            y0 = 0;
            y1 = std::max(1, static_cast<int>(NORM_SIZE * 0.62));
        }

        std::vector<float> proj(NORM_SIZE, 0.0f);
        float maxProj = 0.0f;

        for (int x = 0; x < NORM_SIZE; ++x) {
            cv::Rect r(x, y0, 1, y1 - y0);
            proj[x] = cv::countNonZero(normMask(r)) / static_cast<float>(r.area());
            maxProj = std::max(maxProj, proj[x]);
        }

        std::vector<float> smooth(NORM_SIZE, 0.0f);
        for (int x = 0; x < NORM_SIZE; ++x) {
            int a = std::max(0, x - 2);
            int b = std::min(NORM_SIZE - 1, x + 2);
            float sum = 0.0f;
            for (int j = a; j <= b; ++j) {
                sum += proj[j];
            }
            smooth[x] = sum / static_cast<float>(b - a + 1);
        }

        float threshold = std::max(0.08f, maxProj * 0.35f);
        int runStart = -1;

        for (int x = 0; x <= NORM_SIZE; ++x) {
            bool active = x < NORM_SIZE && smooth[x] >= threshold;

            if (active && runStart < 0) {
                runStart = x;
            } else if (!active && runStart >= 0) {
                int runWidth = x - runStart;
                if (runWidth >= 3) {
                    ++upperProjectionPeaks;
                }
                runStart = -1;
            }
        }
    }

    double vTipGap = 0.0;
    double vTipAngle = 0.0;
    double vValleyDrop = 0.0;
    double vTipBalance = 1.0;
    double vPairScore = 0.0;
    double vPairHeight = 0.0;

    if (fingertips.size() >= 2) {
        std::vector<cv::Point> rankedTips = fingertips;
        std::sort(
            rankedTips.begin(),
            rankedTips.end(),
            [centerPoint](const cv::Point& a, const cv::Point& b) {
                if (a.y != b.y) return a.y < b.y;
                return cv::norm(a - centerPoint) > cv::norm(b - centerPoint);
            }
        );

        if (rankedTips.size() > 6) {
            rankedTips.resize(6);
        }

        for (size_t i = 0; i < rankedTips.size(); ++i) {
            for (size_t j = i + 1; j < rankedTips.size(); ++j) {
                cv::Point p1 = rankedTips[i];
                cv::Point p2 = rankedTips[j];

                double dx = std::abs(p1.x - p2.x);
                double dy = std::abs(p1.y - p2.y);
                if (dx < std::max(7.0, box.width * 0.08)) {
                    continue;
                }

                double gap = cv::norm(p1 - p2) / NORM_SIZE;
                double horizontalGap = dx / NORM_SIZE;
                double balance = dy / NORM_SIZE;
                double pairHeight = ((centerY - p1.y) + (centerY - p2.y)) / (2.0 * NORM_SIZE);

                cv::Point valley;
                double bestValleyDepth = 0.0;
                bool hasValley = false;

                int minTipX = std::min(p1.x, p2.x);
                int maxTipX = std::max(p1.x, p2.x);
                int margin = std::max(4, box.width / 16);

                for (size_t k = 0; k < defectFarPoints.size(); ++k) {
                    const cv::Point& fp = defectFarPoints[k];
                    double depth = defectDepths[k];

                    if (fp.x < minTipX - margin || fp.x > maxTipX + margin) {
                        continue;
                    }
                    if (fp.y <= std::min(p1.y, p2.y) || fp.y > box.y + box.height * 0.92) {
                        continue;
                    }

                    if (!hasValley || depth > bestValleyDepth) {
                        hasValley = true;
                        valley = fp;
                        bestValleyDepth = depth;
                    }
                }

                double valleyDrop = 0.0;
                double angleNorm = 0.0;

                if (hasValley) {
                    valleyDrop = std::max(0.0, static_cast<double>(valley.y - std::min(p1.y, p2.y))) / NORM_SIZE;

                    cv::Point2d v1(p1.x - valley.x, p1.y - valley.y);
                    cv::Point2d v2(p2.x - valley.x, p2.y - valley.y);
                    double n1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
                    double n2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);

                    if (n1 > 1e-6 && n2 > 1e-6) {
                        double dot = v1.x * v2.x + v1.y * v2.y;
                        double cosv = dot / (n1 * n2);
                        cosv = std::max(-1.0, std::min(1.0, cosv));
                        angleNorm = std::acos(cosv) / CV_PI;
                    }
                }

                double gapScore = 1.0 - std::abs(horizontalGap - 0.34) / 0.34;
                gapScore = std::max(0.0, std::min(1.0, gapScore));

                double heightScore = std::max(0.0, std::min(1.0, pairHeight / 0.30));
                double valleyScore = hasValley ? std::max(0.0, std::min(1.0, valleyDrop / 0.24)) : 0.0;
                double angleScore = angleNorm > 0.0 ? (1.0 - std::abs(angleNorm - 0.24) / 0.24) : 0.0;
                angleScore = std::max(0.0, std::min(1.0, angleScore));
                double balanceScore = std::max(0.0, 1.0 - balance / 0.25);

                double pairScore =
                    0.25 * gapScore +
                    0.22 * heightScore +
                    0.25 * valleyScore +
                    0.18 * angleScore +
                    0.10 * balanceScore;

                if (hasValley) {
                    pairScore += std::min(0.08, bestValleyDepth / NORM_SIZE);
                }

                if (pairScore > vPairScore) {
                    vPairScore = pairScore;
                    vTipGap = gap;
                    vTipAngle = angleNorm;
                    vValleyDrop = valleyDrop;
                    vTipBalance = balance;
                    vPairHeight = pairHeight;
                }
            }
        }
    }

    double hu[7];
    cv::HuMoments(mom, hu);

    std::vector<float> f;
    f.reserve(hogDesc.size() + 160);

    // 1. 加入 HOG 特征。
    for (float v : hogDesc) {
        f.push_back(v);
    }

    const double eps = 1e-6;

    // 2. 加入轮廓几何特征。
    f.push_back(static_cast<float>(area / (NORM_SIZE * NORM_SIZE)));
    f.push_back(static_cast<float>(perim / (4.0 * NORM_SIZE)));
    f.push_back(static_cast<float>(box.width / static_cast<double>(std::max(1, box.height))));
    f.push_back(static_cast<float>(box.width / static_cast<double>(NORM_SIZE)));
    f.push_back(static_cast<float>(box.height / static_cast<double>(NORM_SIZE)));
    f.push_back(static_cast<float>(area / (box.area() + eps)));
    f.push_back(static_cast<float>(area / (hullArea + eps)));
    f.push_back(static_cast<float>(4.0 * CV_PI * area / (perim * perim + eps)));
    f.push_back(static_cast<float>(hullArea / (NORM_SIZE * NORM_SIZE)));
    f.push_back(static_cast<float>(hullPerim / (4.0 * NORM_SIZE)));

    // 3. 加入凸缺陷特征。
    f.push_back(static_cast<float>(defectCount / 5.0));
    f.push_back(static_cast<float>(maxDepth / NORM_SIZE));
    f.push_back(static_cast<float>((defectCount > 0 ? depthSum / defectCount : 0.0) / NORM_SIZE));
    f.push_back(static_cast<float>(std::min(fingertipCount, 6) / 5.0));
    f.push_back(static_cast<float>(std::min(upperProjectionPeaks, 6) / 5.0));
    f.push_back(static_cast<float>(std::min(valleyCount, 6) / 5.0));
    f.push_back(static_cast<float>(fingertipSpread));
    f.push_back(static_cast<float>(fingertipHeightMean));
    f.push_back(static_cast<float>(valleyDepthMean / NORM_SIZE));
    f.push_back(static_cast<float>(vTipGap));
    f.push_back(static_cast<float>(vTipAngle));
    f.push_back(static_cast<float>(vValleyDrop));
    f.push_back(static_cast<float>(vTipBalance));
    f.push_back(static_cast<float>(vPairScore));
    f.push_back(static_cast<float>(vPairHeight));

    // 4. 加入质心和偏心率特征。
    f.push_back(static_cast<float>(cx));
    f.push_back(static_cast<float>(cy));
    f.push_back(static_cast<float>(eccentricity));

    // 5. 加入 Hu 不变矩。
    for (int i = 0; i < 7; ++i) {
        double val = -std::copysign(1.0, hu[i]) * std::log10(std::abs(hu[i]) + 1e-30);
        f.push_back(static_cast<float>(val));
    }

    /*
    6. 水平投影特征：
    将图像按高度划分为若干条带，
    统计每个条带中的前景像素比例。
    */
    const int bins = 16;

    for (int i = 0; i < bins; ++i) {
        int y0 = i * NORM_SIZE / bins;
        int y1 = (i + 1) * NORM_SIZE / bins;
        cv::Rect r(0, y0, NORM_SIZE, y1 - y0);
        float ratio = cv::countNonZero(normMask(r)) / static_cast<float>(r.area());
        f.push_back(ratio);
    }

    /*
    7. 垂直投影特征：
    将图像按宽度划分为若干条带，
    统计每个条带中的前景像素比例。
    */
    for (int i = 0; i < bins; ++i) {
        int x0 = i * NORM_SIZE / bins;
        int x1 = (i + 1) * NORM_SIZE / bins;
        cv::Rect r(x0, 0, x1 - x0, NORM_SIZE);
        float ratio = cv::countNonZero(normMask(r)) / static_cast<float>(r.area());
        f.push_back(ratio);
    }

    /*
    8. 网格占空比特征：
    将 128×128 图像划分为 8×8 网格，
    统计每个格子内手部前景像素比例，
    用于描述手势的局部空间分布。
    */
    const int grid = 8;

    for (int gy = 0; gy < grid; ++gy) {
        for (int gx = 0; gx < grid; ++gx) {
            int x0 = gx * NORM_SIZE / grid;
            int x1 = (gx + 1) * NORM_SIZE / grid;
            int y0 = gy * NORM_SIZE / grid;
            int y1 = (gy + 1) * NORM_SIZE / grid;

            cv::Rect r(x0, y0, x1 - x0, y1 - y0);
            float ratio = cv::countNonZero(normMask(r)) / static_cast<float>(r.area());
            f.push_back(ratio);
        }
    }

    // 将 std::vector<float> 转换为 OpenCV Mat，便于后续训练和预测。
    feature = cv::Mat(1, static_cast<int>(f.size()), CV_32F);

    for (int i = 0; i < static_cast<int>(f.size()); ++i) {
        feature.at<float>(0, i) = f[i];
    }

    return true;
}

//计算训练集特征的均值和标准差。
inline void computeScaler(const cv::Mat& X, cv::Mat& mean, cv::Mat& stddev) {
    CV_Assert(!X.empty());
    CV_Assert(X.type() == CV_32F);

    cv::reduce(X, mean, 0, cv::REDUCE_AVG, CV_32F);

    cv::Mat meanRep;
    cv::repeat(mean, X.rows, 1, meanRep);

    cv::Mat diff = X - meanRep;
    cv::Mat sq = diff.mul(diff);

    cv::Mat var;
    cv::reduce(sq, var, 0, cv::REDUCE_AVG, CV_32F);

    cv::sqrt(var, stddev);

    // 防止某些维度标准差过小导致除零。
    for (int c = 0; c < stddev.cols; ++c) {
        float& v = stddev.at<float>(0, c);
        if (v < 1e-6f || !std::isfinite(v)) {
            v = 1.0f;
        }
    }
}

/*
根据训练阶段保存的均值和标准差，对特征进行标准化。

公式：
x' = (x - mean) / stddev
*/
inline cv::Mat standardize(const cv::Mat& X, const cv::Mat& mean, const cv::Mat& stddev) {
    CV_Assert(!X.empty());
    CV_Assert(X.type() == CV_32F);
    CV_Assert(mean.rows == 1 && stddev.rows == 1);
    CV_Assert(X.cols == mean.cols && X.cols == stddev.cols);

    cv::Mat meanRep, stdRep;
    cv::repeat(mean, X.rows, 1, meanRep);
    cv::repeat(stddev, X.rows, 1, stdRep);

    cv::Mat out;
    cv::divide(X - meanRep, stdRep, out);
    return out;
}

//加权 KNN 分类器。
inline int predictWeightedKNN(
    const cv::Mat& trainX,
    const cv::Mat& trainLabels,
    const cv::Mat& sample,
    int k,
    int numClasses
) {
    CV_Assert(trainX.type() == CV_32F);
    CV_Assert(sample.type() == CV_32F);
    CV_Assert(sample.rows == 1);
    CV_Assert(sample.cols == trainX.cols);
    CV_Assert(trainLabels.rows == trainX.rows);

    int n = trainX.rows;
    k = std::max(1, std::min(k, n));

    std::vector<std::pair<float, int>> dist;
    dist.reserve(n);

    const float* sp = sample.ptr<float>(0);

    // 计算测试样本到所有训练样本的欧氏距离平方。
    for (int i = 0; i < n; ++i) {
        const float* xp = trainX.ptr<float>(i);

        double s = 0.0;
        for (int j = 0; j < trainX.cols; ++j) {
            double d = static_cast<double>(sp[j]) - xp[j];
            s += d * d;
        }

        int label = 0;
        if (trainLabels.type() == CV_32S) {
            label = trainLabels.at<int>(i, 0);
        } else {
            label = static_cast<int>(trainLabels.at<float>(i, 0));
        }

        dist.emplace_back(static_cast<float>(s), label);
    }

    // 找到距离最近的 K 个训练样本。
    if (k < n) {
        std::nth_element(
            dist.begin(),
            dist.begin() + k,
            dist.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            }
        );
    } else {
        std::sort(
            dist.begin(),
            dist.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            }
        );
    }

    std::vector<double> votes(numClasses, 0.0);

    int nearestLabel = dist[0].second;
    float nearestDist = dist[0].first;

    // 对最近 K 个样本进行加权投票。
    for (int i = 0; i < k; ++i) {
        int label = dist[i].second;
        float d2 = dist[i].first;

        if (d2 < nearestDist) {
            nearestDist = d2;
            nearestLabel = label;
        }

        if (label >= 0 && label < numClasses) {
            double d = std::sqrt(std::max(0.0f, d2));
            double w = 1.0 / (d + 1e-6);
            votes[label] += w;
        }
    }

    // 选择投票权重最大的类别作为最终预测结果。
    int best = nearestLabel;
    double bestVote = -1.0;

    for (int i = 0; i < numClasses; ++i) {
        if (votes[i] > bestVote) {
            bestVote = votes[i];
            best = i;
        }
    }

    return best;
}
}
