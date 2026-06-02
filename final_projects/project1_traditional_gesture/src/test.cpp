//识别全部四种手势
/*
机器视觉技术课程大作业 Part1 —— 传统手势识别


识别类别：
A、C、Five、V

测试程序功能：
加载训练好的传统分类模型文件 gesture_knn.yml，
对指定文件夹中的所有 PNG 格式图像进行手势识别，
并依次打印每张图像的识别结果：A、C、Five 或 V。

运行示例：
test_gesture <测试图片文件夹路径> <gesture_knn.yml>
*/
#include "gesture_features.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

/*
测试程序说明：

该程序不进行训练，只负责：
1. 加载训练好的模型文件；
2. 读取测试文件夹中的所有 PNG 图像；
3. 对每张图像提取同样的传统视觉特征；
4. 使用训练阶段保存的均值和标准差进行标准化；
5. 使用加权 KNN 进行分类；
6. 输出识别结果。
*/

namespace fs = std::filesystem;

/*将字符串转换为小写。*/
static std::string lowerString(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

/*使用二进制方式读取图片，避免中文路径下 imread 可能读取失败的问题。*/
static cv::Mat imreadAnyPath(const fs::path& path, int flags = cv::IMREAD_UNCHANGED) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return cv::Mat();

    std::vector<unsigned char> buffer(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>()
    );

    if (buffer.empty()) return cv::Mat();

    return cv::imdecode(buffer, flags);
}

/*列出测试文件夹中的所有 PNG 图像。
根据作业要求：
测试代码需要能直接对一个文件夹中的所有 PNG 图像进行识别。
*/
static std::vector<fs::path> listPngFiles(const fs::path& dir) {
    std::vector<fs::path> files;

    if (!fs::exists(dir)) return files;

    for (const auto& e : fs::recursive_directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;

        std::string ext = lowerString(e.path().extension().string());
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
            files.push_back(e.path());
        }
    }

    // 排序后输出
    std::sort(files.begin(), files.end());
    return files;
}

/*
从模型文件中读取类别名称。
模型文件中的 classes 字段保存：
A、C、Five、V。
*/
static std::vector<std::string> readClassNames(const cv::FileNode& node) {
    std::vector<std::string> classes;

    if (node.type() == cv::FileNode::SEQ) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            classes.push_back(static_cast<std::string>(*it));
        }
    }

    return classes;
}

static std::string sanitizeFileName(std::string s) {
    for (char& c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '.' && c != '_' && c != '-') {
            c = '_';
        }
    }
    return s;
}

static bool classTokenInPath(const fs::path& path, const std::string& cls) {
    std::string label = lowerString(cls);
    std::string stem = lowerString(path.stem().string());
    std::string parent = lowerString(path.parent_path().filename().string());

    if (label == "five") {
        return stem.find("five") != std::string::npos ||
               stem.find("sign 5") != std::string::npos ||
               parent == "five" ||
               parent == "5";
    }

    if (label.size() == 1) {
        char ch = label[0];
        std::vector<std::string> tokens = {
            label,
            label + "-",
            label + "_",
            std::string("test_") + label,
            std::string("test-") + label,
            std::string("sign_") + label,
            std::string("sign-") + label
        };

        if (parent == label) {
            return true;
        }

        for (const std::string& t : tokens) {
            if (stem == t || stem.rfind(t, 0) == 0) {
                return true;
            }
        }

        for (size_t i = 0; i < stem.size(); ++i) {
            if (stem[i] != ch) continue;

            bool leftOk = i == 0 || !std::isalnum(static_cast<unsigned char>(stem[i - 1]));
            bool rightOk = i + 1 == stem.size() || !std::isalnum(static_cast<unsigned char>(stem[i + 1]));
            if (leftOk && rightOk) {
                return true;
            }
        }
    }

    return false;
}

static int inferExpectedLabel(const fs::path& path, const std::vector<std::string>& classes) {
    for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
        if (classTokenInPath(path, classes[i])) {
            return i;
        }
    }
    return -1;
}

struct FingerCue {
    int fingertips = 0;
    int projectionPeaks = 0;
    int valleys = 0;
    float spread = 0.0f;
    float meanTipHeight = 0.0f;
    float meanValleyDepth = 0.0f;
    float vTipGap = 0.0f;
    float vTipAngle = 0.0f;
    float vValleyDrop = 0.0f;
    float vTipBalance = 1.0f;
    float vPairScore = 0.0f;
    float vPairHeight = 0.0f;
};

static bool readFingerCue(const cv::Mat& feat, FingerCue& cue) {
    constexpr int kFingertipCountIdx = 1777;
    constexpr int kProjectionPeaksIdx = 1778;
    constexpr int kValleyCountIdx = 1779;
    constexpr int kFingertipSpreadIdx = 1780;
    constexpr int kMeanTipHeightIdx = 1781;
    constexpr int kMeanValleyDepthIdx = 1782;
    constexpr int kVTipGapIdx = 1783;
    constexpr int kVTipAngleIdx = 1784;
    constexpr int kVValleyDropIdx = 1785;
    constexpr int kVTipBalanceIdx = 1786;
    constexpr int kVPairScoreIdx = 1787;
    constexpr int kVPairHeightIdx = 1788;

    if (feat.empty() || feat.type() != CV_32F || feat.rows != 1 || feat.cols <= kMeanValleyDepthIdx) {
        return false;
    }

    cue.fingertips = static_cast<int>(std::lround(feat.at<float>(0, kFingertipCountIdx) * 5.0f));
    cue.projectionPeaks = static_cast<int>(std::lround(feat.at<float>(0, kProjectionPeaksIdx) * 5.0f));
    cue.valleys = static_cast<int>(std::lround(feat.at<float>(0, kValleyCountIdx) * 5.0f));
    cue.spread = feat.at<float>(0, kFingertipSpreadIdx);
    cue.meanTipHeight = feat.at<float>(0, kMeanTipHeightIdx);
    cue.meanValleyDepth = feat.at<float>(0, kMeanValleyDepthIdx);

    if (feat.cols > kVPairHeightIdx) {
        cue.vTipGap = feat.at<float>(0, kVTipGapIdx);
        cue.vTipAngle = feat.at<float>(0, kVTipAngleIdx);
        cue.vValleyDrop = feat.at<float>(0, kVValleyDropIdx);
        cue.vTipBalance = feat.at<float>(0, kVTipBalanceIdx);
        cue.vPairScore = feat.at<float>(0, kVPairScoreIdx);
        cue.vPairHeight = feat.at<float>(0, kVPairHeightIdx);
    }

    return true;
}

static int classIndex(const std::vector<std::string>& classes, const std::string& name) {
    std::string needle = lowerString(name);

    for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
        if (lowerString(classes[i]) == needle) {
            return i;
        }
    }

    return -1;
}

static int applyFingerCueFallback(
    int pred,
    const cv::Mat& feat,
    const cv::Mat& normMask,
    double sourceBorderRatio,
    const std::vector<std::string>& classes
) {
    int aIdx = classIndex(classes, "A");
    int cIdx = classIndex(classes, "C");
    int fiveIdx = classIndex(classes, "Five");
    int vIdx = classIndex(classes, "V");

    if (fiveIdx < 0 || vIdx < 0) {
        return pred;
    }

    FingerCue cue;
    if (!readFingerCue(feat, cue)) {
        return pred;
    }

    double normBorder = normMask.empty() ? 0.0 : gesture::borderWhiteRatio(normMask);
    bool maskTouchesBorder = normBorder > 0.16 || sourceBorderRatio > 0.22;

    bool vFromDeepValley =
        cue.spread >= 0.62f &&
        cue.valleys >= 2 &&
        cue.meanValleyDepth >= 0.36f;

    bool vFromTwinPeakShape =
        (cue.fingertips <= 4 || sourceBorderRatio < 0.06) &&
        cue.spread >= 0.40f &&
        cue.spread <= 0.66f &&
        cue.meanTipHeight >= 0.24f &&
        cue.vPairHeight >= 0.40f &&
        cue.valleys >= 1 &&
        cue.valleys <= 3 &&
        cue.projectionPeaks <= 2;

    bool vFromPairGeometry =
        cue.vPairScore >= 0.70f &&
        cue.vTipGap >= 0.08f &&
        cue.vTipGap <= 0.22f &&
        cue.vTipAngle >= 0.06f &&
        cue.vTipAngle <= 0.58f &&
        cue.vValleyDrop >= 0.08f &&
        cue.vTipBalance <= 0.12f &&
        cue.vPairHeight >= 0.42f;

    bool vFromFiveConfusion =
        pred == fiveIdx &&
        cue.spread >= 0.40f &&
        cue.spread <= 0.64f &&
        cue.valleys <= 3 &&
        cue.projectionPeaks <= 2;

    if (pred == aIdx) {
        if (vFromDeepValley || vFromTwinPeakShape || vFromPairGeometry || vFromFiveConfusion) {
            return vIdx;
        }
    }

    bool isClosedClass = pred == aIdx || pred == cIdx;
    if (!isClosedClass) {
        bool weakFistAsFive =
            pred == fiveIdx &&
            cue.projectionPeaks <= 2 &&
            (
                (cue.spread >= 0.72f && cue.meanValleyDepth < 0.245f &&
                    (cue.meanTipHeight < 0.23f || cue.fingertips <= 3)) ||
                (cue.spread >= 0.78f && cue.meanTipHeight < 0.18f) ||
                (cue.spread >= 0.79f && cue.meanTipHeight < 0.24f && cue.projectionPeaks <= 1) ||
                (maskTouchesBorder && cue.projectionPeaks <= 2)
            );

        if (weakFistAsFive) {
            return aIdx >= 0 ? aIdx : pred;
        }
        return pred;
    }

    bool strongFive =
        !maskTouchesBorder &&
        cue.spread >= 0.42f &&
        (
            cue.projectionPeaks >= 4 ||
            (cue.fingertips >= 5 && cue.valleys >= 4 && cue.spread >= 0.65f &&
                cue.meanTipHeight >= 0.20f && cue.meanValleyDepth >= 0.245f) ||
            (cue.valleys >= 3 && cue.projectionPeaks >= 3 && cue.meanValleyDepth >= 0.245f)
        );

    if (strongFive) {
        return fiveIdx;
    }

    bool strongV =
        cue.spread >= 0.18f &&
        cue.spread <= 0.55f &&
        cue.meanTipHeight > 0.05f &&
        cue.valleys >= 1 &&
        (
            cue.fingertips == 2 ||
            cue.projectionPeaks == 2 ||
            (cue.fingertips == 3 && cue.projectionPeaks == 2)
        );

    if (strongV) {
        return vIdx;
    }

    return pred;
}

int main(int argc, char** argv) {
    /*
    默认参数：
    1. 测试图片文件夹默认为 ./test_images
    2. 模型文件默认为 gesture_knn.yml

    也可以通过命令行指定：
    test_gesture <测试图片文件夹路径> <模型文件路径>
    */
    fs::path testDir = "./test_images";
    std::string modelFile = "gesture_knn.yml";
    fs::path debugMaskDir;

    if (argc >= 2) {
        testDir = argv[1];
    }

    if (argc >= 3) {
        modelFile = argv[2];
    }

    if (argc >= 4) {
        debugMaskDir = argv[3];
        std::error_code ec;
        fs::create_directories(debugMaskDir, ec);
        if (ec) {
            std::cerr << "[Warning] Cannot create debug mask folder: "
                      << debugMaskDir.string() << "\n";
            debugMaskDir.clear();
        }
    }

    /*
    加载训练好的模型文件。

    模型文件由 train.cpp 生成，
    其中保存了训练样本特征、标签、类别名称、均值和标准差。
    */
    cv::FileStorage fsin(modelFile, cv::FileStorage::READ);

    if (!fsin.isOpened()) {
        std::cerr << "[Error] Cannot open model file: " << modelFile << "\n";
        std::cerr << "Usage:\n"
                  << "  test_gesture <test_image_folder> <gesture_knn.yml> [debug_mask_folder]\n\n"
                  << "Example:\n"
                  << "  test_gesture ../test_images ../gesture_knn.yml ../debug_masks\n";
        return 1;
    }

    int k = 7;
    int featureDim = 0;

    fsin["k"] >> k;
    fsin["feature_dim"] >> featureDim;

    std::vector<std::string> classes = readClassNames(fsin["classes"]);

    cv::Mat mean, stddev, trainX, trainLabels;

    fsin["mean"] >> mean;
    fsin["stddev"] >> stddev;
    fsin["train_features"] >> trainX;
    fsin["train_labels"] >> trainLabels;

    fsin.release();

    /*
    检查模型文件是否完整有效
    */
    if (classes.empty() || mean.empty() || stddev.empty() ||
        trainX.empty() || trainLabels.empty()) {
        std::cerr << "[Error] Invalid model file.\n";
        return 1;
    }

    // 读取测试文件夹中的所有 PNG 图像。
    std::vector<fs::path> files = listPngFiles(testDir);

    if (files.empty()) {
        std::cerr << "[Warning] No valid images found in folder: " << testDir << "\n";
        return 0;
    }

    /*
    对每张测试图像依次进行：
    1. 图像读取；
    2. 手部分割；
    3. 特征提取；
    4. 特征标准化；
    5. 加权 KNN 预测；
    6. 输出预测类别。
    */
    for (const auto& p : files) {
        cv::Mat img = imreadAnyPath(p, cv::IMREAD_UNCHANGED);

        if (img.empty()) {
            std::cerr << "[Warning] Cannot read image: " << p << "\n";
            continue;
        }

        cv::Mat sourceMask = gesture::segmentHand(img);
        double sourceBorderRatio = gesture::borderWhiteRatio(sourceMask);

        cv::Mat feat;
        cv::Mat debugNormMask;

        // 提取与训练阶段完全相同的传统视觉特征。
        bool ok = gesture::extractFeature(
            img,
            feat,
            &debugNormMask
        );

        if (!ok || feat.empty()) {
            std::cerr << "[Warning] Feature extraction failed, use zero feature fallback: "
                      << p.filename().string() << "\n";
            feat = cv::Mat::zeros(1, mean.cols, CV_32F);
        }

        /*
        检查测试特征维度是否与训练特征一致。
        如果不一致，说明模型文件或特征提取代码不匹配。
        */
        if (feat.cols != mean.cols) {
            std::cerr << "[Warning] Feature dimension mismatch: "
                      << p.filename().string() << "\n";
            feat = cv::Mat::zeros(1, mean.cols, CV_32F);
        }

        // 使用训练阶段保存的 mean 和 stddev 对测试特征进行标准化。
        cv::Mat featStd = gesture::standardize(feat, mean, stddev);

        // 使用加权 KNN 分类器预测类别编号。
        int pred = gesture::predictWeightedKNN(
            trainX,
            trainLabels,
            featStd,
            k,
            static_cast<int>(classes.size())
        );

        if (pred < 0 || pred >= static_cast<int>(classes.size())) {
            pred = 0;
        }

        pred = applyFingerCueFallback(pred, feat, debugNormMask, sourceBorderRatio, classes);

        if (!debugMaskDir.empty() && !debugNormMask.empty()) {
            int expected = inferExpectedLabel(p, classes);
            bool shouldSave = expected < 0 || expected != pred;

            if (shouldSave) {
                std::string expectedName = expected >= 0 ? classes[expected] : "Unknown";
                std::string outName = expectedName + "_as_" + classes[pred] + "_" + p.stem().string() + ".png";
                fs::path outPath = debugMaskDir / sanitizeFileName(outName);

                if (!cv::imwrite(outPath.string(), debugNormMask)) {
                    std::cerr << "[Warning] Cannot write debug mask: "
                              << outPath.string() << "\n";
                }

                FingerCue cue;
                if (readFingerCue(feat, cue)) {
                    std::cerr << "[Debug] " << p.filename().string()
                              << " expected=" << expectedName
                              << " pred=" << classes[pred]
                              << " tips=" << cue.fingertips
                              << " projection_peaks=" << cue.projectionPeaks
                              << " valleys=" << cue.valleys
                              << " spread=" << cue.spread
                              << " tip_height=" << cue.meanTipHeight
                              << " valley_depth=" << cue.meanValleyDepth
                              << " v_gap=" << cue.vTipGap
                              << " v_angle=" << cue.vTipAngle
                              << " v_drop=" << cue.vValleyDrop
                              << " v_balance=" << cue.vTipBalance
                              << " v_pair=" << cue.vPairScore
                              << " v_height=" << cue.vPairHeight
                              << " border=" << gesture::borderWhiteRatio(debugNormMask)
                              << " source_border=" << sourceBorderRatio
                              << "\n";
                }
            }
        }

        /*
        按照作业要求打印每张 PNG 图像的识别结果。

        输出格式：
        图片文件名 类别

        例如：
        001.png A
        002.png C
        003.png Five
        004.png V
        */
        std::cout << p.filename().string() << " " << classes[pred] << std::endl;
    }

    return 0;
}
