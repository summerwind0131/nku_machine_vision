#include "gesture_features.hpp"

#include <opencv2/opencv.hpp>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

/*
训练程序说明：

功能：
1. 读取训练数据集 Hand_Posture_Easy_Stu；
2. 依次处理 A、C、Five、V 四类手势图像；
3. 对每张图像进行数据增强；
4. 提取传统视觉特征；
5. 计算特征均值和标准差；
6. 保存训练好的加权 KNN 模型到 gesture_knn.yml。

注意：
该程序只需要训练时运行一次。
最终提交时，测试程序会直接加载训练生成的 gesture_knn.yml。
*/

namespace fs = std::filesystem;

// 四分类任务
static const std::vector<std::string> CLASSES = {
    "A", "C", "Five", "V"
};

/*
将字符串转换为小写。
*/
static std::string lowerString(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

/*
使用二进制方式读取图像文件，再通过 imdecode 解码。
*/
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

/*
列出指定目录下的所有 PNG 文件。

recursive = true 时，会递归读取子目录。
训练阶段建议递归读取，防止数据集内部还有子文件夹。
*/
static std::vector<fs::path> listPngFiles(const fs::path& dir, bool recursive = true) {
    std::vector<fs::path> files;

    if (!fs::exists(dir)) return files;

    if (recursive) {
        for (const auto& e : fs::recursive_directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;

            std::string ext = lowerString(e.path().extension().string());
            if (ext == ".png") {
                files.push_back(e.path());
            }
        }
    } else {
        for (const auto& e : fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;

            std::string ext = lowerString(e.path().extension().string());
            if (ext == ".png") {
                files.push_back(e.path());
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

/*
对图像进行小角度旋转，保持输出尺寸不变。
进行数据增强，提高模型对手势轻微旋转的鲁棒性。
*/
static cv::Mat rotateImageKeepSize(const cv::Mat& src, double angleDeg) {
    if (src.empty()) return cv::Mat();

    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat M = cv::getRotationMatrix2D(center, angleDeg, 1.0);

    cv::Mat dst;
    cv::warpAffine(
        src,
        dst,
        M,
        src.size(),
        cv::INTER_LINEAR,
        cv::BORDER_REPLICATE
    );

    return dst;
}

/*
生成增强样本。

每张训练图像扩展为多个轻微旋转版本：
-10°、-5°、0°、5°、10°。

这样可以提升 KNN 对不同手势角度的适应能力。
*/
static std::vector<cv::Mat> makeAugmentations(const cv::Mat& img) {
    std::vector<cv::Mat> out;

    std::vector<double> angles = {-10.0, -5.0, 0.0, 5.0, 10.0};

    for (double a : angles) {
        if (std::abs(a) < 1e-6) {
            out.push_back(img.clone());
        } else {
            out.push_back(rotateImageKeepSize(img, a));
        }
    }

    return out;
}

/*
利用交叉验证对 KNN 的超参数 k 进行调优。
这里使用的是 5 折交叉验证。
*/
static int tuneHyperparameterK(const cv::Mat& X, const cv::Mat& Y, int numClasses, const std::vector<int>& candidateKs, int folds = 5) {
    if (X.rows < folds) return candidateKs[0];

    int bestK = candidateKs[0];
    double bestAcc = -1.0;

    std::vector<int> indices(X.rows);
    std::iota(indices.begin(), indices.end(), 0);

    // 固定随机种子以保证结果可复现
    std::mt19937 g(42);
    std::shuffle(indices.begin(), indices.end(), g);

    int foldSize = X.rows / folds;

    std::cout << "\n--- K Hyperparameter Tuning (" << folds << "-fold cross validation) ---\n";
    for (int k : candidateKs) {
        double totalAcc = 0.0;

        // 五折交叉验证
        for (int f = 0; f < folds; ++f) {
            int valStart = f * foldSize;
            int valEnd = (f == folds - 1) ? X.rows : (f + 1) * foldSize;

            cv::Mat trainX, trainY, valX, valY;
            for (int i = 0; i < X.rows; ++i) {
                int idx = indices[i];
                if (i >= valStart && i < valEnd) {
                    valX.push_back(X.row(idx));
                    valY.push_back(Y.row(idx));
                } else {
                    trainX.push_back(X.row(idx));
                    trainY.push_back(Y.row(idx));
                }
            }

            int correct = 0;
            for (int i = 0; i < valX.rows; ++i) {
                int pred = gesture::predictWeightedKNN(trainX, trainY, valX.row(i), k, numClasses);
                if (pred == valY.at<int>(i, 0)) {
                    correct++;
                }
            }
            totalAcc += static_cast<double>(correct) / valX.rows;
        }

        double avgAcc = totalAcc / folds;
        std::cout << "  [k=" << k << "] Avg Accuracy: " << (avgAcc * 100.0) << "%\n";

        if (avgAcc > bestAcc) {
            bestAcc = avgAcc;
            bestK = k;
        }
    }

    std::cout << "  => Optimal k selected: " << bestK << " (Validation Accuracy: " << (bestAcc * 100.0) << "%)\n";
    std::cout << "--------------------------------------------------------\n\n";

    return bestK;
}

int main(int argc, char** argv) {
    /*
    命令行参数说明：

    argv[1]：训练数据集根目录，例如 ../Hand_Posture_Easy_Stu
    argv[2]：输出模型文件路径，例如 ../gesture_knn.yml
    argv[3]：KNN 中的 k 值，可选，默认 7
    */
    if (argc < 3) {
        std::cout << "Usage:\n"
                  << "  train_gesture <Hand_Posture_Easy_Stu_root> <output_model.yml> [k]\n\n"
                  << "Example:\n"
                  << "  train_gesture ../Hand_Posture_Easy_Stu ../gesture_knn.yml 9\n";
        return 0;
    }

    fs::path datasetRoot = argv[1];
    std::string modelFile = argv[2];

    int k = 7;
    if (argc >= 4) {
        k = std::max(1, std::atoi(argv[3]));
    }

    // trainX 保存所有训练样本的特征，每一行对应一张增强后的图像。
    cv::Mat trainX;

    // trainYVec 保存每个训练样本对应的类别标签。
    std::vector<int> trainYVec;

    int totalOriginal = 0;
    int totalSuccess = 0;

    std::cout << "Training classes:\n";

    /*
    依次读取 A、C、Five、V 四个类别文件夹。
    目录结构为：
    Hand_Posture_Easy_Stu/
    ├── A/
    ├── C/
    ├── Five/
    └── V/
    */
    for (int label = 0; label < static_cast<int>(CLASSES.size()); ++label) {
        const std::string& cls = CLASSES[label];
        fs::path clsDir = datasetRoot / cls;

        std::vector<fs::path> files = listPngFiles(clsDir, true);

        std::cout << "  " << cls << ": " << files.size() << " PNG files\n";

        for (const auto& p : files) {
            cv::Mat img = imreadAnyPath(p, cv::IMREAD_UNCHANGED);

            if (img.empty()) {
                std::cerr << "[Warning] Cannot read image: " << p << "\n";
                continue;
            }

            ++totalOriginal;

            // 对原图生成多个轻微旋转版本。
            std::vector<cv::Mat> augImgs = makeAugmentations(img);

            for (const cv::Mat& aug : augImgs) {
                cv::Mat feat;

                // 提取传统视觉特征。
                bool ok = gesture::extractFeature(aug, feat);

                if (!ok || feat.empty()) {
                    std::cerr << "[Warning] Feature extraction failed: " << p << "\n";
                    continue;
                }

                trainX.push_back(feat);
                trainYVec.push_back(label);
                ++totalSuccess;
            }
        }
    }

    if (trainX.empty() || trainYVec.empty()) {
        std::cerr << "[Error] No training data extracted. Please check dataset path.\n";
        return 1;
    }

    // 将标签 vector 转换为 OpenCV Mat，方便保存到 yml 模型文件中。
    cv::Mat trainLabels(static_cast<int>(trainYVec.size()), 1, CV_32S);
    for (int i = 0; i < static_cast<int>(trainYVec.size()); ++i) {
        trainLabels.at<int>(i, 0) = trainYVec[i];
    }

    /*
    计算训练集特征的均值和标准差，
    并对所有训练特征进行标准化。
    */
    cv::Mat mean, stddev;
    gesture::computeScaler(trainX, mean, stddev);

    cv::Mat trainXStd = gesture::standardize(trainX, mean, stddev);

    /*
    如果用户输入参数里 k = 0 或者在代码里我们强制调优，
    这里将自动在 {1, 3, 5, 7, 9, 11, 15} 中寻找最佳 k。
    */
    if (k <= 0 || argc < 4) {
        std::cout << "Starting automatic hyperparameter K tuning...\n";
        std::vector<int> candidateKs = {1, 3, 5, 7, 9, 11, 15};
        k = tuneHyperparameterK(trainXStd, trainLabels, static_cast<int>(CLASSES.size()), candidateKs, 5);
    }

    cv::FileStorage fsout(modelFile, cv::FileStorage::WRITE);

    if (!fsout.isOpened()) {
        std::cerr << "[Error] Cannot open output model file: " << modelFile << "\n";
        return 1;
    }

    fsout << "model_type" << "TraditionalGestureWeightedKNN";
    fsout << "description" << "Part1 traditional vision: segmentation + HOG + contour/finger features + weighted KNN";
    fsout << "k" << k;
    fsout << "norm_size" << gesture::NORM_SIZE;
    fsout << "feature_dim" << trainX.cols;

    fsout << "classes" << "[";
    for (const std::string& cls : CLASSES) {
        fsout << cls;
    }
    fsout << "]";

    fsout << "mean" << mean;
    fsout << "stddev" << stddev;
    fsout << "train_features" << trainXStd;
    fsout << "train_labels" << trainLabels;

    fsout.release();

    std::cout << "\nTraining finished.\n";
    std::cout << "Original images loaded: " << totalOriginal << "\n";
    std::cout << "Augmented training samples: " << totalSuccess << "\n";
    std::cout << "Feature dimension: " << trainX.cols << "\n";
    std::cout << "KNN k: " << k << "\n";
    std::cout << "Model saved to: " << modelFile << "\n";

    return 0;
}
