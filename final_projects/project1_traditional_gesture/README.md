# 传统视觉手势识别项目

本项目用于识别四类静态手势：`A`、`C`、`Five`、`V`。最终版本采用传统机器视觉特征加加权 KNN 分类，不依赖深度学习，也不保留 SVM 试验代码。

## 目录结构

```text
.
├── CMakeLists.txt
├── include/
│   └── gesture_features.hpp
├── src/
│   ├── train.cpp
│   └── test.cpp
├── Hand_Posture_Easy_Stu/
│   ├── A/
│   ├── C/
│   ├── Five/
│   └── V/
└── gesture_knn.yml
```

主要文件说明：

- `include/gesture_features.hpp`：手部分割、归一化、特征提取、特征标准化、加权 KNN。
- `src/train.cpp`：读取 `Hand_Posture_Easy_Stu`，提取特征，训练并保存模型。
- `src/test.cpp`：单独的测试代码，加载训练好的模型文件，对指定测试文件夹内的图片进行识别。
- `gesture_knn.yml`：训练得到的模型文件，包含类别名、标准化参数、训练特征、训练标签和 `k`。

## 作业要求对应说明

本项目满足“提交的程序代码中必须有单独的测试代码及测试代码中要加载的训练好的分类模型文件”的要求：

- 单独测试代码：`src/test.cpp`，编译后为 `build\Release\test_gesture.exe`。
- 训练好的分类模型文件：`gesture_knn.yml`。
- 测试程序运行时会加载 `gesture_knn.yml`，不需要重新训练。
- 通过运行测试程序，可以直接对一个文件夹中的所有 PNG 图像进行手势识别，并依次打印每个图像的识别结果：`A`、`C`、`Five` 或 `V`。

典型测试命令如下：

```powershell
build\Release\test_gesture.exe <测试图片文件夹> gesture_knn.yml
```

输出格式为：

```text
image001.png A
image002.png C
image003.png Five
image004.png V
```

## 环境要求

- CMake 3.10 或更高版本
- C++17 编译器
- OpenCV

在 Windows + Visual Studio 环境下可以直接使用 CMake 生成并编译。

## 编译

```powershell
cmake -S . -B build
cmake --build build --config Release
```

编译后会生成：

```text
build\Release\train_gesture.exe
build\Release\test_gesture.exe
```

## 训练

最终推荐使用 `k=9`，并在完整的 `Hand_Posture_Easy_Stu` 数据集上重新训练：

```powershell
build\Release\train_gesture.exe Hand_Posture_Easy_Stu gesture_knn.yml 9
```

如果不显式传入 `k`，训练程序会在若干候选值中做 5 折交叉验证并自动选择。但为了提交版本稳定复现，本项目固定使用：

```text
k = 9
```

训练阶段会对每张原图做轻微旋转增强：

```text
-10 度、-5 度、0 度、5 度、10 度
```

当前数据集每类 50 张原图，共 200 张原图。增强后训练样本数为 1000。

## 测试

测试指定文件夹中的图像：

```powershell
build\Release\test_gesture.exe <测试图片文件夹> gesture_knn.yml
```

程序会递归读取测试文件夹中的图片。按作业要求，测试程序可以直接识别文件夹内所有 `.png` 图像；当前实现也额外兼容 `.jpg` 和 `.jpeg`：

```text
.png .jpg .jpeg
```

输出格式为：

```text
filename.png A
filename.png C
filename.png Five
filename.png V
```

## Debug Mask 输出

测试程序支持输出失败样本的归一化 mask，方便判断错误来自分割还是分类。

```powershell
build\Release\test_gesture.exe <测试图片文件夹> gesture_knn.yml debug_masks
```

如果测试图片路径或文件名能推断真实类别，程序会只保存预测错误的样本 mask。保存文件名形如：

```text
expected_as_predicted_originalname.png
```

这个功能主要用于观察手是否被正确分割。例如 `Five` 或 `V` 如果被分割成了类似 `A`、`C` 的轮廓，说明主要问题在分割；如果 mask 很清晰，则问题更可能在特征或分类。

## 算法流程

整体流程如下：

```text
输入图片
  -> 读取图片
  -> 多候选手部分割
  -> mask 后处理
  -> 归一化到 128x128
  -> 提取传统视觉特征
  -> 特征标准化
  -> 加权 KNN 分类
  -> 手指规则微调
  -> 输出类别
```

### 1. 图片读取

训练和测试都使用二进制读取加 `cv::imdecode` 的方式加载图片。这样比直接 `cv::imread` 更适合处理中文路径或特殊路径。

### 2. 多候选手部分割

手势识别最关键的一步是得到稳定的手部二值 mask。项目里不是只使用一种阈值，而是生成多个候选 mask，再给每个候选打分，选择最可信的一个。

候选来源包括：

- 透明通道：如果输入图像带 alpha，则优先利用 alpha 得到前景。
- 边界背景差分：估计图片边界背景颜色，根据像素到背景的颜色距离分割前景。
- 肤色分割：结合 HSV 与 YCrCb 的肤色范围。
- 灰度分割：用于背景和手部明暗差比较明显的情况。

候选 mask 会根据以下指标打分：

- 前景面积比例是否合理
- 前景是否过多贴边
- 外接矩形是否过宽或过高
- extent、solidity、compactness 是否异常
- 面积是否接近手势图像常见范围

这样做的目的，是让程序在背景不同、光照不同、是否有透明通道不同的图片上都能尽量选到合理的手部区域。

### 3. Mask 后处理和归一化

分割得到的 mask 会经过：

- 二值化
- 形态学开闭运算
- 取最大连通域
- 填洞
- 裁剪手部外接框
- 等比例缩放到 `128x128`
- 居中放置

最终所有样本都会被转换成统一尺寸的手部二值图，后续特征都基于这个归一化 mask 计算。

### 4. 特征提取

最终特征维度为 `1895`。特征由多组传统视觉特征拼接而成。

#### HOG 特征

HOG 用来描述手势整体边缘方向分布。参数为：

```text
winSize     = 128x128
blockSize   = 32x32
blockStride = 16x16
cellSize    = 16x16
bins        = 9
```

HOG 对 `A`、`C` 这类整体形状差异有帮助，也能给 `Five`、`V` 提供边缘方向信息。

#### 轮廓几何特征

从最大轮廓中提取：

- 面积比例
- 周长比例
- 外接框宽高比
- 外接框宽度和高度
- extent
- solidity
- circularity
- 凸包面积和凸包周长

这些特征主要描述手势的整体形状。例如 `A` 更紧凑，`C` 有较明显弯曲轮廓，`Five` 和 `V` 的凸包与轮廓关系更复杂。

#### 凸缺陷和手指特征

为了强化 `Five` 和 `V`，项目加入了更直接的手指数量特征：

- 凸缺陷数量
- 最大凸缺陷深度
- 平均凸缺陷深度
- 指尖候选数量
- 上半区垂直投影峰值数量
- 指缝 valley 数量
- 指尖横向展开程度
- 指尖平均高度
- 指缝平均深度

直观上：

- `Five` 通常有更多指尖峰值和多个指缝。
- `V` 通常有两个主要指尖，并且两个指尖之间有明显 valley。
- `A` 和 `C` 的有效指尖峰值通常更少。

#### V 双指角度特征

`V` 容易和 `A`、`Five` 混淆，所以单独加入了双指结构特征：

- 两个主要指尖之间的距离
- 两指夹角
- 两指之间 valley 的下落深度
- 两指高度是否平衡
- 双指匹配分数
- 双指整体高度

这些特征用来描述典型 `V` 手势的结构：两个上方指尖分开，中间有指缝，角度和间距落在合理范围内。

#### Hu 矩、投影和网格特征

项目还加入：

- 质心位置
- 偏心率
- 7 个 Hu 不变矩
- 16 维水平投影
- 16 维垂直投影
- 8x8 网格占空比

投影和网格特征可以保留局部空间分布。例如上半部分有几段前景、左右是否展开、下半部分是否集中等。

### 5. 特征标准化

训练阶段会计算每一维特征的均值和标准差：

```text
x' = (x - mean) / stddev
```

测试阶段使用训练时保存的同一组 `mean` 和 `stddev` 标准化特征，保证训练和测试处于同一尺度。

### 6. 加权 KNN 分类

最终分类器为加权 KNN。流程是：

1. 计算测试样本到所有训练样本的欧氏距离。
2. 取最近的 `k=9` 个训练样本。
3. 对这 9 个邻居按距离加权投票。
4. 权重公式为：

```text
weight = 1 / (distance + 1e-6)
```

距离越近的样本投票权重越高。相比普通 KNN，这能减少较远邻居对结果的干扰。

### 7. 手指规则微调

KNN 输出后，测试程序会根据手指特征做少量后处理，主要用于减少 `Five` 和 `V` 的明显误判：

- 当手指峰值、投影峰值、指缝数量都明显符合 `Five` 时，可以把弱置信的其它类别修正为 `Five`。
- 当双指距离、角度、valley 和高度都明显符合 `V` 时，可以把弱置信的其它类别修正为 `V`。
- 对于边界贴边、面积异常等分割风险较高的样本，规则会更保守。

这些规则不是替代 KNN，而是补充 KNN 在 `Five`、`V` 上容易缺少结构理解的问题。

## 为什么使用 k=9

在 `Hand_Posture_Easy_Stu` 原数据集上做拆分验证时，`k=9` 在稳定性和准确率之间表现最好。此前测试过的结果如下：

| k | 5 折验证准确率 | 错误数 |
|---:|---:|---:|
| 1 | 92.5% | 15 / 200 |
| 3 | 94.0% | 12 / 200 |
| 5 | 94.0% | 12 / 200 |
| 7 | 95.0% | 10 / 200 |
| 9 | 95.5% | 9 / 200 |
| 11 | 95.5% | 9 / 200 |
| 15 | 95.0% | 10 / 200 |
| 21 | 95.0% | 10 / 200 |
| 25 | 94.5% | 11 / 200 |
| 31 | 93.0% | 14 / 200 |

`k=9` 和 `k=11` 的验证准确率相同，但 `k=9` 更小，能保留更多局部邻域差异，因此最终采用 `k=9`。

## 当前模型复现结果

本版本已经使用完整 `Hand_Posture_Easy_Stu` 重新训练：

```text
Original images loaded: 200
Augmented training samples: 1000
Feature dimension: 1895
KNN k: 9
```

训练完成后，对完整 `Hand_Posture_Easy_Stu` 做了一次回测，结果为：

| 类别 | 样本数 | 错误数 |
|---|---:|---:|
| A | 50 | 0 |
| C | 50 | 0 |
| Five | 50 | 0 |
| V | 50 | 0 |
| 合计 | 200 | 0 |

这个结果用于确认训练产物、特征维度和测试程序完全匹配。它不是独立测试集准确率，不能单独证明模型完全没有过拟合。结合 5 折交叉验证结果看，`k=9` 的验证准确率为 `95.5%`，低于完整训练集回测的 `100%`，但差距不大；由于老师最终测试与训练数据同源，当前模型的过拟合风险可接受。

## 模型文件内容

`gesture_knn.yml` 中保存：

- `model_type`：模型类型，当前为 `TraditionalGestureWeightedKNN`
- `description`：模型说明
- `k`：KNN 邻居数量，当前为 `9`
- `norm_size`：归一化尺寸，当前为 `128`
- `feature_dim`：特征维度，当前为 `1895`
- `classes`：类别名，顺序为 `A`、`C`、`Five`、`V`
- `mean`：训练集特征均值
- `stddev`：训练集特征标准差
- `train_features`：标准化后的训练特征
- `train_labels`：训练标签

测试程序会直接读取这些字段完成识别。

## 数据集和泛化说明

本项目最终目标是老师提供的同源测试集，因此训练和参数选择以 `Hand_Posture_Easy_Stu` 为主。若测试集改为跨域图片，可能会遇到明显分布差异：

- 背景不同
- 光照不同
- 手部肤色和边缘质量不同
- 图片裁剪比例不同
- 是否透明背景不同
- 手势姿态和数据集标准不完全一致

如果老师的测试集来自同一套采集规则，那么使用 `Hand_Posture_Easy_Stu` 拆分验证得到的 `k=9` 更有参考价值。如果目标变成识别任意跨域图片，则需要继续增强分割、扩充训练数据，甚至考虑更强的学习方法。

## 当前方法的优点和限制

优点：

- 不需要深度学习训练环境。
- 模型文件可解释，包含明确的传统视觉特征。
- 对同源数据集表现稳定。
- Debug mask 能直接定位分割错误。
- `Five` 和 `V` 有针对性的手指数量和双指角度特征。

限制：

- 对极复杂背景仍然依赖分割质量。
- 对严重遮挡、旋转过大、手势不标准的图片鲁棒性有限。
- 如果测试图片分布和训练集差异很大，准确率会下降。
- 后处理规则是经验规则，适合当前四分类任务，不适合直接泛化到很多新类别。

## 推荐提交流程

提交前建议按下面顺序检查：

```powershell
cmake --build build --config Release
build\Release\train_gesture.exe Hand_Posture_Easy_Stu gesture_knn.yml 9
build\Release\test_gesture.exe Hand_Posture_Easy_Stu gesture_knn.yml
```

如果需要观察错误样本 mask：

```powershell
build\Release\test_gesture.exe Hand_Posture_Easy_Stu gesture_knn.yml debug_masks_verify
```

如果老师提供一个独立测试文件夹，则直接运行：

```powershell
build\Release\test_gesture.exe <老师测试图片文件夹> gesture_knn.yml
```

最终提交时保留：

- `src/train.cpp`
- `src/test.cpp`
- `include/gesture_features.hpp`
- `CMakeLists.txt`
- `README.md`
- 重新训练得到的 `gesture_knn.yml`
