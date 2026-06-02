# 👁️ 机器视觉课程编程作业 (Machine Vision Assignments)

**南开大学 | 智能科学与技术专业**

本仓库用于记录和管理“机器视觉”课程的编程作业、实验代码以及相关的测试结果。

## 📂 仓库结构

项目基于 CMake 构建，源码与编译文件分离。具体目录结构如下：

```text
machine_vision
│
├─ CMakeLists.txt        # CMake 构建配置总文件，定义编译规则与依赖项
├─ src                   # 源代码目录，存放各次作业的 .cpp 和头文件
├─ data                  # 数据目录，用于存放测试图像、视频及程序输出结果
├─ final_projects        # 课程大作业，作为独立子项目保存
├─ build                 # 编译输出目录
├─ readme.md             # 本仓库说明文档
└─ .vscode               # VS Code 工作区配置文件
```

## 📌 课程大作业

两个大作业已整理到 `final_projects/` 目录下，每个子目录都保留独立的 README、源码、报告和必要模型文件。

```text
final_projects/
├─ project1_traditional_gesture/   # 大作业 1：传统视觉四分类手势识别
└─ project2_deep_gesture/          # 大作业 2：ResNet18 六分类手势识别
```

### 大作业 1：传统视觉手势识别

目录：`final_projects/project1_traditional_gesture/`

- 任务：识别 `A`、`C`、`Five`、`V` 四类静态手势。
- 方法：多候选手部分割、HOG、轮廓几何、手指结构特征和加权 KNN。
- 测试代码：`src/test.cpp`，编译后为 `test_gesture.exe`。
- 训练好的模型：`gesture_knn.yml`。
- 报告：`part1_traditional_gesture_report.pdf`。

测试命令示例：

```powershell
build\Release\test_gesture.exe <测试图片文件夹> gesture_knn.yml
```

### 大作业 2：深度学习手势识别

目录：`final_projects/project2_deep_gesture/`

- 任务：识别 `A`、`B`、`C`、`Five`、`Point`、`V` 六类静态手势。
- 方法：ResNet18 迁移学习分类器。
- 测试代码：`test.py`。
- 训练好的模型：`checkpoints/best_model.pth`。
- 报告：`experiment_report.pdf`。

测试命令示例：

```powershell
python test.py --image_dir <PNG测试图片文件夹> --model_path checkpoints/best_model.pth
```

说明：大作业 2 的完整训练数据集和训练过程输出未放入仓库，只保留必要模型、源码、报告和数据目录占位文件。若需要重新训练，请按子项目 README 准备数据。
