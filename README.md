# EuroSAT MLP Classifier 交付说明

## 内容说明

本压缩包包含 EuroSAT_RGB 三层 MLP 图像分类作业的代码、实验报告和实验输出说明。由于数据集和模型权重较大，需要按下文说明手动准备。

主要内容：

- `src/`：纯 NumPy 三层 MLP 实现，包括数据加载、模型、训练、评估、超参数搜索和可视化。
- `report/HW1_EuroSAT_MLP_Report.md`：Markdown 实验报告。
- `outputs/checkpoints/best_model.npz`：训练好的最佳模型权重。该文件不直接包含在项目中，需要从云盘下载后放到对应位置。
- `outputs/test_metrics.json`：最终测试集指标和混淆矩阵。
- `outputs/figures/`：报告使用的训练曲线、混淆矩阵、错误样本和权重可视化图片。
- `requirements.txt`：运行依赖。
- `hw1.zip`：作业初始数据集压缩包。由于数据集较大，本项目不直接包含该文件，需要手动放到项目根目录。

## 最终结果

最终模型配置：

- 输入尺寸：`64 x 64 x 3`
- 隐藏层：`512 / 256`
- 激活函数：ReLU
- Epochs：`130`
- Batch Size：`128`
- Learning Rate：`0.007`
- Learning Rate Decay：`0.993`
- Weight Decay：`0.00005`

测试集结果：

- Test Loss：`1.0800`
- Test Accuracy：`0.6235`

## 数据集准备

由于 EuroSAT_RGB 数据集较大，本项目不直接包含数据集压缩包。请将课程作业提供的初始压缩包 `hw1.zip` 手动放到项目根目录：

```text
eurosat-mlp-classifier-submit/
├── hw1.zip
├── src/
├── outputs/
├── report/
└── README.md
```

训练或评估脚本首次运行时会自动将 `hw1.zip` 解压为 `EuroSAT_RGB/` 目录。也可以提前手动解压，只要根目录下存在 `EuroSAT_RGB/` 即可。

## 模型权重准备

训练好的最佳模型权重请从 Google Drive 下载：

[best_model.npz](https://drive.google.com/file/d/1RXhOvgxwVxSIqWALHtnhtih3EAq8gh6w/view?usp=sharing)

下载后请将文件放到以下位置：

```text
outputs/checkpoints/best_model.npz
```

放置完成后，可以直接使用 README 后面的评估和可视化命令复现实验结果。

## 环境准备

Windows 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux / macOS 示例：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 使用方法

评估已训练模型：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint outputs\checkpoints\best_model.npz --output-dir outputs
```

重新生成可视化图片：

```powershell
.\.venv\Scripts\python.exe -m src.visualize --checkpoint outputs\checkpoints\best_model.npz --history outputs\logs\history.json --metrics outputs\test_metrics.json --output-dir outputs\figures
```

重新训练基线模型：

```powershell
.\.venv\Scripts\python.exe -m src.train --image-size 64 --hidden-dim1 512 --hidden-dim2 256 --epochs 130 --learning-rate 0.007 --lr-decay 0.993 --weight-decay 0.00005 --cache
```

## 注意事项

- 本项目不使用 PyTorch、TensorFlow、JAX 等自动微分框架。
- 本项目不附带 `hw1.zip`；请先将课程作业提供的初始压缩包放到项目根目录，首次运行时会自动解压为 `EuroSAT_RGB/`。
- 本项目不附带模型权重文件；请从 Google Drive 下载 `best_model.npz` 并放到 `outputs/checkpoints/` 目录下。
- MLP 会将图像展平成一维向量，因此性能通常低于 CNN，但符合本次作业从零实现三层神经网络分类器的要求。
