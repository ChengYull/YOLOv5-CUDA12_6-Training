# YOLOv5 Custom Training with CUDA 12.6 (RTX 2050)

## 基于 CUDA 12.6 和 RTX 2050 的 YOLOv5 自定义目标检测训练

训练结果示例：检测自定义目标（Doro 头像）
![](https://img2024.cnblogs.com/blog/2734270/202504/2734270-20250425105421664-653968760.gif)

## 项目简介
本项目使用 YOLOv5 在 CUDA 12.6 + RTX 2050 环境下训练自定义目标检测模型，包含：
- 完整环境配置指南（Anaconda + CUDA + PyTorch）
- 数据集标注教程（LabelImg 工具）
- YOLOv5 模型训练代码（支持自定义数据）
- 模型测试脚本（图片/视频/摄像头实时检测）

## 硬件要求
- NVIDIA GPU（支持 CUDA 12.6+，如 RTX 2050/3060 等）

## 部署

### 安装依赖
```bash
pip install -r requirements.txt
```

### 数据集准备
- 图片标注
- 使用 LabelImg 工具进行数据集标注，生成 YOLO 格式的标签文件。
    - 数据集目录结构示例：
    ```
  train/
  ├── images/  # 存放图片
  ├── labels/  # 存放标注文件
  └── doro.yaml  # 数据集配置文件
  ```

### 训练
执行训练脚本：src/test_train.py

### 测试
执行测试脚本：src/test_doro.py

## 详细教程

博客园地址：[https://www.cnblogs.com/1873cy/p/18844467](https://www.cnblogs.com/1873cy/p/18844467)