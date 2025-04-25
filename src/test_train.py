from train import parse_opt, main
import torch
# 在训练前清理GPU缓存
torch.cuda.empty_cache()
def custom_train():
    opt = parse_opt()
    opt.data = "../train/doro.yaml"  # 数据集路径
    opt.weights = "../yolov5s.pt"     # 权重文件
    opt.imgsz = 480                # 图像尺寸
    opt.batch_size = 16            # 批量大小
    opt.epochs = 100               # 训练轮次
    opt.nosave = False             # 允许保存检查点
    opt.save_period = 20           # 每20轮保存一次
    opt.cache = True               # 启用数据缓存
    opt.device = "0"               # 使用GPU 0（若为CPU则设为"cpu"）
    opt.lr0 = 0.01                 # 初始学习率

    # 启动训练
    main(opt)

if __name__ == "__main__":
    custom_train()