import cv2
import torch

# 加载模型并移动到 GPU
model_path = '../runs/train/exp/weights/best.torchscript'
model = torch.jit.load(model_path)
model.to('cuda')
model.eval()
print("模型加载成功!")

# 读取并预处理图片
img = cv2.imread('E:/test/testImg/doro1.png')  # 用一张测试图片
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img_float = img.astype('float32') / 255.0
tensor_image = torch.from_numpy(img_float)
tensor_image = tensor_image.unsqueeze(0)  # 变为 [1, H, W, 3]
tensor_image = tensor_image.permute(0, 3, 1, 2)  # 调整为 [N, C, H, W]
tensor_image = tensor_image.to('cuda')

print("tensor_image size:", tensor_image.size())
print("tensor_image dtype:", tensor_image.dtype)

# 模型推理
output = model(tensor_image)

print("模型输出:", output)
print("==========================")

# 绘制检测结果
# for pred in output[0]:
#     x1, y1, x2, y2, conf, cls = pred.tolist()
#     class_name = model.names[int(cls)]
#     print(f"检测到：{class_name}, 置信度：{conf:.2f}")
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#     cv2.putText(img, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# # 显示结果
# cv2.imshow("Detection Result", img)