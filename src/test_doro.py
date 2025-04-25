import cv2
import torch


# 读取视频
video_path = "../testVideo/doro3.mp4"
cap = cv2.VideoCapture(video_path)

# 加载训练的模型       ('../../test_doro' 为项目路径)
model = torch.hub.load('../../test_doro', 'custom', path='../runs/train/exp/weights/best.pt', source='local')

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 播放视频
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 模型推理
    results = model(frame)
    # 获取预测结果
    for pred in results.pred[0]:
        x1, y1, x2, y2, conf, cls = pred.tolist()
        class_name = model.names[int(cls)]
        # 输出结果
        print(f"检测到：{class_name}, 置信度：{conf:.2f}")
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # 显示当前帧
    cv2.imshow("Video", frame)
    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象和关闭所有窗口
cap.release()
cv2.destroyAllWindows()