from export import run

run(
    weights='D:/Code/Python/test_doro/runs/train/exp/weights/best.pt',
    include=('torchscript',),  # 只导出torchscript
    device='0'  # '0' 用GPU； ‘cpu’ 用CPU
)