import numpy as np

# 加载npy文件
melody_data = np.load('/mnt/data/home/wanghaoyu/TangoFlux/clmp/training/melody.npy')

# 打印维度信息
print("数组形状:", melody_data.shape)
print("数据类型:", melody_data.dtype)

# 打印数组内容摘要
print("\n数据摘要:")
if melody_data.ndim == 1:
    print("前5个元素:", melody_data[:5])
    print("后5个元素:", melody_data[-5:])
elif melody_data.ndim >= 2:
    print("前2行前3列:")
    print(melody_data[:2, :3])
    
    # 如果是3维或更高维度
    if melody_data.ndim >= 3:
        print("\n形状信息更详细:")
        for i in range(melody_data.ndim):
            print(f"维度 {i} 长度: {melody_data.shape[i]}")