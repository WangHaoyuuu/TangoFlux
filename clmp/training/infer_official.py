import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append('/mnt/data/home/wanghaoyu/TangoFlux/clmp')

import torch
import librosa
import pyloudnorm as pyln
from open_clip import create_model
from training.data import get_audio_features, int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer
import torch.nn.functional as F

# 定义模型PATH
PRETRAINED_PATH = "/mnt/data/home/wanghaoyu/TangoFlux/clmp/training/mg2-clmp.pt"
# 定义GPU设备
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载模型
def load_model():
    """加载并初始化模型"""
    precision = "fp32"
    amodel = "HTSAT-base"
    tmodel = "roberta"
    
    # 首先创建模型但不自动加载权重
    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained="",  # 空字符串避免自动加载
        precision=precision,
        device=DEVICE,
        enable_fusion=False
    )
    
    # 手动加载权重
    print(f"加载检查点: {PRETRAINED_PATH}")
    checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    
    if "epoch" in checkpoint:
        print(f"检测到训练检查点 (epoch {checkpoint['epoch']})")
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            print("检测到分布式训练模型，移除'module.'前缀")
            sd = {k[len("module."):]: v for k, v in sd.items()}
    else:
        print("检测到裸模型检查点")
        sd = checkpoint
    
    # 加载状态字典
    msg = model.load_state_dict(sd, strict=False)
    print(f"缺失键数量: {len(msg.missing_keys)}")
    print(f"意外键数量: {len(msg.unexpected_keys)}")

    # 设置为评估模式
    model.eval()

    # 统计模型总参数数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params:,}")

    # 统计从检查点加载的参数数量（需形状匹配）
    loaded_params = 0
    for name, param in model.named_parameters():
        if name in sd:
            if param.shape == sd[name].shape:
                loaded_params += param.numel()
            else:
                print(f"参数 {name} 形状不匹配！模型: {param.shape} vs 检查点: {sd[name].shape}")

    print(f"成功加载的参数总数: {loaded_params:,}")
    print(f"成功加载的参数比例: {loaded_params/total_params:.2%}")  # 百分比显示

    return model, model_cfg

# 文本推理
# tokenizer初始化
tokenize = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

def preprocess_text(text):
    """文本tokenization处理，保留batch维度"""
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    # 不使用squeeze，保留batch维度
    return result

# 文本推理主函数
def infer_text_embedding(model, text):
    # 如果文本是单个字符串，转为列表
    if isinstance(text, str):
        text = [text]
    # 准备文本输入
    text_data = preprocess_text(text)

    # 移动text_data到正确的设备上
    for k, v in text_data.items():
        if isinstance(v, torch.Tensor):
            text_data[k] = v.to(DEVICE)

    model.eval()
    with torch.no_grad():
        text_embed = model.get_text_embedding(text_data)
        text_features = F.normalize(text_embed, dim=-1)
    return text_features

# 音频推理
def infer_audio_embedding(model, audio_file_path):
    audio_dict = {}

    # 1. 加载音频数据
    audio_waveform, sr = librosa.load(audio_file_path, sr=48000)
    # 添加峰值归一化
    audio_waveform = pyln.normalize.peak(audio_waveform, -1.0)  # 与 CLAP 对齐
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()

    # 2. 处理音频特征
    audio_dict = get_audio_features(
        audio_dict,
        audio_waveform,
        480000,
        data_truncating="rand_trunc",
        data_filling="repeatpad",
        audio_cfg=model.audio_cfg,
    )

    if "waveform" in audio_dict and isinstance(audio_dict["waveform"], torch.Tensor):
        if audio_dict["waveform"].dim() == 1:
            # 添加批次维度
            audio_dict["waveform"] = audio_dict["waveform"].unsqueeze(0)
        audio_dict["waveform"] = audio_dict["waveform"].to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        audio_embed = model.get_audio_embedding(audio_dict)
        audio_features = F.normalize(audio_embed, dim=-1)
    
    return audio_features

# melody 推理
def read_txt_file(file_path):
    """读取文本文件内容"""
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def infer_melody_embedding(model, melody_file_path):
    melody_data = read_txt_file(melody_file_path)

    model.eval()
    with torch.no_grad():
        melody_embed = model.get_melody_embedding(melody_data)
        melody_features = F.normalize(melody_embed, dim=-1)

    return melody_features

# 计算各个模态的相似度
def caculate_similarity(t, a, m):
    melody_text_sim = F.cosine_similarity(m, t, dim=-1)  # dim=-1 表示最后一个维度
    audio_text_sim = F.cosine_similarity(a, t, dim=-1)
    melody_audio_sim = F.cosine_similarity(m, a, dim=-1)

    return {
        "similarities": {
            "melody_text": melody_text_sim,
            "audio_text": audio_text_sim,
            "melody_audio": melody_audio_sim,
        }
    }

if __name__ == "__main__":
    # 加载模型
    model, model_cfg = load_model()

    # 准备输入数据
    text_query = ""
    audio_path = ""
    melody_file_path = ""

    # 分别获取三个feature
    text_features = infer_text_embedding(model, text_query)
    audio_features = infer_audio_embedding(model, audio_path)
    melody_features = infer_melody_embedding(model, melody_file_path)

    #
    results = caculate_similarity(text_features, audio_features, melody_features)
    # 打印相似度结果
    similarities = results["similarities"]
    print("\n===== 多模态相似度 =====")
    print(f"旋律-文本相似度: {similarities['melody_text'].item():.4f}")
    print(f"音频-文本相似度: {similarities['audio_text'].item():.4f}")
    print(f"旋律-音频相似度: {similarities['melody_audio'].item():.4f}")
    # 打印CLMP分数
    
    CLMP_score = (similarities['melody_text'].item() + similarities['audio_text'].item() + similarities['melody_audio'].item())/3
    print(f"CLMP_score:{CLMP_score:.4f}")