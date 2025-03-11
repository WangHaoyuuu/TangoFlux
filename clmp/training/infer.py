import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append('/mnt/data/home/wanghaoyu/TangoFlux/clmp')
import torch
import librosa
from open_clip import create_model
from training.data import get_audio_features, int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer
import torch.nn.functional as F

# 常量定义
PRETRAINED_PATH = "/mnt/data/home/wanghaoyu/TangoFlux/clmp/training/mg2-clmp.pt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 文本tokenizer初始化
tokenize = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

def tokenizer(text):
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

def read_txt_file(file_path):
    """读取文本文件内容"""
    with open(file_path, 'r') as file:
        content = file.read()
    return content

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
    
    return model, model_cfg

def infer_unified(model, text_query, melody_file_path, audio_file_path):
    """统一的推理函数，使用模型的forward方法处理三种模态"""
    # 如果文本是单个字符串，转为列表
    if isinstance(text_query, str):
        text_query = [text_query]
    
    # 准备文本输入
    text_data = tokenizer(text_query)
    
    # 准备音频和旋律输入
    audio_dict = {}
    
    # 1. 加载音频数据
    audio_waveform, sr = librosa.load(audio_file_path, sr=48000)
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
    
    # 关键修改：确保waveform是二维的[batch, time]
    if "waveform" in audio_dict and isinstance(audio_dict["waveform"], torch.Tensor):
        if audio_dict["waveform"].dim() == 1:
            # 添加批次维度
            audio_dict["waveform"] = audio_dict["waveform"].unsqueeze(0)
        audio_dict["waveform"] = audio_dict["waveform"].to(DEVICE)
    
    # 3. 加载旋律数据并添加到audio_dict中
    melody_data = read_txt_file(melody_file_path)
    audio_dict["melody_text"] = [melody_data]
    
    # 确保移动到正确的设备
    for k, v in text_data.items():
        if isinstance(v, torch.Tensor):
            text_data[k] = v.to(DEVICE)
    
    # 执行推理前打印调试信息
    print("\n音频字典内容:")
    for key, value in audio_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    # print("\n测试音频编码器:")
    # try:
    #     with torch.no_grad():
    #         # 将tensor移动到正确设备
    #         for key in audio_dict:
    #             if isinstance(audio_dict[key], torch.Tensor):
    #                 audio_dict[key] = audio_dict[key].to(DEVICE)
                    
    #         # 逐步执行音频编码过程
    #         audio_output = model.audio_branch(audio_dict, None, DEVICE)
    #         print(f"音频分支输出类型: {type(audio_output)}")
    #         print(f"音频分支输出键: {audio_output.keys() if isinstance(audio_output, dict) else 'Not a dict'}")
            
    #         if isinstance(audio_output, dict) and "embedding" in audio_output:
    #             print(f"原始音频嵌入: shape={audio_output['embedding'].shape}")
    #             print(f"包含NaN: {torch.isnan(audio_output['embedding']).any().item()}")
                
    #         # 测试投影层
    #         if isinstance(audio_output, dict) and "embedding" in audio_output:
    #             audio_proj = model.audio_projection(audio_output["embedding"])
    #             print(f"投影后音频嵌入: shape={audio_proj.shape}")
    #             print(f"包含NaN: {torch.isnan(audio_proj).any().item()}")
    # except Exception as e:
    #     print(f"音频编码器错误: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

    # 执行推理
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):  # 更新为新的语法
                outputs = model(audio_dict, text_data, DEVICE)
        else:
            outputs = model(audio_dict, text_data, DEVICE)

    
    # 解析输出
    # melody_features, audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t = outputs
    # melody_embed = model.encode_melody(melody_data, device=DEVICE)

    # melody_embed = model.melody_mlp_1024_to_768(melody_embed)
    # melody_embed = model.melody_projection(melody_embed)
    # melody_features = F.normalize(melody_embed, dim=-1)

    melody_features = model.get_melody_embedding(melody_data)
    # melody_features = model.melody_projection(melody_features)
    
    # 处理旋律特征 - 将序列特征平均池化为单个向量
    print(melody_features.shape)
    if melody_features.dim() > 1 and melody_features.size(0) > 1:
        melody_features_pooled = torch.mean(melody_features, dim=0, keepdim=True)
    else:
        melody_features_pooled = melody_features
    # audio_embed = model.get_audio_embedding([audio_dict])
    # print(audio_embed.size())
    # 确保所有特征都被归一化
    # update melody
    
    melody_features_pooled = F.normalize(melody_features_pooled, dim=-1)

    # update audio
    audio_embed = model.get_audio_embedding(audio_dict)
    audio_features = F.normalize(audio_embed, dim=-1)
    
    # update text
    text_features = model.get_text_embedding(text_data)
    text_features = F.normalize(text_features, dim=-1)
    
    # 打印形状检查
    print(f"旋律特征形状: {melody_features_pooled.shape}")
    print(f"音频特征形状: {audio_features.shape}")
    print(f"文本特征形状: {text_features.shape}")
    
    # 计算各模态间的相似度
    melody_text_sim = torch.matmul(melody_features_pooled, text_features.T)
    audio_text_sim = torch.matmul(audio_features, text_features.T)
    melody_audio_sim = torch.matmul(melody_features_pooled, audio_features.T)
    
    # 返回结果
    return {
        "features": {
            "melody": melody_features,
            "audio": audio_features, 
            "text": text_features,

        },
        "similarities": {
            "melody_text": melody_text_sim,
            "audio_text": audio_text_sim,
            "melody_audio": melody_audio_sim,
        }
    }

# 演示使用方法
if __name__ == "__main__":
    # 1. 加载模型
    model, model_cfg = load_model()

    # 2. 准备输入数据
    text_query = "A piano melody with soft notes"
    melody_file_path = "/mnt/data/home/wanghaoyu/TangoFlux/clmp/training/00005532.txt"
    audio_path = "/mnt/data/home/wanghaoyu/TangoFlux/clmp/training/B496Qv0CuOQ.wav"
    
    # 3. 执行统一推理
    results = infer_unified(model, text_query, melody_file_path, audio_path)
    
    # 4. 打印相似度结果
    print("\n===== 多模态相似度 =====")
    print(f"旋律-文本相似度: {results['similarities']['melody_text'][0][0]:.4f}")
    print(f"音频-文本相似度: {results['similarities']['audio_text'][0][0]:.4f}")
    print(f"旋律-音频相似度: {results['similarities']['melody_audio'][0][0]:.4f}")