import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# sys.path.append("src/clap")
import sys
sys.path.append('/mnt/data/home/wanghaoyu/TangoFlux/clmp')
import os
import torch
import librosa
from open_clip import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer
import torch.nn.functional as F
tokenize = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}


PRETRAINED_PATH = "/mnt/data/home/wanghaoyu/TangoFlux/clmp/training/mg2-clmp.pt"
WAVE_48k_PATH = "/mnt/data/home/wanghaoyu/TangoFlux/clmp/training/1rhsnmWLeGw.wav"


def infer_text():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "fp32"
    amodel = "HTSAT-base"  # or 'PANN-14'
    tmodel = "roberta"  # the best text encoder in our training
    enable_fusion = False  # False if you do not want to use the fusion model
    fusion_type = "aff_2d"
    pretrained = PRETRAINED_PATH

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
    )
    # load the text, can be a list (i.e. batch size)
    text_data = ["I love the contrastive learning", "I love the pretrain model"]
    # tokenize for roberta, if you want to tokenize for another text encoder, please refer to data.py#L43-90
    text_data = tokenizer(text_data)

    text_embed = model.get_text_embedding(text_data)
    print(f"hello {text_embed.size()}")

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def infer_melody(melody_file_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "fp32"
    amodel = "HTSAT-base"
    tmodel = "roberta"
    enable_fusion = False
    fusion_type = "aff_2d"
    
    # 使用您的本地路径
    pretrained = "/mnt/data/home/wanghaoyu/TangoFlux/clmp/training/mg2-clmp.pt"
    
    # 创建模型
    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained="",  # 跳过自动加载
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
    )
    
    # 加载检查点
    print(f"Loading checkpoint from: {pretrained}")
    checkpoint = torch.load(pretrained, map_location=device)
    
    # 检查检查点结构并打印键
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # 根据检查点结构选择加载方式
    if "epoch" in checkpoint:
        print(f"检测到训练检查点 (epoch {checkpoint['epoch']})")
        sd = checkpoint["state_dict"]
        # 处理可能的 "module." 前缀
        if next(iter(sd.items()))[0].startswith("module"):
            print("检测到分布式训练模型，移除 'module.' 前缀")
            sd = {k[len("module."):]: v for k, v in sd.items()}
    else:
        print("检测到裸模型检查点")
        sd = checkpoint
    
    # 打印加载前的一些信息
    print(f"模型总参数数量: {len(model.state_dict().keys())}")
    print(f"加载的参数数量: {len(sd.keys())}")
    
    # 加载状态字典，使用 strict=False 允许缺失键
    msg = model.load_state_dict(sd, strict=False)
    print(f"缺失键数量: {len(msg.missing_keys)}")
    print(f"意外键数量: {len(msg.unexpected_keys)}")
    
    # 如果您想查看具体缺失了哪些键（限制打印前10个）
    if len(msg.missing_keys) > 0:
        print("缺失键样本:")
        for key in msg.missing_keys[:10]:
            print(f"  - {key}")
    if len(msg.unexpected_keys) > 0:
        print("意外键样本:")
        for key in msg.unexpected_keys[:10]:
            print(f"  - {key}")
    
    # 加载旋律数据
    melody_data = read_txt_file(melody_file_path)
    
    # 处理旋律数据
    with torch.no_grad():
        melody_embed = model.encode_melody(melody_data, device=device)
        melody_embed = model.melody_mlp_1024_to_768(melody_embed)
        melody_embed = model.melody_projection(melody_embed)
        melody_embed = F.normalize(melody_embed, dim=-1)
    
    print(f"旋律嵌入形状: {melody_embed.size()}")
    return melody_embed

def infer_audio():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "fp32"
    amodel = "HTSAT-base"  # or 'PANN-14'
    tmodel = "roberta"  # the best text encoder in our training
    enable_fusion = False  # False if you do not want to use the fusion model
    fusion_type = "aff_2d"
    pretrained = PRETRAINED_PATH

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
    )

    # 准备音频和旋律输入
    audio_dict = {}
    
    # 1. 加载音频数据
    audio_waveform, sr = librosa.load(WAVE_48k_PATH, sr=48000)
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
    # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
    audio_embed = model.get_audio_embedding(audio_dict)
    print(audio_embed.size())



if __name__ == "__main__":
    # infer_text()
    infer_audio()
