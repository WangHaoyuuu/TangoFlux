import sys

sys.path.append("src/clap")

import os
import torch
import librosa
from open_clip import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer
import torch.nn.functional as F
tokenize = RobertaTokenizer.from_pretrained("roberta-base")


def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}


PRETRAINED_PATH = "/mnt/fast/nobackup/users/hl01486/projects/contrastive_pretraining/CLAP/assets/checkpoints/epoch_top_0_audioset_no_fusion.pt"
WAVE_48k_PATH = "/mnt/fast/nobackup/users/hl01486/projects/contrastive_pretraining/CLAP/assets/audio/machine.wav"


def infer_text():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "fp32"
    amodel = "HTSAT-tiny"  # or 'PANN-14'
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
    print(text_embed.size())

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
    pretrained = PRETRAINED_PATH
    
    # Load the model
    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
    )
    
    # Load melody data from file
    melody_data = read_txt_file(melody_file_path)
    
    # Process melody data through model
    with torch.no_grad():
        melody_embed = model.encode_melody(melody_data, device=device)
        # Apply same transformations as in the forward pass if needed
        melody_embed = model.melody_mlp_1024_to_768(melody_embed)
        melody_embed = model.melody_projection(melody_embed)
        melody_embed = F.normalize(melody_embed, dim=-1)
    
    print(f"Melody embedding shape: {melody_embed.size()}")
    return melody_embed

def infer_audio():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "fp32"
    amodel = "HTSAT-tiny"  # or 'PANN-14'
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

    # load the waveform of the shape (T,), should resample to 48000
    audio_waveform, sr = librosa.load(WAVE_48k_PATH, sr=48000)
    # quantize
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    audio_dict = {}

    # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
    import ipdb

    ipdb.set_trace()
    audio_dict = get_audio_features(
        audio_dict,
        audio_waveform,
        480000,
        data_truncating="fusion",
        data_filling="repeatpad",
        audio_cfg=model_cfg["audio_cfg"],
    )
    # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
    audio_embed = model.get_audio_embedding([audio_dict])
    print(audio_embed.size())
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    infer_text()
    infer_audio()
