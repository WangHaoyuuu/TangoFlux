#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m training.main \
    --save-frequency 20 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --datasetpath='/mnt/data/melody_all/webdataset_tar/dataset' \
    --melody-path='/mnt/data/melody_all/melody_text' \
    --precision="fp32" \
    --batch-size=32 \
    --lr=1e-5 \
    --wd=0.0 \
    --epochs=60 \
    --workers=0 \
    --use-bn-sync \
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --warmup 3200 \
    --datasetnames "MusicBench" \
    --datasetinfos "train" \
    --top-k-checkpoint-select-dataset="MusicBench-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs '/mnt/data/logs/new_clap_debug' \
    --name 'debug1_3dclap_3d_train_resume' \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --resume '/mnt/data/logs/new_clap_debug/debug1_3dclap_3d_train/checkpoints/epoch_top_0.pt'

    # test git
