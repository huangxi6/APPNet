#!/bin/bash

CS_PATH='../progressive/dataset/LIP'
BS=32
GPU_IDS='0'
INPUT_SIZE='473,473'
SNAPSHOT_FROM='./s_lip_appn_150'
DATASET='val'
NUM_CLASSES=20

python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}
