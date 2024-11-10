#!/bin/bash

timestamp=$(date +%s)
redirect_log="train_log_${timestamp}.txt"

echo "train log saved to ${redirect_log}"

export MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MUSA_KERNEL_TIMEOUT=3600000

nohup torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 25555 train.py --batch 384 --cfg models/yolov5m.yaml --data data/coco.yaml --epochs 300 --weights  '' --hyp data/hyps/hyp.scratch-high.yaml --device musa:0,1,2,3,4,5,6,7 > ${redirect_log} 2>&1 &


# run evaluation
# python val.py --data data/coco.yaml --weight runs/train/exp4/weights/best.pt --device 0
