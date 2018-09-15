#!/bin/bash

source activate simon

python eval.py --split 'test' --stream 'rgb' --resize_frames 1.0 --dataset 'ViolentHumanActions_v2' --suffix 'resize_frame=0_5'
