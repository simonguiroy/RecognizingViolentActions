#!/bin/bash

source activate simon

python eval.py --split 'test' --stream 'flow' --dataset 'ViolentHumanActions_v2_15fps'
