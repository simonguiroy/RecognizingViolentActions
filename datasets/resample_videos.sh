#!/bin/bash

#declare -a splits=('train' 'valid' 'test')
declare -a splits=('test')

frame_rate=$1

input_dataset=$2
output_dataset=${input_dataset}_${frame_rate}fps

mkdir $output_dataset
cp $input_dataset/*.csv $output_dataset/
mkdir $output_dataset/data

for split in "${splits[@]}"; do
	mkdir $output_dataset/data/$split
	for action in $(ls $input_dataset/data/$split); do
		mkdir $output_dataset/data/$split/$action
		for video in $(ls $input_dataset/data/$split/${action}); do
			ffmpeg -i $input_dataset/data/$split/${action}/$video  -r $frame_rate -y $output_dataset/data/$split/${action}/${video}
		done
	done
done
