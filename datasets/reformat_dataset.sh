#!/bin/bash

dataset_name='ViolentHumanActions_v1'

declare -a actions=('playing_squash_or_racquetball' 'capoeira' 'punching_bag' 'kissing' 'headbanging' 'shaking_head' 'playing_cricket' 'stretching_arm' 'tango_dancing' 'singing' 'juggling_soccer_ball' 'drop_kicking' 'high_kick' 'punching_person' 'side_kick' 'sword_fighting' 'tai_chi' 'wrestling' 'headbutting' 'slapping')
declare -a splits=("train" "valid" "test")

extension='mp4'

for split in "${splits[@]}"; do
	rm $dataset_name/dataset_${split}.csv
	for action in "${actions[@]}"; do
		idx=1
		for i in $(ls $dataset_name/data/$split/$action)
		do
			num=$(printf %04d ${idx})
			new_video_name=${action}-${split}-${num}.${extension}
			mv $dataset_name/data/$split/$action/$i $dataset_name/data/$split/$action/$new_video_name
			echo $action","$new_video_name >> $dataset_name/dataset_${split}.csv
			idx=$(( $idx + 1 ))
		done
	done
done
