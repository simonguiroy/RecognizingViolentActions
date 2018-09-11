#!/bin/bash
declare -a arr=('playing_squash_or_racquetball' 'capoeira' 'punching_bag' 'kissing' 'headbanging' 'shaking_head' 'playing_cricket' 'stretching_arm' 'tango_dancing' 'singing' 'juggling_soccer_ball' 'drop_kicking' 'high_kick' 'punching_person' 'side_kick' 'sword_fighting' 'tai_chi' 'wrestling' 'headbutting' 'slapping')
declare -a splits=('train' 'valid' 'test')

dataset_name='ViolentHumanActions_v1'

total_count=0
for split in "${splits[@]}"; do
	split_count=$(cat $dataset_name/dataset_${split}.csv | wc -l)
	total_count=$(( $total_count + $split_count ))
done

for split in "${splits[@]}"; do
	split_count=$(cat $dataset_name/dataset_${split}.csv | wc -l)
	for i in "${arr[@]}"; do
		num_videos=$(ls $dataset_name/data/$split/${i} | wc -l )
		ratio=$(bc <<< "scale=3; $num_videos/${split_count}")
		echo $i : $split : $num_videos - $ratio

	done
	echo
	echo
done
