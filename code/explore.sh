lr_list=(0.0001)
alpha_list=(0.0 0.3 0.6 0.9 1.5 2.0 2.5 3.0)
vid_list=(3 7 8)

for lr in ${lr_list[@]}; do
	for alpha in ${alpha_list[@]}; do
		for vid_num in ${vid_list[@]}; do
			echo submitting lr=$lr, alpha=$alpha, vid_num=$vid_num
			bash submit.sh $lr $alpha $vid_num
		done
	done
done
