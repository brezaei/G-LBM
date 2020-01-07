
lr=$1
alpha=$2
vid_num=$3

python BetaVAE_BMC2012.py -lr $lr -alpha $alpha -vid_num $vid_num #&>> ../Result/BMC2012/log_explore_vid_$vid_num.txt
