lr=0.001
epochs=200
vid_path="/media/behnaz/My Book/BackgroundSubtraction/Data/BMC2012/Video_004/frames"


python train_singleVid.py -lr $lr -epochs $epochs -clip 1 -vid_path "${vid_path}" --train_with_eval #&>> ../Result/BMC2012/log_explore_vid_$vid_num.txt