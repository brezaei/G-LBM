lr=0.002
epochs=1000
vid_path="/media/behnaz/My Book/BackgroundSubtraction/Data/BMC2012/Video_008/frames"

python train_singleVid.py -lr $lr -epochs $epochs -clip 1 \
    -decay_step_list 50 100 150 200 250 300 350 400 450 500 \
    -vid_path "${vid_path}" --train_with_eval &>> log.txt