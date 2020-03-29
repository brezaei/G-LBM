vid_list=(
"basic/511"
"basic/Blurred"
"clutter/Board"
"clutter/boulevardJam"
"intermittentMotion/AVSS2007"
"intermittentMotion/busStation"
"jitter/badminton"
"jitter/boulevard"
"veryLong/BusStopMorning"
"veryShort/CUHK_Square"
"veryShort/DynamicBackground"
"illuminationChanges/CameraParameter"
"backgroundMotion/advertisementBoard"
)

#conda activate torch_py3
cd /media/behnaz/My\ Book/BackgroundSubtraction/BackgroundSubtraction_LowRankVAE/code

result_base="../result_nbn_nshuffle/SBMnet2016_nsampl_npp"
chk_base="../output_nbn_nshuffle/checkpoints/SBMnet2016"
vid_base="/media/behnaz/My Book/BackgroundSubtraction/Data/SBMnet2016"


for vid in ${vid_list[@]}; do 
    echo evaluating video:$vid
    result_path="${result_base}/$vid/epoch_480"
    chk_path="${chk_base}/$vid/checkpoint_epoch_480.pth"
    vid_path="${vid_base}/$vid/input"
    mkdir -p $result_path
    #echo $vid_path
    #echo $chk_path
    #echo $result_path

    python test.py -vid_path "${vid_path}" -ckpt "${chk_path}" -result_path "${result_path}" -device 'cpu' &> "${result_path}/log.txt"
    echo evaluation on $vid is finished 
done