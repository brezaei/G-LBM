#[500-9000] frames
address_list_1=(
    "backgroundMotion/overpass"
    "backgroundMotion/advertisementBoard"
    "intermittentMotion/busStation"
    "intermittentMotion/streetCorner"
    "clutter/HumanBody2"
    "basic/511"
    "clutter/IndianTraffic3"
    "illuminationChanges/cubicle"
    "jitter/traffic"
    "clutter/tramway"
    "intermittentMotion/AVSS2007"
    "jitter/badminton"
    "basic/PETS2006"
    "jitter/sidewalk"
    "basic/Intersection"
    "intermittentMotion/office"
    "intermittentMotion/Teknomo"
    "basic/highway"
    "intermittentMotion/UCF-traffic"
    "intermittentMotion/tramstop"
    "illuminationChanges/Dataset3Camera2"
    "illuminationChanges/Dataset3Camera1"
    "backgroundMotion/fall"
    "clutter/boulevardJam"
    "basic/skating"
    "intermittentMotion/copyMachine"
    "jitter/boulevard"
    "intermittentMotion/sofa"
    "veryLong/BusStopMorning"
    "veryLong/PedAndStorrowDrive3"
    "veryLong/Dataset4Camera1"
    "veryLong/Terrace"
    "veryLong/PedAndStorrowDrive"
)
#[100-500] frames
address_list_2=(
    "clutter/UCF-fishes"
    "basic/Blurred"
    "clutter/Crowded"
    "basic/fluidHighway"
    "intermittentMotion/I_MB_02"
    "backgroundMotion/fountain02"
    "jitter/CMU"
    "basic/IntelligentRoom"
    "clutter/Board"
    "backgroundMotion/canoe"
    "basic/Hybrid"
    "intermittentMotion/I_CA_01"
    "intermittentMotion/CaVignal"
    "jitter/O_SM_04"
    "basic/wetSnow"
    "basic/CamouflageFgObjects"
    "basic/IPPR2"
    "clutter/groupCampus"
    "clutter/ICRA3"
    "intermittentMotion/I_CA_02"
    "illuminationChanges/I_IL_02"
    "clutter/People&Foliage"
    "intermittentMotion/Candela_m1.10"
    "intermittentMotion/I_MB_01"
    "clutter/Foliage"
    "illuminationChanges/CameraParameter"
    "basic/MPEG4_40"
    "intermittentMotion/Uturn"
)
#[50-100] frames
address_list_3=(
    "backgroundMotion/fountain01"
    "basic/ComplexBackground"
    "illuminationChanges/I_IL_01"
    "jitter/I_MC_02"
    "jitter/O_MC_02"
    "basic/streetCornerAtNight"
    "basic/I_SI_01"
    "jitter/I_SM_04"
)
#[6-10] frames
address_list_4=(
    "veryShort/NoisyNight"
    "veryShort/MIT"
    "veryShort/snowFall"
    "veryShort/CUHK_Square"
    "veryShort/pedestrians"
    "veryShort/peopleInShade"
    "veryShort/TownCentre"
    "veryShort/DynamicBackground"
    "veryShort/Toscana"
    "veryShort/TwoLeaveShop1cor"
)

if [ $1 == 1 ]
then
    echo processing list_1 with batch size of 120
    video_list=${address_list_1[@]}
    batch_size=120
elif [ $1 == 2 ]
then
    echo processing list_2 with batch size of 50
    video_list=${address_list_2[@]}
    batch_size=50
elif [ $1 == 3 ]
then
    echo processing list_3 with batch size of 25
    video_list=${address_list_3[@]}
    batch_size=25
elif [ $1 == 4 ]
then
    echo processing list_4 with batch size of 25
    video_list=${address_list_4[@]}
    batch_size=3
else
    echo "Error!-video list does not exist"
    exit 125
fi


for vid in ${video_list[@]}; do
    echo submitting video:$vid
    sbatch  train_SBMnet2016_discovery.sh $vid $batch_size
done

