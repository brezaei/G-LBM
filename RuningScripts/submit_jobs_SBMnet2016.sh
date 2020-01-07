address_list = (
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

for vid in ${address_list[@]}; do
    echo submitting video:$vid
    sbatch  exec_SBMnet2016_discovery.sh $vid
done

