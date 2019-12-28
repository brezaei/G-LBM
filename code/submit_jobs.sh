for i in {1..9}; do
    sbatch exec.sh $i
    # echo $i
done