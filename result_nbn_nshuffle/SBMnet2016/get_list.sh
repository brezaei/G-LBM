dirs=`ls -d -- */`

for dir in $dirs; do

	cd $dir
	subdirs=`ls`
	
	for subdir in $subdirs; do
		if [ -d "$subdir/epoch_480" ]; then
			echo $dir$subdir	`ls $subdir/epoch_480 | wc -l`
		fi
	done

	cd ..

done
