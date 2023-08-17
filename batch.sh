for i in {0..40}
do
  bsub -R "rusage[mem=30G]" -J tiling_$i -q verylong -o ./log_file/list_$i.out -e ./log_file/list_$i.err python -W ignore main_create_tiles.py --patch_level 0 --source_list ./splits/slide_split_$i.csv --save_dir ./tiles/20X_256  --patch_size 256 --step_size 256
done