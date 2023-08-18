for i in {0..40}
do
  bsub -R "rusage[mem=30G]" -J tile_mag20_256_$i -q verylong -o ./log_file/list_$i.out -e ./log_file/list_$i.err python -W ignore main_create_tiles.py --index $i --source_list ./splits_yaml/slide_split_$i.yaml --save_dir ./20x_256  --patch_size 256 --step_size 256 --mag 20
done