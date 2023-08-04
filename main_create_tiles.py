import os
import argparse
from segment_patching import segment_tiling


parser = argparse.ArgumentParser(description='Tiling')
parser.add_argument('--source_dir', type=str, help='path to the source slide image (.svs) directory')
parser.add_argument('--save_dir', type=str, help='path to the save directory')
parser.add_argument('--patch_size', type=int, default=256, help='patch size')
parser.add_argument('--step_size', type=int, default=256, help='step size')
parser.add_argument('--patch_level', type=int, default=0, help='downsample level for patch extraction')
parser.add_argument('--process_list', type=str, default=None, help='name of list of images to process with parameters (.csv)')


if __name__=='__main__':
    args = parser.parse_args()
    tile_save_dir = os.path.join(args.save_dir, 'tiles')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')
    directories = {'source': args.source_dir,
                   'save_dir': args.save_dir,
                   'tile_save_dir': tile_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}
    
    for key, val in directories.items():
            os.makedirs(val, exist_ok=True)

    seg_time = segment_tiling(**directories, patch_size=args.patch_size, patch_level=args.patch_level, step_size= args.step_size)