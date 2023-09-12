#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main_create_tiles.py
@Time    :   2023/08/08 16:55:54
@Author  :   Darui Jin
@Version :   2.0
@Site    :   https://www.researchgate.net/profile/Darui-Jin
@Desc    :   This is the script to tile whole slide images stored in the .svs format at the desired magnification, along with corresponding segmentation masks and stitches.
'''



import os
import click
from slide_lib import segment_tiling


@click.command()
@click.option('--source_dir', type=str, help='path to the source slide image (.svs) directory')
@click.option('--source_list', type=str, help='path to the source slide image (.svs) list to be processed')
@click.option('--save_dir', type=str, help='path to the save directory')
@click.option('--patch_size', type=int, default=256, help='patch size')
@click.option('--step_size', type=int, default=256, help='step size')
@click.option('--mag', type=int, default=20, help='magnification for patch extraction')
@click.option('--process_list', type=str, default=None, help='name of list of images to process with parameters (.csv)')
@click.option('--index', type=int, default=None)
def batch_tiling(source_dir: str, source_list: str, save_dir: str, patch_size: int, step_size: int, mag: int, process_list: str, index: int) -> None:
    """
    Tile whole slide images stored in the .svs format at the desired magnification.
    
    Parameters
    ----------
    source_dir : str
        Path to the source slide image (.svs) directory.
    source_list : str
        Path to the source slide image (.svs) list to be processed.
    save_dir : str
        Path to the save directory.
    patch_size : int
        Patch size.
    step_size : int
        Step size.
    mag : int
        Magnification for patch extraction.
    process_list : str
        Name of list of images to process with parameters (.csv).

    Returns
    -------
    None
    """

    tile_save_dir = os.path.join(save_dir, 'tiles')
    mask_save_dir = os.path.join(save_dir, 'masks')
    stitch_save_dir = os.path.join(save_dir, 'stitches')
    directories = {'source': source_list if source_list else source_dir,
                   'save_dir': save_dir,
                   'tile_save_dir': tile_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}
    
    for key, val in directories.items():
            if key == 'source':
                continue
            os.makedirs(val, exist_ok=True)

    total_time = segment_tiling(**directories, patch_size=patch_size, mag_level=mag, step_size= step_size, index=index)
    print(f"The average processing time for each slide is {total_time:.2f} seconds.")


if __name__ == '__main__':
    batch_tiling()
