import os
import cv2
import glob
import time
import openslide
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import pandas as pd
from PIL import Image
from utils import *
import multiprocessing as mp
import pdb


def segment(wsi: openslide.OpenSlide)->tuple[list, list, Image.Image, float]:
    start_time = time.time()
    seg_level = wsi.level_count - 1
    img = np.array(wsi.read_region((0, 0), seg_level, wsi.level_dimensions[-1]).convert('RGB'))  # 最低分辨率下进行分割
    img_gray = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw_1 = hysteresis_threshold(img_gray).astype(np.uint8)
    bw_2 = gray_filter(img)
    bw = move_small((bw_1 & bw_2)).astype(np.uint8)
    
    scale = wsi.level_downsamples[seg_level]
    scaled_ref_patch_area = int(512 ** 2 / scale ** 2)

    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    foreground_contours, hole_contours = filter_contours(contours, hierarchy, {'a_t':10 * scaled_ref_patch_area, 'a_h': 10 * scaled_ref_patch_area, 'max_n_holes':8})
    
    # save mask file
    line_thickness = int(200 / scale)
    cv2.drawContours(img, foreground_contours, -1, (0, 255, 0), line_thickness, lineType=cv2.LINE_8)
    img = Image.fromarray(img)

    contours_tissue = scaleContourDim(foreground_contours, scale)
    holes_tissue = scaleHolesDim(hole_contours, scale)
    contours_tissue = [contours_tissue[i] for i in range(len(contours_tissue))]  # 最高分辨率下坐标
    holes_tissue = [holes_tissue[i] for i in range(len(contours_tissue))]  # 最高分辨率下坐标
    return contours_tissue, holes_tissue, img, time.time() - start_time


def patching(wsi: openslide.OpenSlide, contours: list, holes: list, tile_save_dir: str, 
             patch_size: float | int, patch_level: float | int, step_size: float | int, slide_path: str)->tuple[list, float]:
    start_time = time.time()
    patch_downsample = wsi.level_downsamples[patch_level]  # to be done. 20x
    ref_patch_size = int(patch_size * patch_downsample)  # 最高分辨率对应尺寸
    ref_step_size = int(step_size * patch_downsample)  # 最高分辨率对应步长
    slide_id = os.path.basename(slide_path).split('.')[0]
    coord_record = []
    for cont_idx, cont in enumerate(contours):
        start_x, start_y, w, h = cv2.boundingRect(cont)
        stop_x = start_x + w
        stop_y = start_y + h
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))
        cont_check_fn = isInContour(contour=cont, patch_size=ref_patch_size, center_shift=0.5)
        x_range = np.arange(start_x, stop_x, step=ref_step_size)
        y_range = np.arange(start_y, stop_y, step=ref_step_size)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()
        # 多核并行
        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(processes=num_workers)
        iterable = [(coord, holes[cont_idx], ref_patch_size, cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(process_coord_candidate, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])
        print('Extracted {} points within the contour'.format(len(results)))
        if len(results)>1:
            asset_dict = {'coords': results}
            attr = {'patch_size' :            patch_size, 
                    'patch_level' :           patch_level,
                    'downsample':             wsi.level_downsamples[patch_level],
                    'level_dim':              wsi.level_dimensions[patch_level],
                    'name':                   slide_id,
                    'save_path':              tile_save_dir}
            attr_dict = {'coords': attr}
            os.makedirs(os.path.join(tile_save_dir, slide_id), exist_ok=True)
            final_coords = save_tiles(slide_path, asset_dict, attr_dict['coords'])
            coord_record.extend(final_coords)
    return coord_record, time.time() - start_time


def stitching(wsi: openslide.OpenSlide, coords: list, patch_size: float | int, patch_level: float | int, step_size: float | int, 
              downscale: float | int = 64)->tuple[Image.Image, float]:
    start_time = time.time()
    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    print('Start stitching...')
    print(f'Original size: w: {w} x h: {h}')
    w, h = wsi.level_dimensions[vis_level]
    print(f'Downscaled size for stiching: w: {w} x h: {h}')
    print(f'Number of patches: {len(coords)}')
    print(f'Patch size: {patch_size}x{patch_size} at patch level: {patch_level}')
    print(f'Ref patch size: {patch_size*int(wsi.level_downsamples[patch_level])}')
    patch_size = int(patch_size * wsi.level_downsamples[patch_level])
    heatmap = Image.new(size=(w, h), mode="RGB", color=(0, 0, 0))
    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi, coords, patch_size, vis_level)
    return heatmap, time.time() - start_time


def segment_tiling(source: str, save_dir: str, tile_save_dir: str, mask_save_dir: str, stitch_save_dir: str,
                   patch_size: int | float =256, patch_level: int | float =0, step_size: int | float =256)->None:
    if source.endswith('.csv'):
        slides = sorted(pd.read_csv(source)['file_path'].tolist())  # source是csv路径时使用
    else:
        slides = sorted(glob.glob(os.path.join(source, '*.svs')))  # source是svs路径时使用
    d = {}
    d["slide_path"] = slides
    df = pd.DataFrame(d)
    df['slide_mpp'] = np.nan
    df['slide_mag'] = np.nan
    seg_time = 0.
    tile_time = 0.
    stitch_time = 0.
    for i in range(len(df)):
        slide_path = df['slide_path'][i]
        slide_name = os.path.basename(slide_path).split('.')[0]
        print(f"\nprogress: {i+1}/{len(df)}")
        print(f"processing {slide_name}")
        wsi = openslide.open_slide(slide_path)
        df.loc[i, 'slide_mpp'] = float(wsi.properties['aperio.MPP'])
        df.loc[i, 'slide_mag'] = float(wsi.properties['openslide.objective-power'])
        contour_coord, hole_coord, mask, seg_time_elapsed = segment(wsi)
        mask.save(os.path.join(mask_save_dir, f'{slide_name}.png'))
        coords, tile_time_elapsed = patching(wsi, contour_coord, hole_coord, tile_save_dir, patch_size, patch_level, step_size, slide_path)
        heatmap, stitch_time_elapsed = stitching(wsi, coords, patch_size, patch_level, step_size, downscale=64)
        heatmap.save(os.path.join(stitch_save_dir, slide_name+'.png'))
        print(f"segmentation took {seg_time_elapsed} seconds")
        print(f"patching took {tile_time_elapsed} seconds")
        print(f"stitching took {stitch_time_elapsed} seconds")
        seg_time += seg_time_elapsed
        tile_time += tile_time_elapsed
        stitch_time += stitch_time_elapsed
    
    df.to_csv(os.path.join(save_dir, 'slide_info.csv'), index=False)
    print(f"slides info (mpp, magnification) saved to {os.path.join(save_dir, 'slide_info.csv')}")
    seg_time /= len(df)
    tile_time /= len(df)
    stitch_time /= len(df)
    print(f"average segmentation time in s per slide: {seg_time}")
    print(f"average patching time in s per slide: {tile_time}")
    print(f"average stiching time in s per slide: {stitch_time}")
        


