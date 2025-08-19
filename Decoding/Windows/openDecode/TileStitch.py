from .Register import _EulerTransform, _applyEulerTransform2048, _applyRegisterMatrix, _getRegisterMatrix
from .Stitching import _prepareMIST, _runMIST
import SimpleITK as sitk
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
import tifffile
import glob
import cv2
import os

from numba import njit


def _apply_clahe(image:np.ndarray, clipLimit = 10.0):
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    image = clahe.apply(image)
    # image = cv2.GaussianBlur(image, (3,3), 0)
    
    return image

@njit
def _boundary_handle(pseudo_index, y_pos, x_pos, STITCHED_IMAGE_SIZE):
    
    # Boundary handling
    d_dim1 = pseudo_index[0] + y_pos
    dim1_criteria = (d_dim1 >= 0) & (d_dim1 < STITCHED_IMAGE_SIZE[0])
    
    d_dim2 = pseudo_index[1] + x_pos
    dim2_criteria = (d_dim2 >= 0) & (d_dim2 < STITCHED_IMAGE_SIZE[1])
    
    dim_critria = dim1_criteria & dim2_criteria
    
    d_dim1 = d_dim1[dim_critria]
    m_dim1 = pseudo_index[0][dim_critria]
    d_dim2 = d_dim2[dim_critria]
    m_dim2 = pseudo_index[1][dim_critria]
    
    return d_dim1, d_dim2, m_dim1, m_dim2
      
def _stitch_handle(move_image, transform_parameter_map, ch_stitch_image: np.ndarray, i: int,
                               STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID, shift, phase_cross_correction):
    
    tmp_stitch_image = np.copy(STITCHED_IMAGE)
    tmp_ch_stitch_image = np.copy(ch_stitch_image)
    pseudo_move_image = np.full((2048,2048), 0, dtype=np.uint16)
    # transformation + rotation
    
    move_image = _applyRegisterMatrix(move_image, shift)
    pseudo_move_image = _applyRegisterMatrix(pseudo_move_image, shift)
    
    if isinstance(transform_parameter_map, str):
        # Channel translation
        pseudo_index = np.where(pseudo_move_image == 0)
        pass
        # pseudo_move_image = pseudo_image_shift
    elif phase_cross_correction:
        tmp_move = np.full((5000,5000), 65535, dtype=np.uint16)
        tmp_move[1476:3524,1476:3524] = move_image
        move_image = _applyRegisterMatrix(tmp_move, transform_parameter_map)
        tmp_pseudo = np.full((5000,5000), 65535, dtype=np.uint16)
        tmp_pseudo[1476:3524,1476:3524] = 0
        pseudo_move_image = _applyRegisterMatrix(tmp_pseudo, transform_parameter_map)
        tmp_index = np.where(pseudo_move_image == 0)
        pseudo_index = (tmp_index[0] - 1476, tmp_index[1] - 1476)
        
    else:
        move_image = _applyEulerTransform2048(transform_parameter_map, sitk.GetImageFromArray(move_image))
        move_image = sitk.GetArrayFromImage(move_image)
        pseudo_move_image = _applyEulerTransform2048(transform_parameter_map, sitk.GetImageFromArray(pseudo_move_image))
        pseudo_move_image = sitk.GetArrayFromImage(pseudo_move_image)
        pseudo_index = np.where(pseudo_move_image == 0)
        
    y_pos = GLOBAL_GRID.loc[i,"y_pos2"]
    x_pos = GLOBAL_GRID.loc[i,"x_pos2"]
    
    # Boundary handling
    d_dim1, d_dim2, m_dim1, m_dim2 = _boundary_handle(pseudo_index, y_pos, x_pos, STITCHED_IMAGE_SIZE)
    
    if isinstance(transform_parameter_map, str):
        tmp_stitch_image[d_dim1, d_dim2] = move_image[m_dim1, m_dim2]
    elif phase_cross_correction:
        tmp_stitch_image[d_dim1, d_dim2] = move_image[m_dim1+1476, m_dim2+1476]
    else:
        tmp_stitch_image[d_dim1, d_dim2] = move_image[m_dim1, m_dim2]

    final_index = np.where(tmp_stitch_image > 0)

    tmp_ch_stitch_image[final_index] = tmp_stitch_image[final_index]
    
    return tmp_ch_stitch_image


# Stitch other channels
def _tile_stitching(rd_ch, Para, STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID):
    
    rd = rd_ch[0]
    ch = rd_ch[1]
    
    ch_stitch_image = np.copy(STITCHED_IMAGE)
    for i,m in tqdm(enumerate(Para.tile_numbers), total = len(Para.tile_numbers), leave = True, desc = f"[INFO] Stitch cycle {rd}, channel {ch}"):

        if rd == Para.Anchor_Round:

            transform_parameter_map = "PASS"

        else:

            FIX = sitk.ReadImage(os.path.join(Para.output, f"Registration/unstitched/{Para.Anchor_Round}_DAPI/{Para.Anchor_Round}_{m}_AF405.tif"))
            MOVE = sitk.ReadImage(os.path.join(Para.output, f"Registration/unstitched/{rd}_DAPI/{rd}_{m}_AF405.tif"))
            
            if Para.phase_cross_correction:
                
                transform_parameter_map = _getRegisterMatrix(sitk.GetArrayFromImage(FIX), sitk.GetArrayFromImage(MOVE))
            
            else:

                transform_parameter_map = _EulerTransform(FIX, MOVE)

        move_image = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/{rd}/{rd}_{m}_{ch}.tif"))

        ch_stitch_image = _stitch_handle(move_image, transform_parameter_map, ch_stitch_image, i,  
                   STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID, Para.channel_shift[ch], Para.phase_cross_correction)

    ch_stitch_image = _apply_clahe(ch_stitch_image, clipLimit=Para.CLAHE_clip[ch])
    
    # 这里再加一步去掉过曝点，CLAHE似乎增加过亮的点
    channel_clip = max(np.percentile(ch_stitch_image, 99.9), Para.channel_clip[ch] * 1.5)
    channel_mask = ch_stitch_image > channel_clip
    # channel_mask = ch_stitch_image > Para.channel_clip[ch] * 1.5
                
    if np.sum(channel_mask) > 1:
        min_val = np.min(ch_stitch_image[channel_mask])
        max_val = np.max(ch_stitch_image[channel_mask])
        if min_val != max_val:
            # image_in[channel_mask] = (image_in[channel_mask] - min_val) / (max_val - min_val) * Para.channel_clip[channel] * 0.1 + Para.channel_clip[channel] * 0.9
            ch_stitch_image[channel_mask] = (ch_stitch_image[channel_mask] - min_val) / (max_val - min_val) * channel_clip * 0.1
        else:
            ch_stitch_image[channel_mask] = 0
    else:
        ch_stitch_image[channel_mask] = 0
    
    tifffile.imwrite(os.path.join(Para.output, f"Registration/stitched/processed/{rd}/{rd}_{ch}.tif"), ch_stitch_image)


def multiCycleRegister(Para):
    
    DAPI_file = glob.glob(os.path.join(Para.output, f"Registration/unstitched/{Para.Anchor_Round}_DAPI/*AF405.tif"))
    xml_file = glob.glob(os.path.join(Para[Para.Anchor_Round], "*_info.xml"))[0]
    
    info = "[INFO] Running MIST..."
    print(info)
    
    IMAGES, startx, starty = _prepareMIST(merged_file = DAPI_file, 
                                          xmlfile = xml_file, 
                                          channel_list=Para.Round_channel[Para.Anchor_Round])
    
    try:
        GLOBAL_GRID = pd.read_csv(os.path.join(Para.output, "tmp/MIST_grid.csv"), index_col = 0)
        info = "[INFO] Using existing MIST results."
        print(info)
    except FileNotFoundError:
        GLOBAL_GRID = _runMIST(images = IMAGES, startx = startx, starty = starty)
        GLOBAL_GRID.to_csv(os.path.join(Para.output, "tmp/MIST_grid.csv"))
        
    STITCHED_IMAGE_SIZE = (
        GLOBAL_GRID["y_pos2"].max() + 2048,
        GLOBAL_GRID["x_pos2"].max() + 2048,
    )
    STITCHED_IMAGE = np.zeros_like(IMAGES, shape=STITCHED_IMAGE_SIZE)
    
    
    # Stitch DAPI channel
    # info = "[INFO] Stitching DAPI channel..."
    # print(info)
    DAPI = np.stack([tifffile.imread(x) for x in DAPI_file], axis = 0)
    # stitched_DAPI = np.copy(STITCHED_IMAGE)
    # for i, row in GLOBAL_GRID.iterrows():
    #     stitched_DAPI[
    #         row["y_pos2"] : row["y_pos2"] + 2048,
    #         row["x_pos2"] : row["x_pos2"] + 2048,
    #     ] = _applyRegisterMatrix(DAPI[i], Para.channel_shift['AF405'])
        
    transform_parameter_map = "PASS"
    DAPI_stitch_image = np.copy(STITCHED_IMAGE)
    
    for i,tile in tqdm(enumerate(DAPI), total = len(Para.tile_numbers), desc = f"[INFO] Stitching DAPI channel..."): 

        shift = Para.channel_shift['AF405']
        DAPI_stitch_image = _stitch_handle(tile, transform_parameter_map, DAPI_stitch_image,i,STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID, 
                                           shift, Para.phase_cross_correction)
    
    DAPI_stitch_image = _apply_clahe(DAPI_stitch_image, clipLimit = 10.0)
    tifffile.imwrite(os.path.join(Para.output, "Registration/Stitched_DAPI.tif"), DAPI_stitch_image)
        
    
    rds_chs = []
    for rd, channels in Para.Round_channel.items():
        for ch in channels:
            # if Para.phase_cross_correction:
            #     info = f"[INFO] Stitching in sequential..."
            #     print(info)
            #     _tile_stitching([rd,ch], Para, STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID)
            # else:
            if ch in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']:
                rds_chs.append([rd,ch])
        
    # for ch in channels[1:]:
    #     _tile_stitching(rd, ch, Para, STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID)
    # if not Para.phase_cross_correction:

    # info = f"[INFO] Stitching in sequencial..."
    # print(info)
    # for rd_ch in rds_chs:
    #     _tile_stitching(rd_ch, Para, STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID)
    info = f"[INFO] Stitching in palallel..."
    print(info)  
    Parallel(n_jobs = 12, backend='loky')(delayed(_tile_stitching)(rd_ch, Para, STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID) for rd_ch in rds_chs)
    
    # Stitch rRNA channel and extra round
    if Para.rRNAseg:
        
        # info = f"[INFO] Stitching rRNA..."
        # print(info)
        
        transform_parameter_map = "PASS"
        DAPI_stitch_image = np.copy(STITCHED_IMAGE)
        rRNA_stitch_image = np.copy(STITCHED_IMAGE)
        
        C = 2
        if Para.extraseg:
            
            extra_stitch_image = {}
            for ech in Para.extra_channels:
                if ech in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7', 'DIC']:
                    extra_stitch_image[ech] = np.copy(STITCHED_IMAGE)
            
                    C += 1
        if C < 3:
            C = 3
        
        for i,m in tqdm(enumerate(Para.tile_numbers), total = len(Para.tile_numbers), desc = f"[INFO] Stitching rRNA cycle..."): 
            
            FIX = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/{Para.Anchor_Round}_DAPI/{Para.Anchor_Round}_{m}_AF405.tif"))
            MOVE = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/rRNA_DAPI/rRNA_DAPI_{m}_DAPI.tif"))

            shift = _getRegisterMatrix(FIX, MOVE)

            move_image = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/rRNA_DAPI/rRNA_DAPI_{m}_rRNA.tif"))
            
            DAPI_stitch_image = _stitch_handle(MOVE, transform_parameter_map, DAPI_stitch_image,i,STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID, 
                                               shift + Para.channel_shift['AF405'], Para.phase_cross_correction)
            
            rRNA_stitch_image = _stitch_handle(move_image, transform_parameter_map, rRNA_stitch_image,i,STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID, 
                                               shift + Para.channel_shift['AF546'], Para.phase_cross_correction)
            
            if Para.extraseg:
                
                for ech in Para.extra_channels:
                    
                    if ech in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7', 'DIC']:
                        MOVE = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/extra/extra_{m}_DAPI.tif"))
                        shift = _getRegisterMatrix(FIX, MOVE)
                        move_image = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/extra/extra_{m}_{ech}.tif"))
                        extra_stitch_image[ech] = _stitch_handle(move_image, transform_parameter_map, extra_stitch_image[ech],i,STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID, 
                                                                 shift + Para.channel_shift[ech], Para.phase_cross_correction)
                    
                    
        width, height = STITCHED_IMAGE_SIZE
        DAPI_stitch_image = _apply_clahe(DAPI_stitch_image, clipLimit=10.0)
        rRNA_stitch_image = _apply_clahe(rRNA_stitch_image, clipLimit=3.5)
        
        # Image for Xenium
        new_image = np.zeros((C, width, height), dtype=np.uint16)
        # Image for Cellpose3
        new_image_rRNA = np.zeros((width, height, 3), dtype=np.uint16)
        
        new_image_rRNA[:,:,0] = rRNA_stitch_image
        new_image_rRNA[:,:,2] = DAPI_stitch_image
        
        new_image[0,:,:] = rRNA_stitch_image
        new_image[1,:,:] = DAPI_stitch_image
        
        if Para.extraseg:
            i = 2
            for ech in Para.extra_channels:
                if ech in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']:
                    extra_stitch_image[ech] = _apply_clahe(extra_stitch_image[ech], clipLimit=Para.CLAHE_clip[ech])
                    new_image[i,:,:] = extra_stitch_image[ech]
                    i += 1
                elif ech == 'DIC':
                    new_image[i,:,:] = extra_stitch_image[ech]
                    i += 1
            
        tifffile.imwrite(os.path.join(Para.output, "Registration/Stitched_rRNA_DAPI.tif"), new_image_rRNA)
        tifffile.imwrite(os.path.join(Para.output, "Registration/Stitched_CWH.tif"), new_image)
        
        
    elif Para.extraseg:
        
        # info = f"[INFO] Stitching extra..."
        # print(info)
        
        transform_parameter_map = "PASS"
        DAPI_stitch_image = np.copy(STITCHED_IMAGE)
        
        C = 2
        
        extra_stitch_image = {}
        for ech in Para.extra_channels:
            
            if ech in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7', 'DIC']:
                extra_stitch_image[ech] = np.copy(STITCHED_IMAGE)

                C += 1
                
        if C < 3:
            C = 3
        
        for i,m in tqdm(enumerate(Para.tile_numbers), total = len(Para.tile_numbers), desc = f"[INFO] Stitching extra cycle..."): 
            
            FIX = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/{Para.Anchor_Round}_DAPI/{Para.Anchor_Round}_{m}_AF405.tif"))
                
            for ech in Para.extra_channels:

                if ech in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7', 'DIC']:
                    MOVE = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/extra/extra_{m}_DAPI.tif"))
                    shift = _getRegisterMatrix(FIX, MOVE)
                    # shift = shift + 
                    move_image = tifffile.imread(os.path.join(Para.output, f"Registration/unstitched/extra/extra_{m}_{ech}.tif"))
                    extra_stitch_image[ech] = _stitch_handle(move_image, transform_parameter_map, extra_stitch_image[ech],i,STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID, 
                                                             shift + Para.channel_shift[ech], Para.phase_cross_correction)
                    DAPI_stitch_image = _stitch_handle(MOVE, transform_parameter_map, DAPI_stitch_image,i,STITCHED_IMAGE, STITCHED_IMAGE_SIZE, GLOBAL_GRID, 
                                                       shift + Para.channel_shift['AF405'], Para.phase_cross_correction)
                    
                    
        width, height = STITCHED_IMAGE_SIZE
        DAPI_stitch_image = _apply_clahe(DAPI_stitch_image, clipLimit=10.0)
        
        # Image for Xenium
        new_image = np.zeros((C, width, height), dtype=np.uint16)
        new_image[0,:,:] = DAPI_stitch_image
        
        i = 1
        for ech in Para.extra_channels:
            if ech in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']:
                extra_stitch_image[ech] = _apply_clahe(extra_stitch_image[ech], clipLimit=Para.CLAHE_clip[ech])
                new_image[i,:,:] = extra_stitch_image[ech]
                i += 1
            elif ech == 'DIC':
                new_image[i,:,:] = extra_stitch_image[ech]
                i += 1

        tifffile.imwrite(os.path.join(Para.output, "Registration/Stitched_CWH.tif"), new_image)
        
    
    else:
        
        pass
        
            
            