import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import m2stitch
import cv2
import os

def _prepareMIST(merged_file: list, xmlfile: str, channel_list:list):
    """
    Prepare the input for m2stitch
    """
    
    channel_num = len(channel_list)
    images = np.stack([cv2.imread(x,-1) for x in merged_file], axis = 0)
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    startx = []
    starty = []
    tile_num = int(len(root)/channel_num)
    for child in root[0: tile_num]:
        startx.append(int(child[1].attrib["StartX"]))
        starty.append(int(child[1].attrib["StartY"]))
    return images, startx, starty

def _runMIST(images: np.ndarray, startx: list, starty: list):
    """
    Run Mist
    """

    rows = np.array(starty) // 1843 + 1
    cols = np.array(startx) // 1843 + 1
    
    position_initial_guess = np.array([starty, startx]).T
    
    result_df = "MIST CHECK"
    
    ncc_ts = [0.05,0.025,0.01,0.005,0.001]
    for ncc_t in ncc_ts:
        
        info = "[INFO] Trying MIST with ncc_threshod = {}".format(ncc_t)
        print(info)
        try:
            result_df, _ = m2stitch.stitch_images(images, rows, cols, row_col_transpose=False, ncc_threshold = ncc_t, 
                                          position_initial_guess = position_initial_guess, overlap_diff_threshold=5, pou=2)
            break
        except:
            continue
    
    if isinstance(result_df, str):
        info = "[INFO] MIST failed, continue with initial guess..."
        print(info)
        result_df = pd.DataFrame({"row": rows,
                                  "col": cols,
                                  "y_pos": starty,
                                  "x_pos": startx})
    
    result_df["y_pos2"] = result_df["y_pos"] - result_df["y_pos"].min()
    result_df["x_pos2"] = result_df["x_pos"] - result_df["x_pos"].min()
    
    return result_df