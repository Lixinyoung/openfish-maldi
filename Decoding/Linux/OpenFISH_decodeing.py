
"""
该Nscript是V11版本的OpenFISH decoding代码，特性如下：
接受的输入为zen导出的小图（tile），且需要对应的xml文件；
全程都可以在notebook内部完成
使用的包见 D:/openFISH/Decoding/openDecode
主要包含图像对齐-图像拼接-信号点识别-图像解码四个部分
图像对齐使用SimpleElastix，图像拼接使用m2stitch，信号点识别使用RS-FISH，图像解码使用PoSTCode
支持多模态分割与DAPI膨胀
"""

# 1.填写参数（最重要的部分，之后的函数中有一些参数也可以根据需要修改）

filename = {
    "R1":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R1_depth_deblur",
    "R2":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R2_depth_deblur",
    "R3":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R3_depth_deblur",
    "R4":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R4_depth_deblur",
    "R5":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R5_depth_deblur",
    "R6":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R6_depth_deblur",
    "R7":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R7_depth_deblur",
    "R8":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R8_depth_deblur",
    "R9":"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R9_depth_deblur",
    'R10':"/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R10_depth_deblur",
    'R11':'/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_R11_depth_deblur'
} 


# CodingBooks，如果方法是10N，具体格式参考D:/openFISH/Decoding/CodingBook_10N.csv
# CodingBooks，如果方法是DDC，具体格式参考D:/openFISH/Decoding/CodingBook_MultiCycle.csv
# 空缺的编码请用FP1，FP2等或者任意不重复字符替代
CodingBook = "/media/duan/sda2/MALDI/Data/20250727_Saline_MALDI/MALDI_CodingBook.CSV"


method_type = "10N" # 10N 或者 MultiCycle
workpath = "./"
output_path = "Result"
objective = "20X" # 20X or 40X
rRNA= "/media/duan/sda2/MALDI/Data/20250727_Saline_noMALDI/20250727_Saline2_6_rRNA_depth" # path or None
extra = None # path or None
baysor = False # True or False
phase_cross_correction = False # 不同轮之间位移较大（x或y大于50个像素时，填True）



import sys
sys.path.append("/media/duan/DuanLab_Data/openFISH/Decode")

from openDecode.Preprocess import Para, tilePreprocess
from openDecode.TileStitch import multiCycleRegister
from openDecode.ThirdParty import runRSFISHforTiles, runBaysor, runSpotiflow
from openDecode.PointsCalling import filterPoints
from openDecode.Decoding import Decoding
from openDecode.Visualization import *

# 2.图像对齐与拼接

# i.初始化参数，新建output文件夹，重构codebook
para = Para(filename = filename, CodingBook = CodingBook, method_type = method_type, 
                 workpath = workpath,  output_path = output_path, objective = objective, rRNA = rRNA,
                 extra = extra, baysor = baysor, phase_cross_correction = phase_cross_correction)


# ii.原始图像预处理
tilePreprocess(para)

multiCycleRegister(para)

runSpotiflow(para,
             model_name = "hybiss",
             intensity_threshold = { 
                    'AF488': 500, 'AF546': 1000, 'AF594': 1000, 'Cy5': 500, 'Cy7': 100
                },
             prob_thresh = 0.40,
             min_distance = 1,
             exclude_border = True,
             subpix = True,
             peak_mode ='fast',
             normalizer = 'auto',
             device = 'cuda')




# # 4.解码
Spots_Dict, image_size = filterPoints(para)

with open(os.path.join(para.output, "tmp/image_size.txt"), 'w') as handle:
    handle.write(str(image_size))
    
# image_size = (22374, 53488)


# import numpy as np
# Spots_Dict = np.load("/media/duan/sda2/MALDI/Data/20250721_Saline_MALDI/Result/tmp/Spots_Dict.npy", allow_pickle=True).item()


Final_df = Decoding(Spots_Dict, para)


plotGene(Final_df, para, image_size)
plot_probs(Final_df, para)

# import pandas as pd
# Final_df = pd.read_parquet("/media/duan/DuanLab_Data/openFISH/Decode/Data/20250320ABADemo_Rep2/ResultRep2/Decoded_filtered.parquet")


if para.baysor == True:
    import os
    Final_df.to_csv(os.path.join(para.output, "gene_location_merged_filtered.csv"))
    runBaysor(para, min_molecules_per_cell = 75, count_matrix_format = "loom", prior_segmentation_confidence = 0.3, scale = 40, scale_std = "25%")
    
plotXenium(Final_df, para, stardist_roi)