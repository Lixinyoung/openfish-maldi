"""
Author: Li Xinyang
Last modified: 2025.9.25


Change log:

    2025.9.25:
        New script
    
    2025.10.03
        Fix the BaSiC GPU memory mangement code
"""

import logging

# from skimage.restoration import richardson_lucy
from cucim.skimage import restoration
from basicpy import BaSiC
from tqdm import tqdm
import pandas as pd
import numpy as np
import cupy as cp
import cv2
import os
import multiprocessing as mp


log = logging.getLogger(__name__)

def _keep_xy(img: np.ndarray) -> np.ndarray:
    
    return img.reshape(-1, *img.shape[-2:])[0]


def _clip_normalize(img: np.ndarray, pmin:float = 0, pmax:float = 99.0) -> np.ndarray:
        
    low_value, high_value = np.percentile(img, [pmin, pmax], method='linear')
    
    img = np.clip(img, 0, 3.0 * high_value)
    
    img[img <= low_value] = 0
    
    return img


# def _deblur(img:np.ndarray, kernel_size: int = 5) -> np.ndarray:
    
#     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
#     deblurred = cv2.filter2D(img, -1, kernel)
#     unsharp = cv2.addWeighted(img, 1.5, deblurred, -0.5, 0)
    
#     return unsharp

def _deblur(img: np.ndarray):
    
    psf = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 2.53524673e-09, 6.02846919e-08, 7.08883012e-07,
        4.12214059e-06, 1.18536401e-05, 1.68562663e-05, 1.18536401e-05,
        4.12214059e-06, 7.08883012e-07, 6.02846919e-08, 2.53524673e-09,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 6.02846919e-08, 1.43348734e-06, 1.68562663e-05,
        9.80188526e-05, 2.81863313e-04, 4.00818907e-04, 2.81863313e-04,
        9.80188526e-05, 1.68562663e-05, 1.43348734e-06, 6.02846919e-08,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 7.08883012e-07, 1.68562663e-05, 1.98211527e-04,
        1.15259608e-03, 3.31440882e-03, 4.71319840e-03, 3.31440882e-03,
        1.15259608e-03, 1.98211527e-04, 1.68562663e-05, 7.08883012e-07,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.12214059e-06, 9.80188526e-05, 1.15259608e-03,
        6.70232322e-03, 1.92732212e-02, 2.74071548e-02, 1.92732212e-02,
        6.70232322e-03, 1.15259608e-03, 9.80188526e-05, 4.12214059e-06,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.18536401e-05, 2.81863313e-04, 3.31440882e-03,
        1.92732212e-02, 5.54221341e-02, 7.88120984e-02, 5.54221341e-02,
        1.92732212e-02, 3.31440882e-03, 2.81863313e-04, 1.18536401e-05,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.68562663e-05, 4.00818907e-04, 4.71319840e-03,
        2.74071548e-02, 7.88120984e-02, 1.12073397e-01, 7.88120984e-02,
        2.74071548e-02, 4.71319840e-03, 4.00818907e-04, 1.68562663e-05,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 1.18536401e-05, 2.81863313e-04, 3.31440882e-03,
        1.92732212e-02, 5.54221341e-02, 7.88120984e-02, 5.54221341e-02,
        1.92732212e-02, 3.31440882e-03, 2.81863313e-04, 1.18536401e-05,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.12214059e-06, 9.80188526e-05, 1.15259608e-03,
        6.70232322e-03, 1.92732212e-02, 2.74071548e-02, 1.92732212e-02,
        6.70232322e-03, 1.15259608e-03, 9.80188526e-05, 4.12214059e-06,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 7.08883012e-07, 1.68562663e-05, 1.98211527e-04,
        1.15259608e-03, 3.31440882e-03, 4.71319840e-03, 3.31440882e-03,
        1.15259608e-03, 1.98211527e-04, 1.68562663e-05, 7.08883012e-07,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 6.02846919e-08, 1.43348734e-06, 1.68562663e-05,
        9.80188526e-05, 2.81863313e-04, 4.00818907e-04, 2.81863313e-04,
        9.80188526e-05, 1.68562663e-05, 1.43348734e-06, 6.02846919e-08,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 2.53524673e-09, 6.02846919e-08, 7.08883012e-07,
        4.12214059e-06, 1.18536401e-05, 1.68562663e-05, 1.18536401e-05,
        4.12214059e-06, 7.08883012e-07, 6.02846919e-08, 2.53524673e-09,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00]])
    
    img_max = img.max()
    # img_max = 65535
    
    img = img / img_max
    
    img_gpu = cp.asarray(img)
    psf_gpu = cp.asarray(psf)

    deconvolved_gpu = restoration.richardson_lucy(
        img_gpu,
        psf_gpu,
        num_iter=5, 
        clip=True
    )

    deconvolved = cp.asnumpy(deconvolved_gpu)
    
    # del deconvolved_gpu, img_gpu, psf_gpu
    # cp.get_default_memory_pool().free_all_blocks()
    # cp.get_default_pinned_memory_pool().free_all_blocks()

    return (deconvolved * img_max).astype(np.uint16)

# BaSiC do not release the gpu memory
def _BaSiC(npy_file_path, max_workers: int = 48, max_reweight_iterations:int = 200) -> np.ndarray:
    
    with open("run_BaSiC.py", 'w') as handle:
        
        handle.write(
        f"""
from basicpy import BaSiC
import numpy as np

images = np.load('{npy_file_path}')

basic = BaSiC(get_darkfield=True, smoothness_flatfield=1, max_workers = {max_workers}, max_reweight_iterations = {max_reweight_iterations})
basic.fit(images)

np.save('{npy_file_path}', basic.transform(images))
        """)
    
    import subprocess
    
    subprocess.run(f'python run_BaSiC.py', shell = True, check = True)
    
    os.remove("run_BaSiC.py")
    
    return None


def _preprocess_channel(ch: int, M: int, czi, name: str, npy_path = None,
                        max_workers:int = 48, max_reweight_iterations: int = 200, 
                        run_deblur: bool = True, run_BaSiC: bool = True, 
                        **kwargs) -> None:
    
    image_list = []
    
    for m in range(M):
        img,_ = czi.read_image(M = m, C = ch)
        img = _keep_xy(img)
        img = _clip_normalize(img)
        
        if run_deblur:
            img = _deblur(img)
            
            
        image_list.append(img)
        # del img
        # cp.get_default_memory_pool().free_all_blocks()
        # cp.get_default_pinned_memory_pool().free_all_blocks()
    
    image = np.stack(image_list)
        
    if run_BaSiC:
        
        np.save(npy_path, image)
        _BaSiC(npy_path, max_workers = max_workers, max_reweight_iterations = max_reweight_iterations)
        image = np.load(npy_path)
        
    return image.astype(np.uint16)
    
    

def preprocess(para, force: bool = False):
    

    log.info("Preprocessing transcripts cycles")
    for rd,czi in tqdm(para.CZI_FILES.items(), total = len(para.CYCLES), dynamic_ncols = True):
        
        for i,ch in enumerate(para.CHANNEL_INFO[rd]):
            
            PROGRESS_DICT = para._parse_progress_yaml()
            
            if not PROGRESS_DICT['preprocess'][rd][ch] or force:
            
                run_deblur = para.run_deblur
                run_BaSiC = para.run_BaSiC

                if ch == para.ANCHOR_CHANNEL:
                    run_deblur = False
                    run_BaSiC = False
                    
                npy_path = os.path.join(para.OUTPUT_PATH, f"Registration/unstitched/{rd}/{ch}.npy")

                image = _preprocess_channel(i, para.TILES_NUMBER, czi, rd, npy_path = npy_path, max_workers = para.THREADS, 
                                            run_deblur = run_deblur, run_BaSiC = run_BaSiC)
                
                np.save(npy_path, image)
                
                PROGRESS_DICT['preprocess'][rd][ch] = True
                
                para.save_progress_yaml(PROGRESS_DICT)
                
        
    if para.EXTRA:
        
        log.info("Preprocessing extra transcripts cycles")

        for name,czi in tqdm(para.EXTRA_CZI_FILES.items(), total = len(para.EXTRA_NAMES), dynamic_ncols = True):

            for i,ch in enumerate(para.EXTRA_CHANNEL_INFO[name]):
                
                PROGRESS_DICT = para._parse_progress_yaml()

                if ch != 'SKIP':
                    
                    if not PROGRESS_DICT['preprocess'][name][ch] or force:
                        
                        npy_path = os.path.join(para.OUTPUT_PATH, f"Registration/unstitched/{name}/{ch}.npy")

                        image = _preprocess_channel(i, para.TILES_NUMBER, czi, name, npy_path = npy_path, max_workers = para.THREADS, 
                                                    run_deblur = False, run_BaSiC = para.run_BaSiC)

                        np.save(npy_path, image)
                        
                        PROGRESS_DICT['preprocess'][name][ch] = True
                
                        para.save_progress_yaml(PROGRESS_DICT)

                else:
                    continue