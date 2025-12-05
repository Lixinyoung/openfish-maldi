"""
Author: Li Xinyang
Last modified: 2025.9.25


Change log:

    2025.9.25:
        New script
"""

import logging

# from skimage.restoration import richardson_lucy
# from cucim.skimage import restoration

import torch
import torch.nn.functional as F

from basicpy import BaSiC
from tqdm import tqdm
import pandas as pd
import numpy as np
# import cupy as cp
import cv2
import os

log = logging.getLogger(__name__)

def _keep_xy(img: np.ndarray) -> np.ndarray:
    
    return img.reshape(-1, *img.shape[-2:])[0]


def _clip_normalize(img: np.ndarray, pmin:float = 0, pmax:float = 99.0) -> np.ndarray:
        
    low_value, high_value = np.percentile(img, [pmin, pmax], method='linear')
    
    img = np.clip(img, 0, 3 * high_value)
    
    img[img <= low_value] = 0
    
    return img


class RLDeconvolution:

    
    def __init__(self, psf, num_iter=5, device='cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_iter = num_iter
        
        self.psf_tensor = torch.from_numpy(psf).float().to(self.device)
        if self.psf_tensor.ndim == 2:
            self.psf_tensor = self.psf_tensor.unsqueeze(0).unsqueeze(0)
        
        self.psf_mirror = torch.flip(self.psf_tensor, dims=[2, 3])
        
        self._warm_up()
    
    def _warm_up(self):

        dummy_input = torch.randn(1, 1, 64, 64, device=self.device)
        _ = self._process_batch(dummy_input)
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
    
    def _process_batch(self, batch_images):

        deconvolved = torch.full_like(batch_images, 0.5)
        
        for _ in range(self.num_iter):
            est_conv = F.conv2d(deconvolved, self.psf_tensor, padding='same')
            relative_blur = batch_images / (est_conv + 1e-12)
            im_rev_conv = F.conv2d(relative_blur, self.psf_mirror, padding='same')
            deconvolved = deconvolved * im_rev_conv
        
        return deconvolved
    
    def process_images(self, images, batch_size=8, clip=True):

        img_maxs = [x.max() for x in images]
        images = np.stack([x / y for x,y in zip(images, img_maxs)])
        
        images_tensor = torch.from_numpy(images).float().to(self.device)
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(1)  # (N, 1, H, W)
        
        num_images = images_tensor.shape[0]
        all_results = []
        
        for i in range(0, num_images, batch_size):
            batch_end = min(i + batch_size, num_images)
            batch_images = images_tensor[i:batch_end]
            
            with torch.no_grad():
                deconvolved = self._process_batch(batch_images)
            
            if clip:
                deconvolved = torch.clamp(deconvolved, 0, 1)
            
            all_results.append(deconvolved.squeeze(1).cpu().numpy())

        deconvolved_images = np.vstack(all_results) if len(all_results) > 1 else all_results[0]

        deconvolved_images = np.stack([x * y for x,y in zip(deconvolved_images, img_maxs)])
        
        return deconvolved_images.astype(np.uint16)


def _BaSiC(images: np.ndarray, max_workers: int = 48, max_reweight_iterations:int = 200, **kwargs) -> np.ndarray:
    
    basic = BaSiC(get_darkfield=True, smoothness_flatfield=1, max_workers = max_workers, max_reweight_iterations = max_reweight_iterations, **kwargs)
    basic.fit(images)
    
    return basic.transform(images)


def _preprocess_channel(ch: int, M: int, czi, name: str,
                        max_workers:int = 48, max_reweight_iterations: int = 200, 
                        run_deblur: bool = True, run_BaSiC: bool = True, 
                        **kwargs) -> None:
    
    image_list = []
    
    for m in range(M):
        img,_ = czi.read_image(M = m, C = ch)
        img = _keep_xy(img)
        img = _clip_normalize(img)
        
        # if run_deblur:
        #     img = _deblur(img)
            
        image_list.append(img)
    
    image = np.stack(image_list)

    if run_deblur:

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

        deconv_processor = RLDeconvolution(psf, num_iter=5)
        image = deconv_processor.process_images(image, batch_size=8)
        
    if run_BaSiC:

        image = _BaSiC(image, max_workers = max_workers, max_reweight_iterations = max_reweight_iterations, **kwargs)
            
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

                image = _preprocess_channel(i, para.TILES_NUMBER, czi, rd, max_workers = para.THREADS, 
                                            run_deblur = run_deblur, run_BaSiC = run_BaSiC)

                if torch.cuda.is_available() and run_deblur:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                np.save(os.path.join(para.OUTPUT_PATH, f"Registration/unstitched/{rd}/{ch}.npy"), image)
                
                PROGRESS_DICT['preprocess'][rd][ch] = True
                
                para.save_progress_yaml(PROGRESS_DICT)
                
        
    if para.EXTRA:
        
        log.info("Preprocessing extra transcripts cycles")

        for name,czi in tqdm(para.EXTRA_CZI_FILES.items(), total = len(para.EXTRA_NAMES), dynamic_ncols = True):

            for i,ch in enumerate(para.EXTRA_CHANNEL_INFO[name]):
                
                PROGRESS_DICT = para._parse_progress_yaml()

                if ch != 'SKIP':
                    
                    if not PROGRESS_DICT['preprocess'][name][ch] or force:

                        image = _preprocess_channel(i, para.TILES_NUMBER, czi, name, max_workers = para.THREADS, 
                                                    run_deblur = False, run_BaSiC = para.run_BaSiC)

                        np.save(os.path.join(para.OUTPUT_PATH, f"Registration/unstitched/{name}/{ch}.npy"), image)
                        
                        PROGRESS_DICT['preprocess'][name][ch] = True
                
                        para.save_progress_yaml(PROGRESS_DICT)

                else:
                    continue