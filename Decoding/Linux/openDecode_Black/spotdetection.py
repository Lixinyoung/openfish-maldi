import warnings
warnings.filterwarnings('ignore')

from skimage.io import imread
from spotiflow.model import Spotiflow
from tqdm import tqdm
import pandas as pd

import subprocess
import os



def _run_single_spotiflow(img,
                          model = None,
                          intensity_threshold = 100,
                          prob_thresh = None,
                          n_tiles = (3, 3),
                          min_distance = 1, 
                          exclude_border = True,
                          scale = None,
                          subpix = True,
                          peak_mode ='fast',
                          normalizer = 'auto',
                          verbose = True,
                          device = 'cuda', **kargs):

    spots, details = model.predict(
                img,
                prob_thresh = prob_thresh,
                n_tiles = n_tiles,
                min_distance = min_distance, 
                exclude_border = exclude_border,
                scale = scale,
                subpix = subpix,
                peak_mode = peak_mode,
                normalizer = normalizer,
                verbose = verbose,
                device = device, **kargs
            ) # predict expects a numpy array

    df = pd.DataFrame({
                'x': spots[:,1],
                'y': spots[:,0],
                'intensity': details.prob,
                'threshold': details.intens.flatten()
            })
    
    return df[df['threshold'] > intensity_threshold].copy()


def runSpotiflow(para,
                model_name = "general",
                intensity_threshold = {
                    'AF488': 100, 'AF546': 100, 'AF594': 1000, 'Cy5': 1000, 'Cy7': 1000
                },
                prob_thresh = None,
                min_distance = 1,
                exclude_border = True,
                scale = None,
                subpix = True,
                peak_mode ='fast',
                normalizer = 'auto',
                verbose = False,
                device = 'cuda', 
                force = False,
                **kargs
                ):
    
    model = Spotiflow.from_pretrained(model_name)
    tqdm_bar = tqdm(total = 100, bar_format = '{l_bar}{bar}| {n:.0f}/{total} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]', dynamic_ncols = True)
    tqdm_step = 100/sum([len(x) - 1 for x in para.CHANNEL_INFO.values()])
    
    for rd in para.CYCLES:
        for ch in para.CHANNEL_INFO[rd]:
            
            PROGRESS_DICT = para._parse_progress_yaml()
            
            if ch != para.ANCHOR_CHANNEL and (not PROGRESS_DICT['spot_detection'][rd][ch] or force) :
                tqdm_bar.set_description(f'Running Spotiflow: {rd}_{ch}')
                
                input_image = imread(os.path.join(para.OUTPUT_PATH, f"Registration/stitched/{rd}/{rd}_{ch}.tif"))
                output_df = os.path.join(para.OUTPUT_PATH, f"Registration/stitched/{rd}/{rd}_{ch}.parquet")
    
                n_tiles = (input_image.shape[0]//1024, input_image.shape[1]//1024)
                
                df = _run_single_spotiflow(
                    input_image, 
                    intensity_threshold = intensity_threshold[ch],
                    model = model,
                    prob_thresh = prob_thresh,
                    n_tiles = n_tiles,
                    min_distance = min_distance, 
                    exclude_border = exclude_border,
                    scale = scale,
                    subpix = subpix,
                    peak_mode = peak_mode,
                    normalizer = normalizer,
                    verbose = verbose,
                    device = device, **kargs)
                
                df.to_parquet(output_df, index = False)
                
                PROGRESS_DICT['spot_detection'][rd][ch] = True
                para.save_progress_yaml(PROGRESS_DICT)
    
                tqdm_bar.update(tqdm_step)

    tqdm_bar.close()