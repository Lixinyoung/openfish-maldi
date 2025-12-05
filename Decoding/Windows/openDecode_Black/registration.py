"""
Author: Li Xinyang
Last modified: 2025.9.26


Change log:

    2025.9.25:
        New script
        
    2025.9.26
        Finish first version

    2025.10.01
        Split progress yaml registration into registration and stitching
"""
import logging

from skimage.registration import phase_cross_correlation
from skimage import transform
from joblib import Parallel, delayed
from collections import defaultdict
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import numpy as np
import tifffile
import m2stitch
import yaml
import cv2
import os

log = logging.getLogger(__name__)

from .utils import is_all_true


# def _getRegisterMatrix(ref:np.ndarray, dst: np.ndarray):
#     shift, _, _ = phase_cross_correlation(ref, dst, upsample_factor=100, overlap_ratio=0.5, normalization = 'phase')
#     return shift


class Stitching_Registration():
    
    def __init__(self, para):
        
        self.para = para
        
    def _runMIST(self):
        """
        Run Mist
        """
        czi = self.para.CZI_FILES[self.para.ANCHOR_CYCLE]
        ch = self.para.CHANNEL_INFO[self.para.ANCHOR_CYCLE].index(self.para.ANCHOR_CHANNEL)
        
        if not czi.is_mosaic():
            result_df = pd.DataFrame({"row": [1],
                                      "col": [1],
                                      "y_pos": [0],
                                      "x_pos": [0]})
        else:

            startx = []
            starty = []
            for m in range(self.para.TILES_NUMBER):

                tile = czi.get_mosaic_tile_bounding_box(C = ch, M = m)
                startx.append(tile.x)
                starty.append(tile.y)

            images = np.load(os.path.join(self.para.OUTPUT_PATH, f"Registration/unstitched/{self.para.ANCHOR_CYCLE}/{self.para.ANCHOR_CHANNEL}.npy"))

            rows = np.array(starty) // 1843 + 1
            cols = np.array(startx) // 1843 + 1

            position_initial_guess = np.array([starty, startx]).T

            result_df = "MIST CHECK"

            ncc_ts = [0.05,0.01]
            for ncc_t in ncc_ts:

                log.info(f"Trying MIST with ncc_threshod = {ncc_t}")

                try:
                    result_df, _ = m2stitch.stitch_images(images, rows, cols, row_col_transpose=False, ncc_threshold = ncc_t, 
                                                  position_initial_guess = position_initial_guess, overlap_diff_threshold=5, pou=2)
                    break
                except:
                    continue

            if isinstance(result_df, str):
                
                log.warning("MIST failed, continue with initial guess...")

                result_df = pd.DataFrame({"row": rows,
                                          "col": cols,
                                          "y_pos": starty,
                                          "x_pos": startx})

        result_df["y_pos2"] = result_df["y_pos"] - result_df["y_pos"].min()
        result_df["x_pos2"] = result_df["x_pos"] - result_df["x_pos"].min()

        return result_df
    
    
    def _EulerTransform(self, fix: sitk.SimpleITK.Image, move: sitk.SimpleITK.Image) -> dict:
    
        elastixImageFilter = sitk.ElastixImageFilter()

        elastixImageFilter.SetNumberOfThreads(self.para.THREADS)
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.SetLogToFile(False) 

        elastixImageFilter.SetFixedImage(fix)
        elastixImageFilter.SetMovingImage(move)

        param_map = sitk.GetDefaultParameterMap("rigid", numberOfResolutions=3)
        param_map['ImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2', '1', '1']
        param_map['MaximumNumberOfIterations'] = ['100', '200', '400']
        param_map['NumberOfSpatialSamples'] = ['2048']
        param_map['SP_a'] = ['350.0']
        param_map['SP_alpha'] = ['0.602']
        param_map['Metric'] = ['AdvancedMattesMutualInformation']
        param_map['NumberOfHistogramBins'] = ['16'] 
        param_map['Interpolator'] = ['LinearInterpolator']
        param_map['BSplineInterpolationOrder'] = ['3']

        elastixImageFilter.SetParameterMap(param_map)
        elastixImageFilter.Execute()

        transform_parameter_map = elastixImageFilter.GetTransformParameterMap()

        theta, dy, dx = transform_parameter_map[0]['TransformParameters']

        return {'theta': float(theta), 'dy': float(dy), 'dx': float(dx)}
    
    
    
    def _applyEulerTransform2048(self, theta: float, dy: float, dx: float, move: sitk.SimpleITK.Image) -> sitk.SimpleITK.Image:

        transform = sitk.Euler2DTransform()
        transform.SetCenter([1024,1024])
        transform.SetTranslation([dy, dx])
        transform.SetAngle(theta)

        return sitk.Resample(move, transform, sitk.sitkLinear, 65535)
    
    
    
    def _applyRegisterMatrix(self, dst:np.ndarray, shift):
    
        t_m = np.array([[0,1,shift[0]],[1,0,shift[1]]])
        aligned = cv2.warpAffine(dst, t_m, dst.shape, borderValue=0).T

        return aligned

    def _applyRegisterMatrix_skimage(self, dst: np.ndarray, shift):
        t_m_3x3 = np.array([
            [0, 1, shift[0]],
            [1, 0, shift[1]],
            [0, 0, 1]
        ])

        tform = transform.AffineTransform(matrix=t_m_3x3)
        aligned = transform.warp(dst, tform.inverse, 
                               output_shape=dst.shape[::-1],
                               cval=0, 
                               preserve_range=True)

        return aligned.T.astype(np.uint16)
    

    def _register_handle(self, name_list:[str], force = False) -> dict:
        
        """
        name_list: a ['R1', R2', 'R3']-like list

        """

        # transform_parameter_maps = defaultdict(lambda: defaultdict(dict))

        for name in tqdm(name_list, total = len(name_list), dynamic_ncols = True):

            PROGRESS_DICT = self.para._parse_progress_yaml()

            if not PROGRESS_DICT['registration'][name] or force:

                move_images = np.load(os.path.join(self.para.OUTPUT_PATH, f"Registration/unstitched/{name}/{self.para.ANCHOR_CHANNEL}.npy"))
                
                transform_parameter_maps = {}
                
                for m in range(self.para.TILES_NUMBER):
                    
                    if name == self.para.ANCHOR_CYCLE:
                        transform_parameter_maps[m] = {'theta': 0.0, 'dy': 0.0, 'dx': 0.0}
                        moved_anchor = self.ANCHOR_IMAGES[m,:,:]
                    else:   
                        transform_parameter_maps[m] = self._EulerTransform(sitk.GetImageFromArray(self.ANCHOR_IMAGES[m,:,:]), sitk.GetImageFromArray(move_images[m,:,:]))
                        moved_anchor = self._applyEulerTransform2048(**transform_parameter_maps[m], move = sitk.GetImageFromArray(move_images[m,:,:]))
                        moved_anchor = sitk.GetArrayFromImage(moved_anchor)
    
                with open(os.path.join(self.para.OUTPUT_PATH, f'tmp/{name}_transform_parameter_maps.yaml'), 'w', encoding='utf-8') as file:
                    yaml.dump(dict(transform_parameter_maps), file, allow_unicode=True, default_flow_style=False)

            PROGRESS_DICT['registration'][name] = True
            self.para.save_progress_yaml(PROGRESS_DICT)

        return None
    
    
    def _get_remove_edge(self, arr):

        unique_values, counts = np.unique(arr, return_counts=True)

        sorted_unique = np.sort(unique_values)
        values_to_remove = np.concatenate([sorted_unique[:7], sorted_unique[-7:]])

        mask = np.where(~np.isin(arr, values_to_remove))

        return mask
    
    
    
    def _stitch_handle(self, rd:str, ch: str, transform_parameter_maps: dict) -> np.ndarray:

        stitched_image = np.zeros(self.STITCHED_IMAGE_SIZE, dtype = np.uint16)
        
        move_images = np.load(os.path.join(self.para.OUTPUT_PATH, f"Registration/unstitched/{rd}/{ch}.npy"))

        for m in range(self.para.TILES_NUMBER):
            
            transformed_image = self._applyEulerTransform2048(**transform_parameter_maps[m], move = sitk.GetImageFromArray(move_images[m,:,:]))
            transformed_image = sitk.GetArrayFromImage(transformed_image)
            
            mrows,mcols = np.where((transformed_image < 65535) & 
                         (np.arange(transformed_image.shape[0])[:, None] + self.GLOBAL_GRID.loc[m,"y_pos2"] >= 0) &
                         (np.arange(transformed_image.shape[0])[:, None] + self.GLOBAL_GRID.loc[m,"y_pos2"] < self.STITCHED_IMAGE_SIZE[0]) &
                         (np.arange(transformed_image.shape[1])[None, :] + self.GLOBAL_GRID.loc[m,"x_pos2"] >= 0) &
                         (np.arange(transformed_image.shape[1])[None, :] + self.GLOBAL_GRID.loc[m,"x_pos2"] < self.STITCHED_IMAGE_SIZE[1]))
                    
            mask1 = self._get_remove_edge(mrows) 
            mask2 = self._get_remove_edge(mcols)
            mask = np.intersect1d(mask1, mask2)
            
            mrows  = mrows[mask]
            mcols  = mcols[mask]
            
            srows = mrows + self.GLOBAL_GRID.loc[m,"y_pos2"]
            scols = mcols + self.GLOBAL_GRID.loc[m,"x_pos2"]

            stitched_image[srows, scols] = transformed_image[mrows, mcols].copy()

        # apply hardware channel drift
        if max(stitched_image.shape) > 32000:
            stitched_image = self._applyRegisterMatrix_skimage(stitched_image, self.para.translation_matrix[self.para.OBJECTIVE][ch])
        else:
            stitched_image = self._applyRegisterMatrix(stitched_image, self.para.translation_matrix[self.para.OBJECTIVE][ch])

        
        return stitched_image
    
    
    def _apply_clahe(self, image:np.ndarray, clipLimit = 10.0) -> np.ndarray:
    
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
        image = clahe.apply(image)
        # image = cv2.GaussianBlur(image, (3,3), 0)

        return image
        
        
    def stitching_and_registration(self, force = False):

        # for same cycle, transform parameter for each channel is shared, so it can be processed seamlessly.
        # fisrt get global grid using m2stitch
        # than stitch the anchor round first
        
        # Special handle for no stitching-need situation
        
        PROGRESS_DICT = self.para._parse_progress_yaml()
        
        MIST_path = os.path.join(self.para.OUTPUT_PATH, "tmp/MIST_grid.csv")
        
        if not PROGRESS_DICT['stitching']["MIST"] or force:
            
            
            self.GLOBAL_GRID = self._runMIST()
            self.GLOBAL_GRID.to_csv(MIST_path)
            
            PROGRESS_DICT['stitching']["MIST"] = True
            self.para.save_progress_yaml(PROGRESS_DICT)
            
        else:
            
            self.GLOBAL_GRID = pd.read_csv(MIST_path, index_col = 0)
            


        # Final stitched image size
        self.STITCHED_IMAGE_SIZE = (
            self.GLOBAL_GRID["y_pos2"].max() + 2048,
            self.GLOBAL_GRID["x_pos2"].max() + 2048,
        )

        self.ANCHOR_IMAGES = np.load(os.path.join(self.para.OUTPUT_PATH, f"Registration/unstitched/{self.para.ANCHOR_CYCLE}/{self.para.ANCHOR_CHANNEL}.npy"))

        # Get all tramsform matrix first
        # Split Registration and Stitching
        
        log.info('Registering tiles...')
        
        name_list = self.para.CYCLES.copy()
        
        self._register_handle(name_list, force)
        
        
        log.info('Stitching tiles...')
        for rd in tqdm(name_list, total = len(name_list), dynamic_ncols = True):

            with open(os.path.join(self.para.OUTPUT_PATH, f'tmp/{rd}_transform_parameter_maps.yaml'), 'r', encoding='utf-8') as file:
                transform_parameter_maps = yaml.safe_load(file)
            
            for ch in self.para.CHANNEL_INFO[rd]:
                
                PROGRESS_DICT = self.para._parse_progress_yaml()
                
                if not PROGRESS_DICT['stitching'][rd][ch] or force:
                
                    stitched_image = self._stitch_handle(rd, ch, transform_parameter_maps)
                    # stitched_image = self._apply_clahe(stitched_image, self.para.CLAHE_clip[ch])
                    tifffile.imwrite(os.path.join(self.para.OUTPUT_PATH, f"Registration/stitched/{rd}/{rd}_{ch}.tif"), stitched_image)

                    if ch == self.para.ANCHOR_CHANNEL and rd == self.para.ANCHOR_CYCLE:
                        tifffile.imwrite(os.path.join(self.para.OUTPUT_PATH, f"Registration/morphology_{ch}.tif"), stitched_image)
                        
                    PROGRESS_DICT['stitching'][rd][ch] = True
                    self.para.save_progress_yaml(PROGRESS_DICT)
                    

        if self.para.EXTRA:
            
            log.info('Registering extra tiles...')
            name_list = self.para.EXTRA_NAMES.copy()
            self._register_handle(name_list, force)
            

            morphology = []
            morphology_meta = []


            if 'rRNA' in name_list:

                morphology_rRNA = np.zeros((self.STITCHED_IMAGE_SIZE[0], self.STITCHED_IMAGE_SIZE[1], 3), dtype=np.uint16)


            log.info('Stitching extra tiles...')    
            for name in tqdm(name_list, total = len(name_list), dynamic_ncols = True):

                with open(os.path.join(self.para.OUTPUT_PATH, f'tmp/{name}_transform_parameter_maps.yaml'), 'r', encoding='utf-8') as file:
                    extra_transform_parameter_maps = yaml.safe_load(file)
                
                for ch in self.para.EXTRA_CHANNEL_INFO[name]:

                    if ch != 'SKIP':

                        PROGRESS_DICT = self.para._parse_progress_yaml()

                        if not PROGRESS_DICT['stitching'][name][ch] or force:
                        
                            stitched_image = self._stitch_handle(name, ch, extra_transform_parameter_maps)
                            tifffile.imwrite(os.path.join(self.para.OUTPUT_PATH, f"Registration/stitched/{name}/{name}_{ch}.tif"), stitched_image)

                            PROGRESS_DICT['stitching'][name][ch] = True               
                            self.para.save_progress_yaml(PROGRESS_DICT)

                        else:
                            stitched_image = tifffile.imread(os.path.join(self.para.OUTPUT_PATH, f"Registration/stitched/{name}/{name}_{ch}.tif"))
                            
                    
                        if name == 'rRNA':
                            if ch == self.para.ANCHOR_CHANNEL:
                                morphology_rRNA[:,:,0] = stitched_image
                            else:
                                morphology_rRNA[:,:,2] = stitched_image

                        if ch == self.para.ANCHOR_CHANNEL and len(morphology) == 0:
                            morphology.append(stitched_image)
                            morphology_meta.append(ch)
                        elif ch != self.para.ANCHOR_CHANNEL:
                            morphology.append(stitched_image)
                            morphology_meta.append(f"{name}_{ch}")
                        else:
                            continue
                            
            if 'rRNA' in name_list:
                tifffile.imwrite(os.path.join(self.para.OUTPUT_PATH, f"Registration/morphology_rRNA.tif"), morphology_rRNA)
                
            tifffile.imwrite(os.path.join(self.para.OUTPUT_PATH, f"Registration/morphology.tif"), np.stack(morphology, dtype = np.uint16))

            with open(os.path.join(self.para.OUTPUT_PATH, f"Registration/morphology_meta.csv"), 'w') as handle:
                handle.write(f"channel,name\n")
                for i,value in enumerate(morphology_meta):
                      handle.write(f"{i},{value}\n")
                        
        
            

    
    