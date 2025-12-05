"""
Author: Li Xinyang
Last modified: 2025.9.25


Change log:

    2025.9.25:
        New script
"""

import logging

from collections import defaultdict
import xml.etree.ElementTree as ET
from aicspylibczi import CziFile
import pandas as pd
import numpy as np
import itertools
import yaml
import re
import os

log = logging.getLogger(__name__)

from .utils import is_all_true


class _Constants:
    # hardware shift of imaging system, in pixel
    translation_matrix = {"20X":{
            "DAPI": [0.0, 0.0], 
            "AF405": [0.0, 0.0], 
            "AF488": [-0.38+5.38, 0.25], 
            "AF546": [-3.61+5.38,0.12], 
            "AF594": [5.38,0.0], 
            "Cy5": [-2.54+5.38, -0.10],
            "Cy7": [-2.15+5.38,-0.39],
            "DIC": [5.38, 0.0]},
                             "40X":{
            "DAPI": [0.0 ,0.0],
            "AF405": [0.0, 0.0], 
            "AF488": [-0.44, 0.85], 
            "AF546": [-4.48, -0.03],
            "AF594": [0.0, 0.0], 
            "Cy5": [-2.68, 0.28],
            "Cy7": [-1.94, -0.03]}}
    
    # CLAHE clip for each channel
    CLAHE_clip = {
            "AF405": 10.0,
            "AF488":10.0,
            "AF546":5.0,
            "AF594":7.0,
            "Cy5":7.0,
            "Cy7":5.0 # 15.0
    }
    
    pixel_scaler = {
        '20X': 0.325,
        '40X': 0.1625
    }
    



class Para(_Constants):
    
    def __init__(self, 
                 filename: dict,
                 codebook_path: str, 
                 output_path: str = 'Result',
                 anchor_channel: str = None,
                 extra: dict = None,
                 run_deblur: bool = True,
                 run_BaSiC: bool = True,
                 objective: str = '20X',
                 threads = 48,
                 run_basic_clustering = False,
                 # restart from save point can be override with this parameter
                 skip_procedure: [int] = None, # 0: preprocess, 1: stitching&regietration, 2: point detection, 3: segmentation, 4: decoding, 5: matrixization
                 force_procedure: [int] = None,
                 **kwargs
                 ):
        """
        filename = {
                "R7": "/media/duan/DuanLab_Data/openFISH/Decode/TestData/20250118_P16_13M_58_V5_8_ABA109_GFP_R7_depth.czi",
                'R11': "/media/duan/DuanLab_Data/openFISH/Decode/TestData/20250118_P16_13M_58_V5_8_ABA109_GFP_R11_depth.czi",
                ...
            }


        extra = {
            'rRNA': {
                'filepath': '/media/duan/DuanLab_Data/openFISH/Decode/TestData/20250118_P16_13M_58_V5_8_ABA109_GFP_rRNA_depth_new.czi',
                'channel': ['AF405', 'AF546']
            },
            "P16": {
                'filepath': '/media/duan/DuanLab_Data/openFISH/Decode/TestData/20250118_P16_13M_58_V5_8_ABA109_GFP_rRNA_depth.czi',
                'channel': ['AF405', 'AF488']
            },
            ...
        }
        
        """
        
        self.OUTPUT_PATH = output_path
        self.THREADS = min(threads, os.cpu_count()-1)
        self.run_deblur = run_deblur
        self.run_BaSiC = run_BaSiC
        self.run_basic_clustering = run_basic_clustering
        self.skip_procedure = skip_procedure
        self.force_procedure = force_procedure

        self.CYCLES = list(filename.keys())
        self.ANCHOR_CYCLE = self.CYCLES[0]
        
        if not anchor_channel:
            self.ANCHOR_CHANNEL = 'AF405'
        else: 
            self.ANCHOR_CHANNEL = anchor_channel
        
        self.CZI_FILES = {rd: CziFile(fp) for rd,fp in filename.items()}
        
        # multi tile
        if self.CZI_FILES[self.ANCHOR_CYCLE].is_mosaic():
            self.TILES_NUMBER = self.CZI_FILES[self.ANCHOR_CYCLE].get_dims_shape()[0]['M'][1]
        # single tile
        else:
            self.run_BaSiC = False
            self.TILES_NUMBER = 1
        
        self.OBJECTIVE = objective

        self.CHANNEL_INFO = {}
        self.CODEBOOK = {}
        
        _codebook = pd.read_csv(codebook_path)
        
        for rd in self.CYCLES:
            _czi = self.CZI_FILES[rd]
            _czi_dim_shape = _czi.get_dims_shape()[0]
            _czi_meta  = ET.tostring(_czi.meta, encoding='unicode')
            self.CHANNEL_INFO[rd] = [self._get_channel_info(x, _czi_meta) for x in range(_czi_dim_shape['C'][1])]
            self.CODEBOOK[rd] = self._generate_codebook(_codebook[_codebook['Round'] == rd], self.CHANNEL_INFO[rd])
            
        if extra:
            self.EXTRA = True
            self.EXTRA_NAMES = list(extra.keys())
            self.EXTRA_CHANNEL_INFO = {}
            self.EXTRA_CZI_FILES = {name: CziFile(fpch['filepath']) for name,fpch in extra.items()}
            for name, fpch in extra.items():
                _czi = self.EXTRA_CZI_FILES[name]
                _czi_dim_shape = _czi.get_dims_shape()[0]
                _czi_meta  = ET.tostring(_czi.meta, encoding='unicode')
                self.EXTRA_CHANNEL_INFO[name] = [self._get_channel_info(x, _czi_meta) for x in range(_czi_dim_shape['C'][1])]
                # self.CHANNEL_INFO[name].remove(self.ANCHOR_CHANNEL) # anchor channel name for each cycle must be the same
                for i, ch in enumerate(self.EXTRA_CHANNEL_INFO[name]):
                    if ch not in fpch['channel']:
                        self.EXTRA_CHANNEL_INFO[name][i] = 'SKIP'
        else:
            self.EXTRA = False
        
        # Overrides constants
        self._overrides = {}
        
        for key, value in kwargs.items():
            if hasattr(_Constants, key):
                self._overrides[key] = value
            else:
                setattr(self, key, value)
                
                
        self._sanity_check()
                
        self._create_folders()
        
        self._sanity_yaml()
        
        
    def __getattr__(self, name: str):
        
        if name in self._overrides:
            return self._overrides[name]
        
        elif hasattr(_Constants, name):
            return getattr(_Constants, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
    
    def _get_channel_info(self, channel_id:int, czi_meta: str) -> str:
        
        pattern = rf'Id="Channel:{channel_id}"[^>]*Name="([^"]*)"'
        match = re.search(pattern, czi_meta)
        if match:
            return match.group(1)
        
        raise ValueError(f"{pattern} is not avaiable for channel name detection, check the image meta data.")
         
        
    def _generate_codebook(self, codebook: pd.core.frame.DataFrame, channel: [str]) -> dict:
        """
        codebook: codebook consists of only one cycle
        channel: used channel detected in czi meta file
        """
        codedict = defaultdict(dict)
        channel = channel.copy()
        channel.remove(self.ANCHOR_CHANNEL)

        if np.all(codebook['RO1'] == codebook['RO2']):
            # single-color code only
            # real channel number should always less than codebook metioned ones
            for _, row in codebook.iterrows():
                    ch = row['RO1']
                    codedict[ch][ch] = f"{row['gene']}"

        elif np.any(codebook['RO1'] == codebook['RO2']) | np.all(codebook['RO1'] != codebook['RO2']):
            # dual-color code and single-color code mixed OR dual-color code only
            # order of channel in RO1 and RO2 column doesn't matter
            combinations = list(itertools.combinations(channel, 2))

            _tmp_dict = {}
            for _, row in codebook.iterrows():
                if row['RO1'] == row['RO2']:
                    codedict[row['RO1']][row['RO1']] = f"{row['gene']}"
                else:
                    _tmp_dict[f"{row['RO1']}_{row['RO2']}"] = row['gene']
                    _tmp_dict[f"{row['RO2']}_{row['RO1']}"] = row['gene']

            for ch1,ch2 in combinations:
                try:
                    codedict[ch1][ch2] = _tmp_dict[f"{ch1}_{ch2}"]
                except:
                    
                    rd = codebook['Round'].values[0]
                    
                    codedict[ch1][ch2] = f"{rd}_{ch1}_{ch2}"

        else:
            raise ValueError("Unacceptable codebook, only dual-color or single-color or mixed allowed. Fill RO1 and RO2 column, do not leave it blank.")

        return codedict
    
    
    def _sanity_tile_check(self, czi_files):
        
        for name, _czi in czi_files.items():
            if self.TILES_NUMBER == 1:
                if _czi.is_mosaic():
                    raise AssertionError(f'{name} has no same number of tiles to {self.ANCHOR_CYCLE}')
            else:
                if _czi.get_dims_shape()[0]['M'][1] != self.TILES_NUMBER:
                    raise AssertionError(f'{name} has no same number of tiles to {self.ANCHOR_CYCLE}')
    
    
    def _sanity_check(self) -> None:
    
        if self.OBJECTIVE not in self.translation_matrix.keys():
            raise KeyError(f'{self.OBJECTIVE} not in pre-defined in translation matrix.\n Fix the objective or pass new translation_matrix')
            
        if self.OBJECTIVE not in self.pixel_scaler.keys():
            raise KeyError(f'{self.OBJECTIVE} not in pre-defined in pixel scaler.\n Fix the objective or pass new pixel_scaler')

        for rd, chs in self.CHANNEL_INFO.items():
            # ensure every image has anchor channel and names are the same
            if self.ANCHOR_CHANNEL not in chs:
                raise KeyError(f'{self.ANCHOR_CHANNEL} not in {rd}, check the input')
            # ensure translation matrix is defined for proper channel names
            for ch in chs:
                if ch not in self.translation_matrix[self.OBJECTIVE].keys():
                    raise KeyError(f'{rd} {ch} not pre-defined in translation matrix.\n Pass new translation_matrix')
                    
                if ch not in self.CLAHE_clip.keys(): 
                    raise KeyError(f'{rd} {ch} not pre-defined in CLAHE matrix.\n Pass new CLAHE_clip')
                    
        # ensure every image has same number of tiles
        self._sanity_tile_check(self.CZI_FILES)
    
        # ensure every image has same number of tiles
        if self.EXTRA:
            self._sanity_tile_check(self.EXTRA_CZI_FILES)
            
            for name, chs in self.EXTRA_CHANNEL_INFO.items():
                # ensure every image has anchor channel and names are the same
                if self.ANCHOR_CHANNEL not in chs:
                    raise KeyError(f'{self.ANCHOR_CHANNEL} not in {name}, check the input')
                # ensure translation matrix is defined for proper channel names
                for ch in chs:
                    if ch not in self.translation_matrix[self.OBJECTIVE].keys() and ch != 'SKIP':
                        raise KeyError(f'{rd} {ch} not pre-defined in translation matrix.\n Pass new translation_matrix')
                        
        for step in self.skip_procedure:
            if step not in [0,1,2,3,4,5]:
                raise ValueError(f'{step} not pre-defined in skip procedure.\n Pass correct skip_procedure:\n 0: preprocess, 1: stitching&regietration, 2: point detection, 3: segmentation, 4: decoding, 5: matrixization')
                
        for step in self.force_procedure:
            if step not in [0,1,2,3,4,5]:
                raise ValueError(f'{step} not pre-defined in force procedure.\n Pass correct force_procedure:\n 0: preprocess, 1: stitching&regietration, 2: point detection, 3: segmentation, 4: decoding, 5: matrixization')
            
                
                
    def _folder_creator(self, dic: str) -> None:
        
        if not os.path.exists(dic):
            os.makedirs(dic)
        else:
            pass
            
            
    def _create_folders(self) -> None:
        
        dic_list = [
            # For temporary files
            os.path.join(self.OUTPUT_PATH, "tmp"),
            # For preprocessed images
            os.path.join(self.OUTPUT_PATH, "Segmentation"),
            os.path.join(self.OUTPUT_PATH, "Registration"),
        ]

        for rd in self.CYCLES:
            dic_list.append(os.path.join(self.OUTPUT_PATH, f"Registration/stitched/{rd}"))
            dic_list.append(os.path.join(self.OUTPUT_PATH, f"Registration/unstitched/{rd}"))
            
        if self.EXTRA:
            for name in self.EXTRA_NAMES:
                dic_list.append(os.path.join(self.OUTPUT_PATH, f"Registration/stitched/{name}"))
                dic_list.append(os.path.join(self.OUTPUT_PATH, f"Registration/unstitched/{name}"))
            
        for dic in dic_list:
            
            self._folder_creator(dic)
            
    
    def save_progress_yaml(self, progress_dict: dict):
        
        with open(os.path.join(self.OUTPUT_PATH, 'tmp/progress.yaml'), 'w', encoding='utf-8') as file:
             yaml.dump(progress_dict, file, allow_unicode=True, default_flow_style=False)
                
                
    def _merge_nested_dicts_new(self,dict1, dict2):
        
        """
        add dict2 missed keys to dict2 compared to dict1
        """
    
        result = dict2.copy() 

        for key, value in dict1.items():
            if key not in result:
                result[key] = value
            else:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._merge_nested_dicts_new(value, result[key])

        return result
    
    
    def _sanity_yaml(self):
        
        progress_dict = {}
            
        preprocess_dict = {}
        registration_dict = {}
        stitching_dict = {}


        for rd,chs in self.CHANNEL_INFO.items():
            preprocess_dict[rd] = {ch:False for ch in chs}
            stitching_dict[rd] = {ch:False for ch in chs}
            registration_dict[rd] = False

        if self.EXTRA:
            # stitching_dict['Extra'] = False
            for name,chs in self.EXTRA_CHANNEL_INFO.items():
                preprocess_dict[name] = {ch:False for ch in chs if ch != 'SKIP'}
                stitching_dict[name] = {ch:False for ch in chs}
                registration_dict[name] = False
                # registration_stitching_dict[name] = {ch:False for ch in chs}

        progress_dict['preprocess'] = preprocess_dict

        stitching_dict['MIST'] = False

        progress_dict['registration'] = registration_dict
        progress_dict['stitching'] = stitching_dict

        spot_detection_dict = {}
        decoding_dict = {}

        for rd,chs in self.CHANNEL_INFO.items():
            pch = chs.copy()
            pch.remove(self.ANCHOR_CHANNEL)
            spot_detection_dict[rd] = {ch:False for ch in pch}
            decoding_dict[rd] = False

        segmentation_dict = {}

        segmentation_dict[self.ANCHOR_CHANNEL] = False

        if self.EXTRA and 'rRNA' in self.EXTRA_NAMES:
            segmentation_dict['rRNA'] = {'segmentation': False, 'vectorize': False}

        progress_dict['spot_detection'] = spot_detection_dict
        progress_dict['segmentation'] = segmentation_dict
        progress_dict['decoding'] = decoding_dict

        mtx_dict = {}

        mtx_dict['resolve_conflits'] = False
        mtx_dict['assign_transcripts'] = False

        progress_dict['matrixization'] = mtx_dict
        
        
        progress_path = os.path.join(self.OUTPUT_PATH, 'tmp/progress.yaml')
        
        if os.path.exists(progress_path):
            
            with open(progress_path, 'r', encoding='utf-8') as file:
                progress_dict_old = yaml.safe_load(file)
            
            # ensure new added files get processed
            progress_dict_new = self._merge_nested_dicts_new(progress_dict, progress_dict_old)
            
        else:
            
            progress_dict_new = progress_dict.copy()
        
        # ensure newly decoded files to be assigned automatically
        if not is_all_true(progress_dict_new['decoding']):
            
            progress_dict_new['matrixization']['assign_transcripts'] = False
            
        # ensure newly segmentated files to be resolved automatically 
        if not is_all_true(progress_dict_new['segmentation']):
            
            progress_dict_new['matrixization']['resolve_conflits'] = False
            
        
        self.save_progress_yaml(progress_dict_new)
        
                
            
    def _parse_progress_yaml(self):
        
        
        progress_path = os.path.join(self.OUTPUT_PATH, 'tmp/progress.yaml')
            
        with open(progress_path, 'r', encoding='utf-8') as file:
            progress_dict = yaml.safe_load(file)

        return progress_dict
            
            
            
            