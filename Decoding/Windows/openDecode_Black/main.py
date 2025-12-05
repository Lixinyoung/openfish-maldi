"""
Author: Li Xinyang
Last modified: 2025.9.28


Change log:

    2025.9.28:
        New script
        
    2025.9.29
        Keep writing
"""

import logging

from .preprocess import preprocess
from .registration import Stitching_Registration
from .spotdetection import runSpotiflow
from .segmentation import runStarDist, runCellpose
from .decoding import Decoding
from .matrixization import Matrixization

log = logging.getLogger(__name__)


from .utils import is_all_true


class OpenDecoder():
    
    def __init__(self, para):
        
        self.para = para

        self.forced_procedura = para.force_procedure
        
        self.skipped_procedure = [x for x in para.skip_procedure if x not in self.forced_procedura]
        
        
    def runPreprocess(self):
        
        if 0 in self.skipped_procedure:
            
            log.info("Skip preporcess procedure")
            
        else:
            
            force = False
            
            if 0 in self.forced_procedura:
                
                log.info("Force run preporcess procedure")
                force = True
            
            
            preprocess(self.para, force = force)
            
    
    def runRegistration(self):
        
        if 1 in self.skipped_procedure:
            
            log.info("Skip registration&stitching procedure")
            
        else:
            
            force = False
            
            if 0 in self.skipped_procedure:
                
                PROGRESS_DICT = self.para._parse_progress_yaml()
                if not is_all_true(PROGRESS_DICT['preprocess']):
                    log.warning("Run registration&stitching without recorded preprocess...")
                    
            if 1 in self.forced_procedura:
                
                log.info("Force run registration&stitching procedure")
                force = True
                
            
            SR = Stitching_Registration(self.para)
            SR.stitching_and_registration(force = force)  
        

    def runSpotiflow(self, **kwargs):
        
        if 2 in self.skipped_procedure:
            
            log.info("Skip spot detection procedure")
            
        else:
            
            force = False
            
            if 1 in self.skipped_procedure:
                
                PROGRESS_DICT = self.para._parse_progress_yaml()
                if not is_all_true(PROGRESS_DICT['registration']):
                    log.warning("Run spot detection without recorded registration&stitching...")
                    
            if 2 in self.forced_procedura:
                
                log.info("Force run spot detection procedure")
                force = True
        
            runSpotiflow(self.para, force = force, **kwargs)
            
                
    def runSegmentation(self, 
                        stardist_kwargs = {
                            'prob_thresh': 0.5,
                            'nms_thresh': 0.4,
                            'trained_model': '2D_versatile_fluo',
                            'sigma': 2.5
                        },
                        cellpose_kwargs = {
                            'restore_type': 'deblur_cyto3',
                            'diameter': 45.,
                            'flow_threshold': 1.,
                            'cellprob_threshold': -6.,
                        }): 
        
        if 3 in self.skipped_procedure:
            
            log.info("Skip segmentation procedure")
            
        else:
            
            PROGRESS_DICT = self.para._parse_progress_yaml()
            
            force = False
            
            if 3 in self.forced_procedura:
                
                log.info("Force run segmentation procedure")
                force = True
                
                
            if self.para.EXTRA and 'rRNA' in self.para.EXTRA_NAMES:
                
                if not is_all_true(PROGRESS_DICT['stitching']['rRNA']):
                
                    log.warning("Run rRNA segmentation without recorded registration&stitching...")
                    
                runCellpose(self.para, force = force, **cellpose_kwargs)
                    
            if not PROGRESS_DICT['stitching'][self.para.ANCHOR_CYCLE][self.para.ANCHOR_CHANNEL]:
                
                log.warning(f"Run {self.para.ANCHOR_CHANNEL} segmentation without recorded registration&stitching...")
                
            runStarDist(self.para, force = force, **stardist_kwargs)
        
    
    def runDecoding(self, **kwargs):
        
        if 4 in self.skipped_procedure:
            
            log.info("Skip decoding procedure")
        
        else:
            
            force = False
            
            if 2 in self.skipped_procedure:
                
                PROGRESS_DICT = self.para._parse_progress_yaml()
                if not is_all_true(PROGRESS_DICT['spot_detection']):
                    log.warning("Run decoding without recorded spot detection...")
                    
            if 4 in self.forced_procedura:
                
                log.info("Force run decoding procedure")
                force = True
        
            Decoder = Decoding(self.para, **kwargs)
            Decoder.decoding(force = force)
            
            
    def runMatrixization(self, **kwargs):
        
        if 5 in self.skipped_procedure:
            
            log.info("Skip matrixization procedure")
            
            if 4 in self.forced_procedura:
                
                log.warning("Skip matrixization after newly generated decoding results...")
                
        else:
            
            force = False
            
            if 4 in self.skipped_procedure:
                
                PROGRESS_DICT = self.para._parse_progress_yaml()
                if not is_all_true(PROGRESS_DICT['decoding']):
                    log.warning("Run matrixization without recorded decoding...")
            
            if 5 in self.forced_procedura:
                
                force = True
                
                log.info("Force run matrixization procedure")


            Mtx = Matrixization(self.para, **kwargs)
            Mtx.matrixization(force = force)  
    
        
