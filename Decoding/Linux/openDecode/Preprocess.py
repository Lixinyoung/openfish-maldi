from .Register import _applyRegisterMatrix
from basicpy import BaSiC
from tqdm import tqdm
import pandas as pd
import numpy as np
import tifffile
import shutil
import glob
import sys
import os

class Para:
    
    """
    
    filename:dict{str} = a dict for storing input information, where tiles and corresponding xml file stored
        {
        "R1":"D:/Data/Test_R1",
        "R2":"D:/Data/Test_R2",
        ...
        }
    CodingBook:str = Code Book for decoding
    method_type:str = 10N or MultiCycle
        
    workpath:str = workpath for output, default is ./
    output_path:str = output folder, all results will be output into workpath/output_path, default is 'Result'
    objective:str = "20X" or "40X"
    rRNA:str = None, rRNA tiles path, default is None
    extra: str = None, extra staining, which will be output to Green channel in final Xenium explorer file
    
    This class is used to store the parameters and create necessory folders
    
    
    """
    
    
    def __init__(self, filename:dict, 
                 CodingBook:str, 
                 method_type:str = "10N", 
                 workpath:str = "./", 
                 output_path:str = "Result",
                 objective:str = "20X", 
                 rRNA:str = None,
                 extra:str = None,
                 baysor:bool = True,
                 phase_cross_correction = False):
        
        # 以DAPI为参考，则不应该移动DAPI 第一位是row，第二位是column
        self.translation_matrix = {"20X":{
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
    
        self.channel_clip = {
            "DAPI":60000,
            "AF405":60000,
            "AF488":10000,
            "AF546":10000,
            "AF594":10000,
            "Cy5":12000,
            "Cy7":10000 # 6000
        }

        self.CLAHE_clip = {
            "AF488":10.0,
            "AF546":3.5,
            "AF594":7.0,
            "Cy5":7.0,
            "Cy7":3.5 # 15.0
        }
    
        
        self.filename = filename
        self.CodingBook = pd.read_csv(CodingBook)
        self.method_type = method_type
        
        self.workpath = workpath
        self.output = os.path.join(workpath, output_path)
        
        self.rRNA = rRNA
        self.rRNAseg = False
        self.extra = extra
        self.extraseg = False
        
        self.objective = objective
        self.channel_shift = self.translation_matrix[self.objective]
        self.baysor = baysor
        self.phase_cross_correction = phase_cross_correction
        
        # Get anchor round for registration and stitching
        self.Round_list = list(filename.keys())
        self.Anchor_Round = self.Round_list[0]
        
        # Get used channel for each round
        self.Round_channel = {}
        for rd, tilepath in filename.items():
            filepath = sorted(glob.glob(os.path.join(tilepath, "*ORG.tif")))
            self.Round_channel[rd] = np.unique([x.split("_")[-2] for x in filepath[0:6]])
            if rd == self.Anchor_Round:
                self.tile_numbers = np.unique([x.split("_")[-3] for x in filepath])
        
        # Create necessroy output folders
        self._create_folders()
        # Rebuild codebook for PostCode decode
        self.CodeBooks_01, self.CodeBooks_genes = self._rebuild_CodeBook()
        
        
        
    def __getitem__(self, key):
        
        return self.filename[key]
                
                
    def _folder_creator(self, dic):
        
        if not os.path.exists(dic):
            os.makedirs(dic)
        else:
            # shutil.rmtree(dic)
            # os.makedirs(dic)
            pass
            
            
    def _create_folders(self):
        
        dic_list = [
            # For temporary files
            os.path.join(self.output, "tmp"),
            # For preprocessed images and detected spots and decoded genes
            os.path.join(self.output, "Registration/stitched/spots"),
            os.path.join(self.output, "Registration/stitched/gene"),
        ]

        for rd in self.Round_channel.keys():
            dic_list.append(os.path.join(self.output, f"Registration/stitched/processed/{rd}"))
            dic_list.append(os.path.join(self.output, f"Registration/unstitched/{rd}"))
            dic_list.append(os.path.join(self.output, f"Registration/unstitched/{rd}_DAPI"))
        
        if isinstance(self.rRNA, str):
            
            self.rRNAseg = True
            self.rRNA_output = os.path.join(self.output,"Registration/unstitched/rRNA_DAPI")
            dic_list.append(self.rRNA_output)
            
            
        if isinstance(self.extra, str):
            
            self.extraseg = True
            self.extra_output = os.path.join(self.output,"Registration/unstitched/extra")
            dic_list.append(self.extra_output)
            
            filepaths = sorted(glob.glob(os.path.join(self.extra, "*ORG.tif")))
            self.extra_channels = np.unique([x.split("_")[-2] for x in filepaths[0:6]])
            
            
        for dic in dic_list:
            
            self._folder_creator(dic)    
            
    def _rebuild_CodeBook(self):
        
        """
        if method_type == 10N, return CodeBooks_01 = {
        "R1": barcode_01, "R2": barcode_01,
        }
        and CodeBooks = {
        "R1": genes, R2: genes,
        }
        
        if method_type == MultiCycle, return barcode_01, genes
        
        barcodes_01: a numpy array of dim K x C x R (number of barcodes x coding channels x rounds)
        """
        
        if self.method_type == "10N":
            CodeBooks_01 = {}
            CodeBooks = {}
            for rd,channels in self.Round_channel.items():
                tmp = self.CodingBook[self.CodingBook["Round"] == rd].copy()
                tmp = tmp.reset_index(drop = True)
                used_channels = list(np.unique(tmp[["RO1", "RO2"]]))
                array_shape = (len(tmp), 1, len(used_channels))
                barcodes_01 = np.zeros(array_shape, dtype = np.int_)
                
                for i in range(len(tmp)):
                    barcodes_01[i,0,used_channels.index(tmp.loc[i, "RO1"])] = 1
                    barcodes_01[i,0,used_channels.index(tmp.loc[i, "RO2"])] = 1
                    
                CodeBooks_01[rd] = barcodes_01
                CodeBooks[rd] = tmp["gene"].to_numpy()
        
        elif self.method_type == "MultiCycle":
            
            CHANNELS = ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']
            
            array_shape = (len(self.CodingBook), len(CHANNELS), len(self.Round_list))
            barcodes_01 = np.zeros(array_shape, dtype = np.int_)
            
            for i in range(barcodes_01.shape[0]):
                for j,r in enumerate(self.Round_list):
                    if self.CodingBook.loc[i,r] != "BLANK":
                        barcodes_01[i, CHANNELS.index(self.CodingBook.loc[i,r]), j] = 1
            
            CodeBooks_01 = barcodes_01
            CodeBooks = self.CodingBook["gene"].to_numpy()
                        
        else:
            
            print(f"{method_type} is not an option, choose 10N or MultiCycle.")
            sys.exit()
            
            
        return CodeBooks_01, CodeBooks
                    
        

def tilePreprocess(Para):
        
    # Preprocess tiles
    
    Ref_filepath = sorted(glob.glob(os.path.join(Para[Para.Anchor_Round], "*ORG.tif")))
    
    for rd, channels in Para.Round_channel.items():
        
        filepath = sorted(glob.glob(os.path.join(Para[rd], "*ORG.tif")))
        
        for m in tqdm(Para.tile_numbers, total = len(Para.tile_numbers), 
                     desc = f"Preprocessing(Intensity Clip) {rd} tiles"):
            
            tmp_tile_path = [x for x in filepath if m in x]
            
            for im in tmp_tile_path:
                channel = im.split("_")[-2]
                image_in = tifffile.imread(im)
                
                # Channel clip
                channel_clip = max(np.percentile(image_in, 99.9), Para.channel_clip[channel] * 0.9)
                channel_mask = image_in > channel_clip
                
                if np.sum(channel_mask) > 1:
                    min_val = np.min(image_in[channel_mask])
                    max_val = np.max(image_in[channel_mask])
                    if min_val != max_val:
                        # image_in[channel_mask] = (image_in[channel_mask] - min_val) / (max_val - min_val) * Para.channel_clip[channel] * 0.1 + Para.channel_clip[channel] * 0.9
                        image_in[channel_mask] = (image_in[channel_mask] - min_val) / (max_val - min_val) * channel_clip * 0.1
                    else:
                        image_in[channel_mask] = 0
                else:
                    image_in[channel_mask] = 0
            
                if channel == "DAPI" or channel == "AF405":
                    tile_out_path = os.path.join(Para.output, f"Registration/unstitched/{rd}_DAPI/{rd}_{m}_AF405.tif")
                else:
                    tile_out_path = os.path.join(Para.output, f"Registration/unstitched/{rd}/{rd}_{m}_{channel}.tif")
                    
                tifffile.imwrite(tile_out_path, image_in)
                
        # Read into clipped channel of each Round and run BaSiC (20241124)
        
        for ch in tqdm(channels, desc = f"Preprocessing(BaSiC) {rd} tiles"):
            
            if ch == "DAPI" or ch == 'AF405':
                
                continue
                
            else:
                
                tmp_tile_path = sorted(glob.glob(os.path.join(Para.output, f"Registration/unstitched/{rd}/*{ch}.tif")))
                images = [tifffile.imread(x) for x in tmp_tile_path]
                images = np.stack(images)
                basic = BaSiC(get_darkfield=True, smoothness_flatfield=1, max_workers = 48, max_reweight_iterations = 200)
                basic.fit(images)
                images_transformed = basic.transform(images)
                for m in range(images_transformed.shape[0]):
                    tile_out_path = os.path.join(Para.output, f"Registration/unstitched/{rd}/{rd}_{Para.tile_numbers[m]}_{ch}.tif")
                    tifffile.imwrite(tile_out_path, images_transformed[m].astype(np.uint16))
            
        
    # move DAPI/AF546(rRNA) channel image to seperate folder
    if Para.rRNAseg:
        
        filepaths = sorted(glob.glob(os.path.join(Para.rRNA, "*ORG.tif")))
        
        for imName in tqdm(filepaths, total = len(Para.tile_numbers), 
                     desc = f"Preprocessing rRNA tiles"):
        
            img = tifffile.imread(imName)
            old_image_name = os.path.split(imName)[-1]
            if old_image_name.endswith("DAPI_ORG.tif") or old_image_name.endswith("AF405_ORG.tif"):
                # Apply Channel Shift
                # img = _applyRegisterMatrix(img, Para.channel_shift['AF405'])
                new_image_name = "rRNA_DAPI_" + old_image_name.split("_")[-3] + "_DAPI.tif"
            else:
                # Apply Channel Shift
                # img = _applyRegisterMatrix(img, Para.channel_shift['AF546'])
                new_image_name = "rRNA_DAPI_" + old_image_name.split("_")[-3] + "_rRNA.tif"

            tifffile.imwrite(os.path.join(Para.rRNA_output, new_image_name), img)
    
    # move DAPI/extraChannel image to a seperate folder
    # DAPI is necessary, any number of extra channels is allowed
    # extra image will only be aligned and showed in Xenium explorer
    if Para.extraseg:
        
        filepaths = sorted(glob.glob(os.path.join(Para.extra, "*ORG.tif")))
        
        extra_channels = []
        images = {}
        
        for ch in Para.extra_channels:
            if ch != 'DAPI' and ch != 'AF405' and ch != 'EBFP2':
                images[ch] = []
                extra_channels.append(ch)
        
        for imName in tqdm(filepaths, total = len(Para.tile_numbers) * len(Para.extra_channels), 
                     desc = f"Preprocessing extra tiles"):
            
            img = tifffile.imread(imName)
            old_image_name = os.path.split(imName)[-1]
            
            if old_image_name.endswith("DAPI_ORG.tif") or old_image_name.endswith("AF405_ORG.tif") or old_image_name.endswith("EBFP2_ORG.tif"):
                new_image_name = "extra_" + old_image_name.split("_")[-3] + "_DAPI.tif"
                # Apply Channel Shift
                # img = _applyRegisterMatrix(img, Para.channel_shift['AF405'])
                tifffile.imwrite(os.path.join(Para.extra_output, new_image_name), img)
                
            else: 
                images[old_image_name.split("_")[-2]].append(img)
        
        for ch in tqdm(extra_channels, desc = f"Preprocessing extra tiles(BaSiC)"):
            
            tmp_images = np.stack(images[ch])
            basic = BaSiC(get_darkfield=True, smoothness_flatfield=1, max_workers = 48, max_reweight_iterations = 100)
            basic.fit(tmp_images)
            images_transformed = basic.transform(tmp_images)
            
            for i,m in enumerate(Para.tile_numbers):
                
                new_image_name = f"extra_{m}_{ch}.tif"
                # Apply Channel Shift
                # img = _applyRegisterMatrix(images_transformed[i].astype(np.uint16), Para.channel_shift[ch])
                tifffile.imwrite(os.path.join(Para.extra_output, new_image_name), images_transformed[i].astype(np.uint16))

            
            
            
        
        
        
                
            
            
            
            
    