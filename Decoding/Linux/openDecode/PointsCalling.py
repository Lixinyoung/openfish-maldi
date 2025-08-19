import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed
from scipy.spatial import KDTree
from scipy.stats import norm
from scipy import ndimage
from tqdm import tqdm
import pandas as pd
import numpy as np
import tifffile
import cv2
import os

def _draw_spots(spot: np.ndarray, image_size:tuple):
   
    struct1 = ndimage.generate_binary_structure(2, 1)
    mask = np.zeros(image_size, np.uint8)

    mask[np.round(spot[:,1]).astype("int"), np.round(spot[:,0]).astype("int")]=1
    mask = np.uint8(mask)
    mask = ndimage.binary_dilation(mask, structure=struct1,iterations=2).astype(mask.dtype)
    mask[mask==1]=255
    
    return mask

def _read_RSFISH(rd_ch, Para, image_size):
    
    """
    spots: a numpy array of dim N x C x R (number of spots x coding channels x rounds);
    """
    
    rd = rd_ch[0]
    ch = rd_ch[1]

    dst_spot = pd.read_csv(os.path.join(Para.output, f"Registration/stitched/processed/{rd}/{rd}_{ch}.csv"), usecols=["x", "y", "intensity"])
    # Filter out outlier spots
    dst_spot = dst_spot[(dst_spot["x"] <= image_size[1] - 1) & (dst_spot["y"] <= image_size[0] - 1)].copy()
    dst_spot = dst_spot[(dst_spot["x"] > 0) & (dst_spot["y"] > 0)].copy()

    dst_trees = KDTree(dst_spot[["x", "y"]])

    # spot = N * 3 (x, y, intensity)
    dst_spots = dst_spot.to_numpy()

    mask = _draw_spots(dst_spots, image_size)
    fn =os.path.join(Para.output, f"Registration/stitched/spots/{rd}_{ch}_spots.tif")
    tifffile.imwrite(fn, mask)
    
    return [dst_spots, dst_trees]
    
def filterPoints(Para):
    
    Points = {}
    
    # Read into RS-FISH results
    
    tmp_image = tifffile.imread(os.path.join(Para.output, f"Registration/stitched/processed/{Para.Anchor_Round}/{Para.Anchor_Round}_{Para.Round_channel[Para.Anchor_Round][1]}.tif"))
    image_size = tmp_image.shape
    del tmp_image
    
    print("image_size:", image_size)
    
    rds_chs = []
    for rd, channels in Para.Round_channel.items():
        for ch in channels:
            if ch in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']:
                rds_chs.append([rd,ch])
            
    info = f"[INFO] Reading RS-FISH results..."
    print(info)  
    spots_trees = Parallel(n_jobs = -1, backend='loky')(delayed(_read_RSFISH)(rd_ch, Para, image_size) for rd_ch in rds_chs)
    
    # Store parallel results in dict
    
    Points = {}
    Trees = {}
    
    i = 0
    for rd, channels in Para.Round_channel.items():
        
        dst_spots = {}
        dst_trees = {}
        
        for ch in channels:
            if ch in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']:
                # spot = N * 3 (x, y, intensity)
                dst_spots[ch] = spots_trees[i][0]
                dst_trees[ch] = spots_trees[i][1]
                i += 1
        
        Trees[rd] = dst_trees
        Points[rd] = dst_spots
        
    
    # Merge points
  
    info = f'[INFO] Merging spots...'
    print(info)
    Spots_Dict = _merge_points(Para, Points, Trees)
        
    return Spots_Dict, image_size


def _gaussian_intenisty(normal_dist, dist, intensity):
    
    return normal_dist.pdf(dist)/normal_dist.pdf(0.0) * intensity


def _merge_points(Para, Points, Trees):
    
    std_dev_dict = {
        "AF488": 0.8,
        "AF546": 0.8,
        "AF594": 0.8,
        "Cy5": 0.8,
        "Cy7": 0.8
    }
    
    mean = 0.0
    
    Spots_Dict = {}
    
    if Para.method_type == "10N":
        
        for rd,channels in Para.Round_channel.items():
            
            channels = [x for x in channels if x in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']]
            barcodes_01 = Para.CodeBooks_01[rd]
            
            # Acquire possible combination dict through barcode_01
            comb_dict = {}
            for i in range(barcodes_01.shape[2]):
                comb_dict[i] = []
                for j in np.where(barcodes_01[:,0,i] == 1)[0]:
                    try:
                        id1,id2 = np.where(barcodes_01[j,0,:] == 1)[0]
                        if id1 != i:
                            comb_dict[i].append(id1)
                        elif id2 != i:
                            comb_dict[i].append(id2)
                    except:
                        continue
                    
            # Build spots, spots_loc, spots_id considering distance
            rd_spots = []
            rd_spots_loc = []
            rd_spots_id = []
            
            for id1, idx in tqdm(comb_dict.items(), desc = f"Preparing decoding inputs for {rd}"):
                
                # spots: a numpy array of dim N x C x R (number of spots x coding channels x rounds), value is intensity;
                # spots_loc: a numpy array of dim N x 2 (number of spots x 2), value is x,y
                # spots_id: a numpy array of dim N x C x R (number of spots x coding channels x rounds), value is index;
                
                
                ch1 = channels[id1]
                normal_dist = norm(loc=mean, scale=std_dev_dict[ch1])
                dst_spot = Points[rd][ch1]
                spots_loc = dst_spot[:,0:2]
                array_shape = (dst_spot.shape[0], 1, len(channels))
                spots = np.empty(array_shape, dtype = np.float_)
                spots_id = np.empty(array_shape, dtype = np.float_)
                spots_id[:,0,id1] = np.arange(dst_spot.shape[0])
                
                ddist_list = [np.full(spots.shape[0], 1.0)]
                for id2 in idx:
                    ch2 = channels[id2]
                    ddist, ssubject_id = Trees[rd][ch2].query(spots_loc[:, :], workers = -1)
                    spots[:,0,id2] = _gaussian_intenisty(normal_dist, ddist, Points[rd][ch2][ssubject_id, 2])
                    spots_id[:,0,id2] = ssubject_id
                    ddist_list.append(ddist)
                
                ddist = np.column_stack(ddist_list)
                min_ddist = np.min(ddist, axis = 1)
                
                # 防止Query Point衰减太厉害
                min_ddist[min_ddist > std_dev_dict[ch1]] = std_dev_dict[ch1]
                
                    
                spots[:,0,id1] = _gaussian_intenisty(normal_dist, min_ddist, dst_spot[:,2])
                        
                rd_spots.append(spots)
                rd_spots_loc.append(spots_loc)
                rd_spots_id.append(spots_id)
                
            Spots_Dict[rd] = {"intensity": np.vstack(rd_spots), "location": np.vstack(rd_spots_loc), "index": np.vstack(rd_spots_id)}
            
    elif Para.method_type == "MultiCycle":
        
        # Acquire possible combination dict through barcode_01
        barcodes_01 = Para.CodeBooks_01
        comb_dict = {}
        for i in range(barcodes_01.shape[1]):
            for j in range(barcodes_01.shape[2]):
                comb_dict[(i,j)] = []
                for k in np.where(barcodes_01[:,i,j] == 1)[0]:
                    id1s,id2s = np.where(barcodes_01[k,:,:] == 1)
                    for c,r in zip(id1s,id2s):
                        if (c,r) != (i,j):
                            comb_dict[(i,j)].append((c,r))
                    
        # Build spots, spots_loc, spots_id considering distance
        
        CHANNELS = ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']
        ROUNDS = Para.Round_list
        mc_spots = []
        mc_spots_loc = []
        mc_spots_id = []
        
        for id1, idx2 in tqdm(comb_dict.items(), desc = "Preparing decoding inputs"):
            
            ch1 = CHANNELS[id1[0]]
            rd1 = ROUNDS[id1[1]]
            normal_dist = norm(loc=mean, scale=std_dev_dict[ch1])
            
            dst_spot = Points[rd1][ch1]
            spots_loc = dst_spot[:,0:2]
            
            array_shape = (dst_spot.shape[0], 5, len(ROUNDS))
            spots = np.empty(array_shape, dtype = np.float_)
            spots_id = np.empty(array_shape, dtype = np.float_)
            spots_id[:,id1[0],id1[1]] = np.arange(dst_spot.shape[0])
                
            ddist_list = [np.full(spots.shape[0], 2.0)]    
            for id2 in idx2:
                ch2 = CHANNELS[id2[0]]
                rd2 = ROUNDS[id2[1]]
                ddist, ssubject_id = Trees[rd2][ch2].query(spots_loc[:,:])
                spots[:,id2[0],id2[1]] = _gaussian_intenisty(normal_dist, ddist, Points[rd2][ch2][ssubject_id, 2])
                spots_id[:,id2[0],id2[1]] = ssubject_id
                ddist_list.append(ddist)
                
            ddist = np.column_stack(ddist_list)
            min_ddist = np.min(ddist, axis = 1)
            
            min_ddist[min_ddist > std_dev_dict[ch1]] = std_dev_dict[ch1]
                    
            spots[:,id1[0],id1[1]] = _gaussian_intenisty(normal_dist, min_ddist, dst_spot[:,2])

            mc_spots.append(spots)
            mc_spots_loc.append(spots_loc)
            mc_spots_id.append(spots_id)
            
        Spots_Dict = {"intensity": np.vstack(mc_spots), "location": np.vstack(mc_spots_loc), "index": np.vstack(mc_spots_id)}
        
    np.save(os.path.join(Para.output, "tmp/Spots_Dict.npy"), Spots_Dict)
        
    return Spots_Dict