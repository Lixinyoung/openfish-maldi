import warnings
warnings.filterwarnings('ignore')

from skimage.io import imread
from spotiflow.model import Spotiflow
from tqdm import tqdm
import pandas as pd

import subprocess
import os

def runRSFISHforTiles(Para, anisotropyCoefficient = 1.00,
                      ransac = "Multiconsensus RANSAC",
                      sigmaDoG = {
                          'R1': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R2': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R3': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R4': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R5': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R6': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R7': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R8': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R9': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R10': [1.15, 1.15, 1.15, 1.15, 1.15],
                          'R11': [1.15, 1.15, 1.15, 1.15, 1.15],
                      },
                      thresholdDoG = {
                          'R1': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R2': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R3': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R4': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R5': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R6': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R7': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R8': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R9': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R10': [0.007, 0.007, 0.007, 0.007, 0.007],
                          'R11': [0.007, 0.007, 0.007, 0.007, 0.007],
                      },
              supportRadius = 3,inlierRatio = 0.1,maxError = 0.75,
              intensityThreshold = [120,120,120,120,120],
              min_number_of_inliers = 6, initial = 8, final = 20,
              bsMethod = "No background subtraction", bsMaxError = 0.05,bsInlierRatio = 0.1):
    
    """
    Run RS-FISH on every channels, every cycles tile seperaterly
    """
    
    RS_output = os.path.abspath(Para.output)
    ijm_file = os.path.join(Para.output, "Registration/RS_FISH.ijm")
    
    with open(ijm_file, "w") as f:
        for rd in Para.Round_channel.keys():
            input_image = RS_output.replace("\\", "/") + f"/Registration/stitched/processed/{rd}/"
            timeFile = RS_output.replace("\\", "/") + f"/Registration/stitched/processed/{rd}/RS_Exe_times.txt"
    
            macro = r"""
    
    dir = "{input_image}";
    
    timeFile = "{timeFile}";
    
    anisotropyCoefficient = {anisotropyCoefficient}; 
    ransac = "{ransac}"; 
    supportRadius = {supportRadius};
    inlierRatio = {inlierRatio};
    maxError = {maxError};  
    bsMethod = "{bsMethod}";
    bsMaxError = {bsMaxError};
    bsInlierRatio = {bsInlierRatio};
    min_number_of_inliers = {min_number_of_inliers};
    initial = {initial};
    final = {final};
    useMultithread = "use_multithreading";
    numThreads = 8;
    blockSizX = 256;
    blockSizY = 256;
    
    ///////////////////////////////////////////////////
    
    ransac_sub = split(ransac, ' ');
    ransac_sub = ransac_sub[0];
    
    bsMethod_sub = split(bsMethod, ' ');
    bsMethod_sub = bsMethod_sub[0];
    
    setBatchMode(true);
    
    ///////////////////////////////////////////////////
    
    walkFiles(dir);
    
    // Find all files in subdirs:
    function walkFiles(dir) {{
    	list = getFileList(dir);
    	for (i=0; i<list.length; i++) {{
    		if (endsWith(list[i], "/"))
    		   walkFiles(""+dir+list[i]);
    
    		// If image file
    		else  if (endsWith(list[i], ".tif")) 
    		   processImage(dir, list[i]);
    	}}
    }}
    
    function processImage(dirPath, imName) {{
    
    	if (endsWith(imName,"AF488.tif")){{
            thresholdDoG = {AF488_thresholdDoG};
            intensityThreshold = {AF488_intensityThreshold}; 
            sigmaDoG = {AF488_sigmaDoG};
        }}else if (endsWith(imName,"AF546.tif")){{
            thresholdDoG = {AF546_thresholdDoG};
            intensityThreshold = {AF546_intensityThreshold}; 
            sigmaDoG = {AF546_sigmaDoG};
        }}else if (endsWith(imName,"AF594.tif")){{
            thresholdDoG = {AF594_thresholdDoG};
            intensityThreshold = {AF594_intensityThreshold}; 
            sigmaDoG = {AF594_sigmaDoG};
        }}else if (endsWith(imName,"Cy5.tif")){{
            thresholdDoG = {Cy5_thresholdDoG};
            intensityThreshold = {Cy5_intensityThreshold}; 
            sigmaDoG = {Cy5_sigmaDoG};
        }}else if (endsWith(imName,"Cy7.tif")){{
            thresholdDoG = {Cy7_thresholdDoG};
            intensityThreshold = {Cy7_intensityThreshold}; 
            sigmaDoG = {Cy7_sigmaDoG};
        }}
    
        splitStr = split(imName, ".");
        part1 = splitStr[0];
        
        open("" + dirPath + imName);
    	results_csv_path = "" + dirPath + part1 + ".csv";
    
    
    	RSparams =  "image=" + imName + 
    	" mode=Advanced anisotropy=" + anisotropyCoefficient + " robust_fitting=[" + ransac + "] use_anisotropy" + 
    	" compute_min/max" + " sigma=" + sigmaDoG + " threshold=" + thresholdDoG + 
    	" support=" + supportRadius + " min_inlier_ratio=" + inlierRatio + " max_error=" + maxError + " spot_intensity_threshold=" + intensityThreshold + 
        " min_number_of_inliers=" + min_number_of_inliers + " initial=" + initial + " final=" + final +
    	" background=[" + bsMethod + "] background_subtraction_max_error=" + bsMaxError + " background_subtraction_min_inlier_ratio=" + bsInlierRatio + 
    	" results_file=[" + results_csv_path + "]" + " " + useMultithread + " num_threads=" + numThreads + " block_size_x=" + blockSizX + " block_size_y=" + blockSizY;
    
    	print(RSparams);
    
    	startTime = getTime();
    	run("RS-FISH", RSparams);
    	exeTime = getTime() - startTime; //in miliseconds
    	
    	// Save exeTime to file:
    	File.append(results_csv_path + "," + exeTime + "\n ", timeFile);
    
    	// Close all windows:
        close("smFISH localizations");
        close("Log");
    	run("Close All");	
    	while (nImages>0) {{ 
    		selectImage(nImages); 
    		close(); 
        }} 
    }} 
    """.format(input_image = input_image, 
               timeFile = timeFile, 
               anisotropyCoefficient = anisotropyCoefficient,
               ransac = ransac,
               AF488_sigmaDoG = sigmaDoG[rd][0],
               AF546_sigmaDoG = sigmaDoG[rd][1],
               AF594_sigmaDoG = sigmaDoG[rd][2],
               Cy5_sigmaDoG = sigmaDoG[rd][3],
               Cy7_sigmaDoG = sigmaDoG[rd][4],
               supportRadius = supportRadius,
               inlierRatio = inlierRatio,
               maxError = maxError,
               intensityThreshold = intensityThreshold,
               min_number_of_inliers = min_number_of_inliers,
               initial = initial,
               final = final,
               bsMethod = bsMethod,
               bsMaxError = bsMaxError,
               bsInlierRatio = bsInlierRatio,
               AF488_thresholdDoG = thresholdDoG[rd][0],
               AF546_thresholdDoG = thresholdDoG[rd][1],
               AF594_thresholdDoG = thresholdDoG[rd][2],
               Cy5_thresholdDoG = thresholdDoG[rd][3],
               Cy7_thresholdDoG = thresholdDoG[rd][4],
               AF488_intensityThreshold = intensityThreshold[0],
               AF546_intensityThreshold = intensityThreshold[1],
               AF594_intensityThreshold = intensityThreshold[2],
               Cy5_intensityThreshold = intensityThreshold[3],
               Cy7_intensityThreshold = intensityThreshold[4])
    
            f.write(macro)
    
    imageJ = "/media/duan/DuanLab_Data/openFISH/Decode/software/Fiji.app/ImageJ-linux64"
    
    command = [imageJ, "--headless", "--console", "-macro", ijm_file]
    
    info = "[INFO] Running RS-FISH..."
    print(info)
    subprocess.run(" ".join(command), shell = True, check = True)
    
    info = "[INFO] RS-FISH Done."
    print(info)

            
def runStarDist(Para, prob_thresh = 0.5, nms_thresh = 0.4, trained_model = '2D_versatile_fluo', sigma = 2.5):
    
    from csbdeep.utils import normalize
    from stardist import export_imagej_rois
    from stardist.models import StarDist2D
    
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian, sobel
    from skimage.segmentation import watershed
    from skimage.morphology import binary_opening
    
    import numpy as np
    import tifffile
    import os
    
    dic = os.path.join(Para.output, 'Segmentation/StarDist/')
    if not os.path.exists(dic):
        os.makedirs(dic)
        
    info = "[INFO] Loading stardist models..."
    print(info)
    
    model = StarDist2D.from_pretrained(trained_model)
    
    DAPI_image = os.path.join(Para.output, "Registration/Stitched_DAPI.tif")
    image = tifffile.imread(DAPI_image)
    
    nTiles = (image.shape[0]//2048, image.shape[1]//2048)
    
    info = "[INFO] Normalizing image..."
    print(info)
    img = normalize(image, 1,99.8, axis=(0,1))
    info = "[INFO] Running StarDist..."
    print(info)
    labels, polygons = model.predict_instances(img, 
                                               prob_thresh = prob_thresh, 
                                               nms_thresh = nms_thresh,
                                               n_tiles = nTiles, 
                                               verbose = True
                                              )
    
    try:
        export_imagej_rois(os.path.join(Para.output, 'Segmentation/StarDist/RoiSet.zip'), polygons["coord"])
        stardist_roi = True
    except:
        stardist_roi = False
    
    
    np.save(os.path.join(Para.output, 'Segmentation/StarDist/RoiSet.npy'),polygons["coord"])
    tifffile.imwrite(os.path.join(Para.output, 'Segmentation/StarDist/DAPI_stardist.tif'), labels)
    
    binary = np.where(labels > 0, 255, 0).astype(np.uint8)
    info = "[INFO] Applying the watershed segmentation on the StarDist label result..."
    print(info)
    distance = ndi.distance_transform_edt(binary)
    distance = gaussian(distance, sigma=sigma)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    watershed_result = watershed(-distance, markers, mask=binary)
    
    edges_labels = sobel(watershed_result)
    edges_binary = sobel(binary)

    edges = np.logical_xor(edges_labels != 0, edges_binary != 0)
    almost = np.logical_not(edges) * binary
    result = np.where(binary_opening(almost), 255, 0)
    
    tifffile.imwrite(os.path.join(Para.output, 'Segmentation/StarDist/DAPI_watershed.tif'), result.astype("uint8"))
    
    info = f"[INFO] All releated results have been saved into {Para.output}/Segmentation/StarDist/"
    print(info)
    
    return stardist_roi
            
def runCellpose(Para, restore_type="deblur_cyto3", diameter=45.,flow_threshold=1., cellprob_threshold=-6.):
    
    from cellpose import denoise
    from cellpose import io
    import numpy as np
    import tifffile
    import cv2
    import os
    
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from joblib import Parallel, delayed
    import shapely
    
    import torch
    
    # torch.cuda.set_per_process_memory_fraction(0.99, device=0)
    
    filedir = os.path.join(Para.output, "Segmentation/Cellpose")
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    
    info = "[INFO] Running Cellpose3..."
    print(info)
    
    img = tifffile.imread(os.path.join(Para.output, "Registration/Stitched_rRNA_DAPI.tif"))
    
    
    # https://cellpose.readthedocs.io/en/latest/benchmark.html
    gpu = True
    if min(img.shape[0:2]) > 9000:
        gpu = False
    
    model = denoise.CellposeDenoiseModel(gpu=gpu, model_type="cyto3",restore_type=restore_type, chan2_restore=True)

    masks, flows, styles, imgs_dn = model.eval(img, channels=[1,3], diameter=diameter,flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)

    io.save_masks(img, masks, flows, "Cellpose", savedir = filedir, png = False, tif = True, channels=[1, 3], save_outlines=True)
    
    info = f"[INFO] All releated results have been saved into {Para.output}/Segmentation/Cellpose/"
    print(info)
    
    info = f"[INFO] Geometrizing Cellpose segmentation masks..."
    print(info)

    def _ensure_polygon(cell: Polygon | MultiPolygon | GeometryCollection) -> Polygon:

        cell = shapely.make_valid(cell)

        if isinstance(cell, Polygon):
            if cell.interiors:
                cell = Polygon(list(cell.exterior.coords))
            return cell

        if isinstance(cell, MultiPolygon):
            return max(cell.geoms, key=lambda polygon: polygon.area)

        if isinstance(cell, GeometryCollection):
            geoms = [geom for geom in cell.geoms if isinstance(geom, Polygon)]

            if not geoms:
                # log.warn(f"Removing cell of type {type(cell)} as it contains no Polygon geometry")
                return None

            return max(geoms, key=lambda polygon: polygon.area)

        # log.warn(f"Removing cell of unknown type {type(cell)}")
        return None

    def _smoothen_cell(cell: MultiPolygon, smooth_radius: float, tolerance: float) -> Polygon | None:

        cell = cell.buffer(-smooth_radius).buffer(2 * smooth_radius).buffer(-smooth_radius)
        cell = cell.simplify(tolerance)

        return None if cell.is_empty else _ensure_polygon(cell)

    def _default_tolerance(mean_radius: float) -> float:
        if mean_radius < 10:
            return 0.3
        if mean_radius < 20:
            return 0.6
        return 0.9

    def _getContour(cell_id:int, mask:np.ndarray) ->  MultiPolygon:
        contours, _ = cv2.findContours((mask == cell_id).astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return MultiPolygon([Polygon(contour[:, 0, :]) for contour in contours if contour.shape[0] >= 4])

    mask = tifffile.imread(os.path.join(Para.output,"Segmentation/Cellpose/Cellpose_cp_masks.tif"))

    max_cell = np.max(mask)
    cells = Parallel(n_jobs=96, backend='loky')(delayed(_getContour)(cell_id, mask) for cell_id in range(1,max_cell+1))

    mean_radius = np.sqrt(np.array([cell.area for cell in cells]) / np.pi).mean()
    smooth_radius = mean_radius * 0.1
    tolerance = _default_tolerance(mean_radius)

    cells = Parallel(n_jobs=96, backend='loky')(delayed(_smoothen_cell)(cell, smooth_radius, tolerance) for cell in cells)
    cells = [cell for cell in cells if cell is not None]

    print(
        f"[INFO] Percentage of non-geometrized cells: {(max_cell - len(cells)) / max_cell:.3%} (usually due to segmentation artefacts)"
    )
    
    np.save(os.path.join(Para.output,"Segmentation/Cellpose/cellpose_polygon.npy"), cells)
     
        
def runBaysor(Para, min_molecules_per_cell = 5, count_matrix_format = "loom", prior_segmentation_confidence = 0.3, scale = 30, scale_std = "50%"):
    
    import os
    import subprocess
    
    baysor_output = os.path.join(Para.output, "Segmentation/Baysor")
    if not os.path.exists(baysor_output):
        os.makedirs(baysor_output)
    
    baysor_path = "~/software/bin/baysor/bin/baysor run"
    transcript_file = os.path.join(Para.output, "gene_location_merged_filtered.csv")

    DAPI_mask = os.path.join(Para.output, "Segmentation/StarDist/DAPI_stardist.tif")

    print("Running baysor...")
    
#     try:
        
#         # Baysor 0.6.2和0.6.1都无法使用--plot参数
#         command = [baysor_path, transcript_file, DAPI_mask, "-x x -y y -g gene", "--min-molecules-per-cell", str(min_molecules_per_cell),
#                   "--output", baysor_output, "--polygon-format", "GeometryCollection", "--count-matrix-format", count_matrix_format, 
#                    "--prior-segmentation-confidence", str(prior_segmentation_confidence)]
        
#         subprocess.run(" ".join(command), shell = True, check = True)
        
    # except:
        
    command = [baysor_path, transcript_file, "-x x -y y -g gene", "--min-molecules-per-cell", str(min_molecules_per_cell),
              "--output", baysor_output, "--polygon-format", "GeometryCollection", "--count-matrix-format", count_matrix_format, 
               "--scale", str(scale), "--scale-std", scale_std]

    subprocess.run(" ".join(command), shell = True, check = True)

    # finally:        
    #     print(" ".join(command))

def Parallel_sopa_geometrize(workpath):
    
    """
    Parallel version of sopa.segmentation.shapes.geometrize
    https://gustaveroussy.github.io/sopa/api/segmentation/shapes/
    """
    import os
    import numpy as np
    
#     info = f"Using FileZillaClient -> Transfer {workpath}/Segmentation/Cellpose/Cellpose_cp_masks.tif to /home/Fish/workspace/cellpose/Cellpose_cp_masks.tif -> \
#     Using Xshell -> cd /home/Fish/workspace/cellpose -> ~/miniconda3/envs/bidcell/bin/python3.10 sopa_geometrize.py \
#     Using FileZillaClient -> Transfer /home/Fish/workspace/cellpose/Cellpose_cp_masks.tif to {workpath}/Segmentation/Cellpose/cellpose_polygon.npy"
    
#     print(info)

    pipeline_check = os.path.exists(os.path.join(workpath, "Segmentation/Cellpose/cellpose_polygon.npy"))
    
    while pipeline_check == False:
        
        print("[info] No polygon file detected.")
        pipeline_check_input = input("Please run the sopa.geometrize using linux, Input ANYTHING when finished.")
        pipeline_check = os.path.exists(os.path.join(workpath, "Segmentation/Cellpose/cellpose_polygon.npy"))
    
    cells = np.load(os.path.join(workpath, "Segmentation/Cellpose/cellpose_polygon.npy") ,allow_pickle=True)
    
    return cells


def runStardist_and_Cellpose(Para, prob_thresh = 0.5, nms_thresh = 0.4, trained_model = '2D_versatile_fluo', sigma = 2.5,
                            restore_type="deblur_cyto3", diameter=45.,flow_threshold=1., cellprob_threshold=-6.):
    
    import torch
    import gc
    
    if Para.rRNAseg:
    
        runCellpose(Para,restore_type=restore_type, diameter=diameter,flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
        torch.cuda.empty_cache()
        gc.collect()
            
    stardist_roi = runStarDist(Para, prob_thresh = prob_thresh, nms_thresh = nms_thresh, trained_model = trained_model, sigma = sigma)
    
    return stardist_roi

def _run_single_spotiflow(img,
                          model = Spotiflow.from_pretrained("general"),
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


def runSpotiflow(Para,
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
                device = 'cuda', **kargs
                ):
    
    model = Spotiflow.from_pretrained(model_name)
    tqdm_bar = tqdm(total = 100, bar_format = '{l_bar}{bar}| {n:.0f}/{total} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]')
    tqdm_step = 100/sum([len(x) - 1 for x in Para.Round_channel.values()])
    
    for rd in Para.Round_list:
        for ch in Para.Round_channel[rd]:
            if ch in ['AF488', 'AF546', 'AF594', 'Cy5', 'Cy7']:
                tqdm_bar.set_description(f'Running Spotiflow: {rd}_{ch}')
                
                input_image = imread(os.path.join(Para.output, f"Registration/stitched/processed/{rd}/{rd}_{ch}.tif"))
                output_df = os.path.join(Para.output, f"Registration/stitched/processed/{rd}/{rd}_{ch}.csv")
    
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
                
                df.to_csv(output_df, index = False)
    
                tqdm_bar.update(tqdm_step)

    tqdm_bar.close()
