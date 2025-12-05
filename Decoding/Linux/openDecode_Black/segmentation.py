"""
Author: Li Xinyang
Last modified: 2025.9.26


Change log:
        
    2025.9.26
        New script
        
"""

import logging

import numpy as np
import tifffile
import os

log = logging.getLogger(__name__)


def runStarDist(para, prob_thresh = 0.5, nms_thresh = 0.4, trained_model = '2D_versatile_fluo', sigma = 2.5, force = False):
    
    from csbdeep.utils import normalize
    from stardist import export_imagej_rois
    from stardist.models import StarDist2D

    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian, sobel
    from skimage.segmentation import watershed
    from skimage.morphology import binary_opening
    
    dic = os.path.join(para.OUTPUT_PATH, 'Segmentation/StarDist/')
    if not os.path.exists(dic):
        os.makedirs(dic)
        
    PROGRESS_DICT = para._parse_progress_yaml()
    
    if not PROGRESS_DICT['segmentation'][para.ANCHOR_CHANNEL] or force:
        
        log.info("Loading stardist models...")

        model = StarDist2D.from_pretrained(trained_model)

        DAPI_image = os.path.join(para.OUTPUT_PATH, f"Registration/morphology_{para.ANCHOR_CHANNEL}.tif")
        image = tifffile.imread(DAPI_image)

        nTiles = (image.shape[0]//2048, image.shape[1]//2048)

        log.info("Normalizing image...")
        img = normalize(image, 1,99.8, axis=(0,1))
        log.info("Running StarDist...")

        labels, polygons = model.predict_instances(img, 
                                                   prob_thresh = prob_thresh, 
                                                   nms_thresh = nms_thresh,
                                                   n_tiles = nTiles, 
                                                   verbose = True
                                                  )

        export_imagej_rois(os.path.join(para.OUTPUT_PATH, 'Segmentation/StarDist/RoiSet.zip'), polygons["coord"])



        np.save(os.path.join(para.OUTPUT_PATH, 'Segmentation/StarDist/RoiSet.npy'),polygons["coord"])
        tifffile.imwrite(os.path.join(para.OUTPUT_PATH, f'Segmentation/StarDist/{para.ANCHOR_CHANNEL}_stardist.tif'), labels)
        
        PROGRESS_DICT['segmentation'][para.ANCHOR_CHANNEL] = True
        para.save_progress_yaml(PROGRESS_DICT)
        

    
    log.info(f"All releated results have been saved into {para.OUTPUT_PATH}/Segmentation/StarDist/")

    import tensorflow as tf
    
    tf.keras.backend.clear_session()
    
    return None



def runCellpose(para, restore_type="deblur_cyto3", diameter=45.,flow_threshold=1., cellprob_threshold=-6., force = False):
    
    import torch 
    from cellpose import denoise
    from cellpose import io
    
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, LinearRing
    import shapely
    import shapely.affinity
    
    import geopandas as gpd
    from geopandas import GeoDataFrame
    
    import skimage
    from skimage.measure._regionprops import RegionProperties
    
    PROGRESS_DICT = para._parse_progress_yaml()
    
    if not PROGRESS_DICT['segmentation']['rRNA']['segmentation'] or force:
    
        filedir = os.path.join(para.OUTPUT_PATH, "Segmentation/Cellpose")
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        log.info("Running Cellpose3...")

        img = tifffile.imread(os.path.join(para.OUTPUT_PATH, "Registration/morphology_rRNA.tif"))


        # https://cellpose.readthedocs.io/en/latest/benchmark.html
        gpu = True
        batch_size = 8
        
        if min(img.shape[0:2]) > 10000:
            gpu = False
        elif min(img.shape[0:2]) > 5000:
            batch_size = 4
        else:
            pass

        model = denoise.CellposeDenoiseModel(gpu=gpu, model_type="cyto3",restore_type=restore_type, chan2_restore=True)

        masks, flows, styles, imgs_dn = model.eval(img, channels=[1,3], diameter=diameter,flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, batch_size = batch_size)

        io.save_masks(img, masks, flows, "Cellpose", savedir = filedir, png = False, tif = True, channels=[1, 3], save_outlines=True)

        log.info(f"All releated results have been saved into {para.OUTPUT_PATH}/Segmentation/Cellpose/")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    PROGRESS_DICT['segmentation']['rRNA']['segmentation'] = True
    para.save_progress_yaml(PROGRESS_DICT)
    
    log.info(f"Geometrizing Cellpose segmentation masks...")

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
    
    
    def _region_props_to_multipolygon(region_props: RegionProperties, allow_holes: bool) -> Polygon | MultiPolygon:
        mask = np.pad(region_props.image, 1)
        contours = skimage.measure.find_contours(mask, 0.5)

        rings = [LinearRing(contour[:, [1, 0]]) for contour in contours if contour.shape[0] >= 4]

        exteriors = [ring for ring in rings if ring.is_ccw]

        if allow_holes:
            holes = [ring for ring in rings if not ring.is_ccw]

        def _to_polygon(exterior: LinearRing) -> Polygon:
            exterior_poly = Polygon(exterior)

            _holes = None
            if allow_holes:
                _holes = [hole.coords for hole in holes if exterior_poly.contains(Polygon(hole))]

            return Polygon(exterior, holes=_holes)

        polygon = [_to_polygon(exterior) for exterior in exteriors]
        polygon = MultiPolygon(polygon) if len(polygon) > 1 else polygon[0]

        yoff, xoff, *_ = region_props.bbox
        return shapely.affinity.translate(polygon, xoff - 1, yoff - 1)  # remove padding offset
    
    
    def _vectorize_mask(mask: np.ndarray, allow_holes: bool = False) -> GeoDataFrame:
        
        if mask.max() == 0:
            return GeoDataFrame(geometry=[])

        regions = skimage.measure.regionprops(mask)

        return GeoDataFrame(geometry=[_region_props_to_multipolygon(region, allow_holes) for region in regions])
    

    def vectorize(mask: np.ndarray, tolerance: float | None = None, smooth_radius_ratio: float = 0.1) -> gpd.GeoDataFrame:
        """Convert a cells mask to multiple `shapely` geometries. Inspired from https://github.com/Vizgen/vizgen-postprocessing

        Args:
            mask: A cell mask. Non-null values correspond to cell ids
            tolerance: Tolerance parameter used by `shapely` during simplification. By default, define the tolerance automatically.
            smooth_radius_ratio: Ratio of the cell radius used to smooth the cell polygon.

        Returns:
            GeoDataFrame of polygons representing each cell ID of the mask
        """
        max_cells = mask.max()

        if max_cells == 0:
            log.warning("No cell was returned by the segmentation")
            return gpd.GeoDataFrame(geometry=[])

        cells = _vectorize_mask(mask)

        mean_radius = np.sqrt(cells.area / np.pi).mean()
        smooth_radius = mean_radius * smooth_radius_ratio

        tolerance = _default_tolerance(mean_radius) if tolerance is None else tolerance

        cells.geometry = cells.geometry.map(lambda cell: _smoothen_cell(cell, smooth_radius, tolerance))
        cells = cells[~cells.is_empty]

        return cells
    
    
    PROGRESS_DICT = para._parse_progress_yaml()
    
    if not PROGRESS_DICT['segmentation']['rRNA']['vectorize'] or force:
    
        mask = tifffile.imread(os.path.join(para.OUTPUT_PATH,"Segmentation/Cellpose/Cellpose_cp_masks.tif"))

        max_cell = np.max(mask)

        cells = vectorize(mask)

        log.info(f"Percentage of non-geometrized cells: {(max_cell - len(cells)) / max_cell:.3%} (usually due to segmentation artefacts)")

        np.save(os.path.join(para.OUTPUT_PATH,"Segmentation/Cellpose/cellpose_polygon.npy"), cells.geometry)
        
        PROGRESS_DICT['segmentation']['rRNA']['vectorize'] = True
        para.save_progress_yaml(PROGRESS_DICT)
        