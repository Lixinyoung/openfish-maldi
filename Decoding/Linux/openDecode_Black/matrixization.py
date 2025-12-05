"""
Author: Li Xinyang
Last modified: 2025.9.28


Change log:
        
    2025.9.28
        New script
        
"""

import warnings
warnings.filterwarnings('ignore')

import logging

from spatialdata.models import Image2DModel, PointsModel,ShapesModel,TableModel
from shapely.geometry import Polygon
from spatialdata import SpatialData
from scipy.spatial import KDTree
import sopa.segmentation as sseg
from shutil import rmtree
import geopandas as gpd
from typing import List
from tqdm import tqdm
import scanpy as sc
import pandas as pd
import numpy as np
import tifffile
import shapely
import os

from . import explorer

log = logging.getLogger(__name__)



def _mark_low_confidence_point(transcripts_df, threshold: float = 17.0, scaler:float = 0.325, nb_workers: int = 16):
    
    """
    If a spots's nearest same coding spots is too far, mark it with a qv = 10
    threshold: in micrometer
    """
    
    threshold = threshold / scaler
    
    tmp_df = transcripts_df.copy()
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = nb_workers)
    
    def _filter_dist(group, threshold = 30):
        
        tree = KDTree(group[['x', 'y']])
        dist, _ = nearest_pairs = tree.query(group[['x', 'y']], k = 2)
        group.loc[dist[:,1] > threshold, 'qv'] = 10.0
        
        return group
    
    return tmp_df.groupby('gene').parallel_apply(lambda group: _filter_dist(group, threshold)).reset_index(drop = True)



def _merge_multi_geometry(
    geometries: List[gpd.GeoDataFrame],
    priorities: List[int] = None
) -> gpd.GeoDataFrame:
    """
    Merge multiple segmentation GeoDataFrames by resolving overlaps using priority rules.

    Parameters
    ----------
    geometries : List[gpd.GeoDataFrame]
        List of GeoDataFrames containing segmentation polygons.
    priorities : List[int], optional
        Priority for each GeoDataFrame (higher = more dominant). 
        Default: [0] * len(geometries)

    Example
    -------
    If rRNA should dominate DAPI:
        geometries = [DAPI_seg, rRNA_seg]
        priorities = [1, 2]  # higher number = higher priority

    Returns
    -------
    gpd.GeoDataFrame
        Merged GeoDataFrame with non-overlapping polygons (only 'Polygon' geometries).
    """
    if priorities is None:
        priorities = [0] * len(geometries)
    
    if len(geometries) != len(priorities):
        raise ValueError("Length of 'geometries' and 'priorities' must match.")

    # Add priority column and concatenate
    all_cells = []
    for gdf, p in zip(geometries, priorities):
        gdf = gdf.copy()  # avoid modifying original
        gdf['priority'] = p
        all_cells.append(gdf[['geometry', 'priority']])
    
    merged = pd.concat(all_cells, ignore_index=True)
    merged = merged.reset_index(drop = True)
    
    # Precompute areas for efficiency
    merged['area'] = merged.geometry.area

    n = len(merged)
    if n == 0:
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')
    
    cells = merged['geometry'].to_list()
    areas = merged['area'].to_list()
    priorities = merged['priority'].to_list()
    
    keep_indices = [True] * len(cells)
    
    # Build spatial index
    tree = shapely.STRtree(cells)
    conflicts = tree.query(cells, predicate="intersects")
    # Filter out self-intersections
    conflicts = conflicts[:, conflicts[0] != conflicts[1]].T

    log.info(f"Resolving {len(conflicts)} segmentation conflicts")


    for i, j in tqdm(conflicts, disable=len(conflicts) == 0, dynamic_ncols = True):
        if not (keep_indices[i] and keep_indices[j]):
            continue  # Skip if either already discarded

        geom_i = cells[i]
        geom_j = cells[j]
        pri_i = priorities[i]
        pri_j = priorities[j]
        area_i = areas[i]
        area_j = areas[j]

        if pri_i == pri_j:
            # Equal priority: subtract each other and take convex hull
            diff_i = geom_i.difference(geom_j)
            diff_j = geom_j.difference(geom_i)
            cells[i] = shapely.convex_hull(diff_i)
            cells[j] = shapely.convex_hull(diff_j)
            
        elif pri_i > pri_j:
            # i dominates j
            
            inter_area = geom_i.intersection(geom_j).area
            
            if inter_area >= 0.3 * area_j:
                keep_indices[j] = False
            elif inter_area >= 0.7 * area_i:
                keep_indices[i] = False
            else:
                diff_j = geom_j.difference(geom_i)
                cells[j] = shapely.convex_hull(diff_j)
        else:
            
            inter = geom_i.intersection(geom_j)
            inter_area = inter.area
            
            # j dominates i
            if inter_area >= 0.3 * area_i:
                keep_indices[i] = False
            elif inter_area >= 0.7 * area_j:
                keep_indices[j] = False
            else:
                diff_i = geom_i.difference(geom_j)
                cells[i] = shapely.convex_hull(diff_i)

    # Apply keep mask
    
    merged = gpd.GeoDataFrame(geometry=cells)
    
    result = merged.loc[keep_indices,:].reset_index(drop = True)

    # Keep only valid Polygons (exclude MultiPolygon, Point, LineString, etc.)
    is_polygon = result.geometry.geom_type == 'Polygon'
    result = result.loc[is_polygon].copy()

    return result
            


class Matrixization():
    
    def __init__(self, para, buffer_radius:float = 3, points_distance_threshold: float = 17.0):
        
        self.para = para
        self.buffer_radius = buffer_radius
        self.scaler = self.para.pixel_scaler[self.para.OBJECTIVE]
        self.points_distance_threshold = points_distance_threshold
        
    def _prepare_shapes(self):
        
        shapes = {}
        
        geometries = []
        priorities = []
        
        try:
            rois = np.load(os.path.join(self.para.OUTPUT_PATH, 'Segmentation/StarDist/RoiSet.npy'))
            polygon_list = []
            for i in range(len(rois)):
                polygon = Polygon(np.stack([rois[i,1,:], rois[i,0,:]], axis=1))
                polygon_list.append({'geometry': polygon})

            cells = gpd.GeoDataFrame(polygon_list)
            cells = cells[cells.geometry.is_valid].copy()
            cells['geometry'] = cells['geometry'].apply(lambda x: x.buffer(distance = self.buffer_radius / self.scaler))
            
            geometries.append(cells)
            priorities.append(1)
            
            shapes[f'{self.para.ANCHOR_CHANNEL}_boundaries'] = ShapesModel.parse(cells)
            
        except FileNotFoundError:
            
            log.warning(f"No segmentation found for {self.para.ANCHOR_CHANNEL}")
            
        
        if self.para.EXTRA and 'rRNA' in self.para.EXTRA_NAMES:
            
            cells = np.load(os.path.join(self.para.OUTPUT_PATH, "Segmentation/Cellpose/cellpose_polygon.npy") ,allow_pickle=True)
            cells = gpd.GeoDataFrame({"geometry":cells})

            # Buffer 2 pixel for edge handle
            cells = cells[cells.geometry.is_valid].copy()
            cells['geometry'] = cells['geometry'].apply(lambda x: x.buffer(distance = 2))
            geometries.append(cells)
            priorities.append(2)
            
            shapes[f'cytoRNA_boundaries'] = ShapesModel.parse(cells)
        
        if geometries:
            shapes[f'cell_boundaries'] = ShapesModel.parse(_merge_multi_geometry(geometries, priorities))
            
            np.save(os.path.join(self.para.OUTPUT_PATH, 'tmp/shapes.npy'), shapes)
            
            return shapes
        
        else:
            
            np.save(os.path.join(self.para.OUTPUT_PATH, 'tmp/shapes.npy'), {})
            
            return {}
        
        
    def _prepare_points(self, points_distance_threshold: float = 17.0):
            
        points = {}
        
        transcripts_df = pd.read_parquet(os.path.join(self.para.OUTPUT_PATH, "tmp/Decoded_all.parquet"), columns=['x', 'y', 'gene', 'qv'])
        
        transcripts_df['x'] = transcripts_df['x'].astype(np.float_)
        transcripts_df['y'] = transcripts_df['y'].astype(np.float_)
        transcripts_df['qv'] = transcripts_df['qv'].astype(np.float_)
        
        marked_transcripts = _mark_low_confidence_point(transcripts_df, 
                                                        threshold = points_distance_threshold,
                                                        scaler = self.scaler,
                                                        nb_workers = self.para.THREADS)
        
        marked_transcripts.to_parquet(os.path.join(self.para.OUTPUT_PATH, "./Decoded_transcripts.parquet"))
        
        points['transcripts'] = PointsModel.parse(marked_transcripts, coordinates = {'x': 'x', 'y': 'y'}, feature_key = "gene")
        points['transcripts_filtered'] = PointsModel.parse(marked_transcripts[marked_transcripts['qv'] > 20].copy(), coordinates = {'x': 'x', 'y': 'y'}, feature_key = "gene")
        
        return points
    
    
    def _prepare_images(self):
        
        images = {}
        
        if self.para.EXTRA:
            
            fluor_image = tifffile.imread(os.path.join(self.para.OUTPUT_PATH, f"Registration/morphology.tif"))
            images["fluorescence"] = Image2DModel.parse(fluor_image, dims = ("c","y","x"))
            
            cpd = pd.read_csv(os.path.join(self.para.OUTPUT_PATH, f"Registration/morphology_meta.csv"))
            channel_names = cpd['name'].to_numpy()

            images["fluorescence"] = images["fluorescence"].assign_coords(c=channel_names)
            
            
        else:
            
            fluor_image = tifffile.imread(os.path.join(self.para.OUTPUT_PATH, f"Registration/morphology_{self.para.ANCHOR_CHANNEL}.tif"))
            
            fluor_image = np.expand_dims(fluor_image, axis = 0)
            images["fluorescence"] = Image2DModel.parse(fluor_image)
            
            images["fluorescence"] = images["fluorescence"].assign_coords(c=[self.para.ANCHOR_CHANNEL])
        
    
        return images
    
    
    def _prepare_tables(self, adata, sdata, shapes_key: str = 'cell_boundaries'):
        
        adata.obs["region"] = shapes_key
        adata.obs["region"] = adata.obs["region"].astype("category")
        adata.obs["Name"] = adata.obs_names.astype('int')
        adata.obs["x"] = sdata[shapes_key].apply(lambda x: x.centroid.x).to_numpy()
        adata.obs["y"] = sdata[shapes_key].apply(lambda x: x.centroid.y).to_numpy()
        adata.obs["area"] = sdata[shapes_key].apply(lambda x: x.area).to_numpy()
        adata.layers['counts'] = adata.X.copy()
        
        if len(adata.var_names) > 60 or self.para.run_basic_clustering:
            
            adata_back = adata.copy()
            
            adata.var["FP"] = adata.var_names.str.startswith(("sFP", "FP", "FalsePositive"))
            
            sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=True, percent_top=None,qc_vars=['FP'])
            
            adata = adata[(adata.obs["area"] > 300) & (adata.obs["area"] < (3 * adata.obs["area"].mean())),:].copy()
            
            above_thrd = np.percentile(adata.obs["total_counts"], 99.9)
            below_thrd = np.percentile(adata.obs["total_counts"], 1)
            
            sc.pp.filter_cells(adata, max_counts=above_thrd)
            sc.pp.filter_cells(adata, min_counts=below_thrd)
            
            # Normalize using cell area
            from scipy.sparse import csr_matrix
            cell_area = np.array(adata.obs["area"])
            adata.X = csr_matrix((adata.X.T / cell_area).T)

            # Normalizing to library size
            sc.pp.normalize_total(adata, target_sum=1000)
            # Logarithmize the data
            sc.pp.log1p(adata)

            sc.tl.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=2, flavor="igraph",n_iterations=-1)
            
            adata_back.obs['leiden'] = 'Unasssigned'
            adata_back.obs.loc[adata.obs_names, 'leiden'] = adata.obs['leiden']
            
            adata = adata_back.copy()
            
        adata.write_h5ad(os.path.join(self.para.OUTPUT_PATH, 'adata.h5ad'))
        
        return adata
        
        
    def matrixization(self, force = False):
        
        images = self._prepare_images()
        
        PROGRESS_DICT = self.para._parse_progress_yaml()
        
        if not PROGRESS_DICT['matrixization']['resolve_conflits'] or force:
            
            shapes = self._prepare_shapes()
            
            PROGRESS_DICT['matrixization']['resolve_conflits'] = True
                
            self.para.save_progress_yaml(PROGRESS_DICT)
            
        else:
            
            shapes = np.load(os.path.join(self.para.OUTPUT_PATH, 'tmp/shapes.npy'), allow_pickle = True).item()
            
        if shapes:
            
            shapes_key = 'cell_boundaries'
            
            points_distance_threshold = 2 * np.sqrt(shapes[shapes_key].geometry.area / np.pi).mean()
        
            
        else:
            
            points_distance_threshold = self.points_distance_threshold
            
            
        log.info(f'Fitering transcripts within distance {points_distance_threshold * self.scaler: .1f} um...')
        points = self._prepare_points(points_distance_threshold)
            
        
        if shapes:
            
            sdata = SpatialData(images = images, points = points, shapes = shapes)

            log.info(f'Aggregating transcripts into {shapes_key}')
            
            PROGRESS_DICT = self.para._parse_progress_yaml()
            
            if not PROGRESS_DICT['matrixization']['assign_transcripts'] or force:
            
                adata = sseg.aggregation.count_transcripts(sdata, gene_column = "gene", shapes_key=shapes_key, points_key='transcripts_filtered')
            
                log.info(f'Running basic clustering...')

                adata = self._prepare_tables(adata, sdata, shapes_key)
                
                PROGRESS_DICT['matrixization']['assign_transcripts'] = True
                
                self.para.save_progress_yaml(PROGRESS_DICT)
            
            else:
                
                adata = sc.read_h5ad(os.path.join(self.para.OUTPUT_PATH, 'adata.h5ad'))
            
            sdata.table = TableModel.parse(adata, region=shapes_key, region_key="region", instance_key='Name')
            
            save_h5ad = True
            
        else:
            
            shapes_key = None
            save_h5ad = False
            
            sdata = SpatialData(images = images, points = points)
            
            
        raw_sdata_path = os.path.join(self.para.OUTPUT_PATH, "raw_sdata.zarr")
        xenium_explorer_path = os.path.join(self.para.OUTPUT_PATH, "Xenium_explorer")
        
        for path in [raw_sdata_path, xenium_explorer_path]:
            if os.path.exists(path):
                rmtree(path)

        sdata.write(raw_sdata_path)

        explorer.write(xenium_explorer_path, 
                       sdata, gene_column = "gene", 
                       image_key="fluorescence",
                       shapes_key = shapes_key,
                       points_key = "transcripts",
                       save_h5ad = save_h5ad,
                       pixel_size = self.scaler,
                       polygon_max_vertices = 17,
                       layer = 'counts')
            
            
    
    