from spatialdata.models import Image2DModel, PointsModel,ShapesModel,TableModel
from shapely.geometry import Polygon
from spatialdata import SpatialData
from scipy.spatial import KDTree
import sopa.segmentation as sseg
from shutil import rmtree
from tqdm import tqdm
import spatialdata_plot
import geopandas as gpd
import scanpy as sc
import pandas as pd
import numpy as np
import tifffile
import sopa.io
import json
import cv2
import os

from .PointsCalling import _draw_spots
from .ThirdParty import Parallel_sopa_geometrize

def plotXenium(coord, Para, stardist_roi):
    
    scalar = {"20X": 0.325, "40X":0.1625}
    objective = Para.objective
    
    images = {}
    points = {}
    shapes = {}
    
    info = f'[INFO] Note: Unfiltered points will be used to write Xenium_explorer_raw'
    print(info)
    points['raw_coord'] = PointsModel.parse(coord, coordinates = {'x': 'x', 'y': 'y'}, feature_key = "gene")
    raw_coord = coord.copy()
    
    # if not os.path.exists(os.path.join(Para.output, "Decoded_filtered_DistFiltered.parquet")):
    info = f'[INFO] Filtering transcripts according to nearest distance (threshold = 30um)'
    print(info)
    dist_coord = mark_low_confidence_point_dist(coord, scaler = scalar[objective])
    info = f'[INFO] Saving distance filtered transcripts to {os.path.join(Para.output, "Decoded_filtered_DistFiltered.parquet")}'
    print(info)
    info = f'[INFO] Note: This file will be used to write Xenium_explorer_distFiltered'
    print(info)
    dist_coord.to_parquet(os.path.join(Para.output, "Decoded_filtered_DistFiltered.parquet"))
    points['dist_coord'] = PointsModel.parse(dist_coord, coordinates = {'x': 'x', 'y': 'y'}, feature_key = "gene")
    # else:
    #     info = f'[INFO] Distance filtered transcripts file ({os.path.join(Para.output, "Decoded_filtered_DistFiltered.parquet")}) exists.'
    #     print(info)
    #     dist_coord = pd.read_parquet(os.path.join(Para.output, "Decoded_filtered_DistFiltered.parquet"))
    #     points['dist_coord'] = PointsModel.parse(dist_coord, coordinates = {'x': 'x', 'y': 'y'}, feature_key = "gene")
    #     pass
    
    
    
    if Para.extraseg or Para.rRNAseg:
        cellpose_mask = tifffile.imread(os.path.join(Para.output, "Registration/Stitched_CWH.tif"))
        images["fluorescence"] = Image2DModel.parse(cellpose_mask, dims = ("c","y","x"))
        
    else:
        dapi_raw = tifffile.imread(os.path.join(Para.output, "Registration/Stitched_DAPI.tif"))
        dapi_raw = np.expand_dims(dapi_raw, axis = 0)
        images["fluorescence"] = Image2DModel.parse(dapi_raw)
    
    # Multimodel segmentation handle
    cells = _MultiModel_Shape(Para, scaler = scalar[objective], stardist_roi = stardist_roi)
    
    if isinstance(cells, str):
        points["coord"] = PointsModel.parse(coord, coordinates = {'x': 'x', 'y': 'y'}, feature_key = "gene")
        sdata = SpatialData(images = images, points = points, shapes = shapes)
        shapes_key = None
        info = f'[INFO] No available shapes, pass existence-time filter.'
        print(info)
    else:
        
        shapes_key = "MultiModel"
        save_h5ad = True
        
        # Remove low confidence points
        # cells, coord = _remove_low_confidence_points(cells, coord, threshold = 1)
        # info = f'[INFO] Saving existence-time filtered transcripts to {os.path.join(Para.output, "Decoded_filtered_CountFiltered.parquet")}'
        # print(info)
        # info = f'[INFO] Note: This file will be used to write Xenium_explorer_CellFiltered'
        # print(info)
        # coord.to_parquet(os.path.join(Para.output, "Decoded_filtered_CountFiltered.parquet"))
        
        shapes[shapes_key] = ShapesModel.parse(cells)
        # points["coord"] = PointsModel.parse(coord, coordinates = {'x': 'x', 'y': 'y'}, feature_key = "gene")
        
        sdata = SpatialData(images = images, points = points, shapes = shapes)
        info = f'[INFO] Counting transcripts using dist_coord.'
        print(info)
        adata = sseg.aggregation.count_transcripts(sdata, gene_column = "gene", shapes_key=shapes_key, points_key='dist_coord')

        adata.obs["region"] = shapes_key
        adata.obs["region"] = adata.obs["region"].astype("category")
        adata.obs["Name"] = adata.obs_names.astype('int')
        adata.obs["x"] = sdata[shapes_key].apply(lambda x: x.centroid.x).to_numpy()
        adata.obs["y"] = sdata[shapes_key].apply(lambda x: x.centroid.y).to_numpy()
        adata.obs["area"] = sdata[shapes_key].apply(lambda x: x.area).to_numpy()
        table = TableModel.parse(adata, region=shapes_key, region_key="region", instance_key='Name')

        sdata.table = adata

#     sdata.pl.render_images(cmap = "gist_yarg", image_key = "fluorescence").pl.render_points(color ="gene", size=0.001).pl.show(figsize = (10, 10), save = "raw.png", dpi = 300)

#     if os.path.exists(os.path.join(Para.output, "transcripts.png")):
#         os.remove(os.path.join(Para.output, "transcripts.png"))
        
#     os.renames("./figures/raw.png", os.path.join(Para.output, "transcripts.png"))
    
    if os.path.exists(Para.output + "/raw_sdata.zarr"):
        rmtree(Para.output + "/raw_sdata.zarr")
        
    # if os.path.exists(Para.output + "/Xenium_explorer_CellFiltered"):
    #     rmtree(Para.output + "/Xenium_explorer_CellFiltered")
                      
    sdata.write(Para.output + "/raw_sdata.zarr")
    
    # sopa.io.explorer.write(Para.output + "/Xenium_explorer_CellFiltered", 
    #                sdata, gene_column = "gene", 
    #                image_key="fluorescence",
    #                shapes_key = shapes_key,
    #                points_key = "coord",
    #                save_h5ad = save_h5ad,
    #                pixel_size =scalar[objective])
    
    if os.path.exists(Para.output + "/Xenium_explorer_raw"):
        rmtree(Para.output + "/Xenium_explorer_raw")
        
    sopa.io.explorer.write(Para.output + "/Xenium_explorer_raw", 
                   sdata, gene_column = "gene", 
                   image_key="fluorescence",
                   shapes_key = shapes_key,
                   points_key = "raw_coord",
                   save_h5ad = save_h5ad,
                   pixel_size =scalar[objective])
    
    if os.path.exists(Para.output + "/Xenium_explorer_distFiltered"):
        rmtree(Para.output + "/Xenium_explorer_distFiltered")
        
    sopa.io.explorer.write(Para.output + "/Xenium_explorer_distFiltered", 
                   sdata, gene_column = "gene", 
                   image_key="fluorescence",
                   shapes_key = shapes_key,
                   points_key = "dist_coord",
                   save_h5ad = save_h5ad,
                   pixel_size =scalar[objective])

    
def plotGene(Points, Para, image_size):
    
    info = "[INFO] Plotting stitched spots for each genes..."
    print(info)
    gene_list = np.unique(Points["gene"].to_numpy())
    for g in tqdm(gene_list):
        spots = Points[Points["gene"] == g].copy()
        spots = spots[["x", "y"]].to_numpy()
        spot_plot = _draw_spots(spots, image_size)
        fn = os.path.join(Para.output, f"Registration/stitched/gene/{g}_spots.tif")  
        tifffile.imwrite(fn, spot_plot)
        
def plot_probs(Points, Para):
    
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    info = "[INFO] Plotting probs..."
    print(info)
    
    unique_genes = set(Points["gene"])
    gn = len(unique_genes) // 5 + np.min([1, len(unique_genes) % 5])
    fig, axes = plt.subplots(nrows = gn, ncols=5, figsize=(20, 4 * gn))

    combs = []
    for nc in range(gn):
        for nr in range(5):
            combs.append((nc,nr))
    
    if gn == 1:
        for i,g in enumerate(unique_genes):
            axes[i].hist(Points.loc[Points["gene"] == g, "probs"], bins = 50)
            axes[i].set_title(f"{g}_qv")
    else:  
        for i,g in enumerate(unique_genes):
            axes[combs[i]].hist(Points.loc[Points["gene"] == g, "probs"], bins = 50)
            axes[combs[i]].set_title(f"{g}_qv")
    
    plt.savefig(os.path.join(Para.output, "qv.png"))
    
    

def _remove_low_confidence_points(polygon_gdf, point_df, threshold = 1):
    
    """
    This function is used to remove low confidence points from the polygon, defined by existence frequency;
    
    Default, if a gene exists only 1 time in a cell(polygon), it will be removed;
    
    Also, a polygon has no points after above clearance will also be removed;
    
    return filtered_polygon_gdf, filtered_point_df
    
    filtered_polygon_gdf, filtered_point_df = _remove_low_confidence_points(polygon_gdf, point_df, threshold = 1)
    
    """
    
    info = '[INFO] Removing transcripts accroding to existence time...'
    print(info)
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = os.cpu_count() - 1)
    
    def group_handle(group, threshold):
        count = group['gene'].value_counts()
        count = count[count > threshold]
        return group[group['gene'].isin(count.index.to_numpy())]
    
    
    # Translate x,y columns to geoSeries
    geometry = gpd.GeoSeries.from_xy(point_df['x'], point_df['y'])
    point_gdf = gpd.GeoDataFrame(point_df, geometry=geometry)

    # set how = 'inner' to remove first round empty polygons
    tmp_gpd = gpd.sjoin(polygon_gdf, point_gdf, how='inner') 
    
    # Remove low exsitence points
    tmp_gpd = tmp_gpd.groupby(tmp_gpd.index).parallel_apply(lambda group:group_handle(group, threshold))
    
    # Prepare outputs
    filtered_point_df = tmp_gpd.iloc[:,2:].reset_index(drop = True).copy()
    
    filtered_polygon_gdf = tmp_gpd.loc[:,['geometry']].reset_index()
    keep_indices = [0]
    for i in range(1,len(filtered_polygon_gdf)):
        if filtered_polygon_gdf.loc[i, 'level_0'] != filtered_polygon_gdf.loc[i-1, 'level_0']:
            keep_indices.append(i)
            
    filtered_polygon_gdf = gpd.GeoDataFrame(geometry = filtered_polygon_gdf.loc[keep_indices, 'geometry']).reset_index(drop = True).copy()
    # filtered_polygon_gdf = tmp_gpd.loc[:,['geometry']].drop_duplicates(subset = 'geometry').reset_index(drop = True).copy()
    
    return filtered_polygon_gdf, filtered_point_df


def _MultiModel_Shape(Para, scaler = 0.325, stardist_roi = True):
    
    """
    This function will decide how to keep the stardist, cellpose and baysor results;
    
    If cellpose if available(aka rRNA segmentation), any other components intersect with rRNA will be dropped;
    
    Cellpose segmentation will be buffered for 2 pixel for edge handle;
    
    Then baysor results will be checked, remaining stardist results intersecting with baysor will be dropped;
    
    Finally, stardist results(nuclei segmentation) will be used to expansion and conflict resolvation;
    
    Specifically, DAPI polygon will be buffered for a distance Para.scalar['objective'] * 10;
    
    And all intersection will be located and resolved according to convex_hull if their intersection area is larger than threshold(0.1 * min(cell1.area, cell2.area))
    
    """
    
    import shapely
    
    ALL_CELLS = []
    
    if Para.rRNAseg:
        
        cells = Parallel_sopa_geometrize(Para.output)
        cells1 = gpd.GeoDataFrame({"geometry":cells})
        
        # Buffer 2 pixel for edge handle
        cells1['geometry'] = cells1['geometry'].apply(lambda x: x.buffer(distance = 2))
        cells1.loc[:,'model'] = 'rRNA'
        ALL_CELLS.append(cells1)
        
    if Para.baysor:
        
        with open(os.path.join(Para.output, "Segmentation/Baysor/segmentation_polygons_2d.json"),'r',encoding='utf8')as fp:
            json_data = json.load(fp)

        Geometries = json_data["geometries"]
        polygon_list = []
        index = []
        for i in range(len(Geometries)):
            try:
                polygon = Polygon(Geometries[i]['coordinates'][0])
                polygon_list.append({'geometry': polygon})
                index.append(str(Geometries[i]['cell']) + ".0")
            except:
                continue
        cells2 = gpd.GeoDataFrame(polygon_list, index = index)
        cells2.loc[:,'model'] = 'baysor'
        ALL_CELLS.append(cells2)
        
    if stardist_roi:
            
        rois = np.load(os.path.join(Para.output, 'Segmentation/StarDist/RoiSet.npy'))
        polygon_list = []
        for i in range(len(rois)):
            polygon = Polygon(np.stack([rois[i,1,:], rois[i,0,:]], axis=1))
            polygon_list.append({'geometry': polygon})
            
        cells3 = gpd.GeoDataFrame(polygon_list)
        cells3['geometry'] = cells3['geometry'].apply(lambda x: x.buffer(distance = 5 / scaler))
        cells3.loc[:,'model'] = 'DAPI'
        ALL_CELLS.append(cells3)
        
    if len(ALL_CELLS) == 0:
        return 'No_Segmentation'
        
    cells_gpd = pd.concat(ALL_CELLS)
    cells_gpd = cells_gpd.reset_index(drop = True)
    
    cells = cells_gpd['geometry'].to_list()
    n_cells = len(cells)
    keep_indices = [True] * n_cells
    
    tree = shapely.STRtree(cells)
    conflicts = tree.query(cells, predicate="intersects")
    conflicts = conflicts[:, conflicts[0] != conflicts[1]].T
    
    
    for i1, i2 in tqdm(conflicts, desc="Resolving segmentation conflicts"):
        
        cell1 = cells_gpd.loc[i1, 'geometry']
        cell2 = cells_gpd.loc[i2, 'geometry']
        cell1_model = cells_gpd.loc[i1, 'model']
        cell2_model = cells_gpd.loc[i2, 'model']
        
        if cell1_model == 'rRNA':
            
            if cell2_model == 'rRNA':
                pass
            
            elif cell2_model == 'baysor' or cell2_model == 'DAPI':
                
                if cell2.is_valid:
                    intersection = cell1.intersection(cell2).area

                    if intersection >= 0.3 * cell2.area:
                        keep_indices[i2] = False

                    elif intersection >= 0.7 * cell1.area:
                        keep_indices[i1] = False

                    else:
                        cells_gpd.loc[i2, 'geometry'] = shapely.convex_hull(cell2.difference(cell1))
                else:
                    keep_indices[i2] = False
        
        elif cell1_model == 'DAPI':
            
            if cell2_model == 'rRNA':
                intersection = cell1.intersection(cell2).area
                
                if intersection >= 0.3 * cell1.area:
                    keep_indices[i1] = False
                    
                elif intersection >= 0.7 * cell2.area:
                    keep_indices[i2] = False
                    
                else:
                    cells_gpd.loc[i1, 'geometry'] = shapely.convex_hull(cell1.difference(cell2))
            
            elif cell2_model == 'DAPI':
                
                cells_gpd.loc[i1, 'geometry'] = shapely.convex_hull(cell1.difference(cell2))
                cells_gpd.loc[i2, 'geometry'] = shapely.convex_hull(cell2.difference(cell1))
            
            elif cell2_model == 'baysor':
                
                if cell.is_valid:
                    
                    intersection = cell1.intersection(cell2).area

                    if intersection >= 0.3 * cell2.area:
                        keep_indices[i2] = False

                    elif intersection >= 0.7 * cell1.area:
                        keep_indices[i1] = False

                    else:
                        cells_gpd.loc[i2, 'geometry'] = shapely.convex_hull(cell2.difference(cell1))
                        
                else:
                    
                    keep_indices[i2] = False
                
        elif cell1_model == 'baysor':
            
            if cell2.is_valid:
            
                if cell2_model == 'rRNA' or cell2_model == 'DAPI':

                    intersection = cell1.intersection(cell2).area

                    if intersection >= 0.3 * cell1.area:
                        keep_indices[i1] = False

                    elif intersection >= 0.7 * cell2.area:
                        keep_indices[i2] = False

                    else:
                        cells_gpd.loc[i1, 'geometry'] = shapely.convex_hull(cell1.difference(cell2))

                elif cell2_model == 'baysor':
                    
                    if cell2.is_valid:
                        
                        pass

                    else:
                        keep_indices[i2] = False
                        
            else:
                        
                keep_indices[i1] = False
                
                
    cells_gpd = cells_gpd.loc[keep_indices,['geometry']].reset_index(drop = True)
    
    # 简单再过滤一遍去掉非Polygon的cell
    
    keep_indices = [True] * len(cells_gpd)
    
    for i,p in enumerate(cells_gpd['geometry']):
        if p.geom_type != 'Polygon':
            keep_indices[i] = False
            
    cells_gpd = cells_gpd.loc[keep_indices,['geometry']].reset_index(drop = True)
    
    return cells_gpd
    
    
def mark_low_confidence_point_dist(coord, scaler = 0.325):
    
    threshold = 10 / scaler
    
    tmp_coord = coord.copy()
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = os.cpu_count() - 1)
    
    def filter_dist(group, threshold = 30):
        tree = KDTree(group[['x', 'y']])
        nearest_pairs = tree.query_pairs(threshold)
        tmp_idx = set()
        for i1,i2 in nearest_pairs:
            tmp_idx.add(i1)
            tmp_idx.add(i2)
        return group.iloc[list(tmp_idx),:]
    
    return tmp_coord.groupby('gene').parallel_apply(lambda group: filter_dist(group, threshold)).reset_index(drop = True)
    
        
    
    
    