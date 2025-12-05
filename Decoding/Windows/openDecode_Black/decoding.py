"""
Author: Li Xinyang
Last modified: 2025.9.27


Change log:
        
    2025.9.26
        New script
        
    2025.9.27
        Keep writing
        
"""

import logging

from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import KDTree
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


log = logging.getLogger(__name__)


class UnionFind:
    
    def __init__(self):
        self.parent = {}
    
    def find(self, x):

        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        

        while self.parent[x] != root:
            next_node = self.parent[x]
            self.parent[x] = root
            x = next_node
        
        return root
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX
            
    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x



class Decoding():
    
    def __init__(self, para, dist_threshold: float = 1.0, prob_threshold: float = 0.3, prob_diff_threshold: float = 0.6, qv_ratio_threshold: float = 0.9):
        
        self.para = para
        self.dist_threshold = dist_threshold
        self.prob_threshold = prob_threshold
        self.prob_diff_threshold = prob_diff_threshold
        self.qv_ratio_threshold = qv_ratio_threshold
        
        
    def _all_single_decode(self, rd: str):
        
        channel = self.para.CHANNEL_INFO[rd].copy()
        channel.remove(self.para.ANCHOR_CHANNEL)
        
        for ch in channel:
            
            spots = pd.read_parquet(os.path.join(self.para.OUTPUT_PATH, f"Registration/stitched/{rd}/{rd}_{ch}.parquet"))
            
            spots['gene'] == self.para.CODEBOOK[rd][ch][ch]
            
            spots['qv'] = spots['intensity']
            
            return spots
            
            
        
    def _get_feature(self, rd: str) -> dict:
        
        """
        Get feature for one cycle
        
        channel: channel with only coding channel
        
        """
        
        channel = self.para.CHANNEL_INFO[rd].copy()
        channel.remove(self.para.ANCHOR_CHANNEL)
    
        spots_dict = {}
        trees_dict = {}
        
        for ch in channel:
            
            spots = pd.read_parquet(os.path.join(self.para.OUTPUT_PATH, f"Registration/stitched/{rd}/{rd}_{ch}.parquet"))
            spots_dict[ch] = spots.round({'x':1, 'y':1, 'intensity':2, 'threshold': 1}).copy()
            trees_dict[ch] = KDTree(spots[['x', 'y']])

        potenital_points = pd.concat(spots_dict.values())
        potenital_points = potenital_points.drop_duplicates(subset=['x','y'])

        features = []
        metas = []
        
        metas.append(potenital_points['x'].to_numpy())
        metas.append(potenital_points['y'].to_numpy())

        for ch in channel:
            
            dist, idx = trees_dict[ch].query(potenital_points[['x', 'y']])
            # ch_intensity
            feature = spots_dict[ch].loc[idx, 'intensity'].to_numpy()
            features.append(feature)
            feature = spots_dict[ch].loc[idx, 'threshold'].to_numpy()
            features.append(feature)
            # ch_distance
            features.append(dist)
            # ch_angle
            dx = spots_dict[ch].loc[idx,'x'].to_numpy() - potenital_points.loc[:, 'x'].to_numpy()
            dy = spots_dict[ch].loc[idx,'y'].to_numpy() - potenital_points.loc[:, 'y'].to_numpy()

            feature = np.arctan2(dx,dy)
            features.append(feature)

            metas.append(np.array([f"{i}_{ch}" for i in idx]))



        X = np.array(features, dtype = np.float16).T
        meta = np.array(metas).T
        
        return X, meta
     
    
    def _get_training_one_spot(self, x:np.ndarray, channel: [str], codebook: dict):
        
        """
        Get high fidelity results for random forest model training for one spot
        
        channel: channel with only coding channel
        
        dist_threshold: distance between spots to take into consideration
        """
        
        coding_channel_len = len(channel)
        
        dist_mask = x[[4 * j + 2 for j in range(coding_channel_len)]] < self.dist_threshold
        valid_idxs = np.where(dist_mask)[0]

        if len(valid_idxs) == 0:
            return 'Ambiguous'

        intensities = x[[4 * j for j in valid_idxs]]

        if len(valid_idxs) == 1:
            key = channel[valid_idxs[0]]
            return codebook.get(key, {}).get(key, 'Ambiguous')
        
        sorted_order = np.argsort(intensities)
        top1_idx = sorted_order[-1]
        top2_idx = sorted_order[-2]
        anchor_int = intensities[top1_idx]

        if anchor_int - intensities[top2_idx] < self.prob_threshold and anchor_int > 0.50 and intensities[top2_idx] > 0.50:
            
            within_thresh = anchor_int - intensities >= 0
            within_thresh &= (anchor_int - intensities) < self.prob_threshold
            if np.sum(within_thresh) != 2:
                return 'Ambiguous'

            code1 = channel[valid_idxs[top1_idx]]
            code2 = channel[valid_idxs[top2_idx]]

            if channel.index(code1) > channel.index(code2):
                code1, code2 = code2, code1

            return codebook.get(code1, {}).get(code2, 'Ambiguous')
        else:
            return 'Ambiguous'

        
    def _get_training_set(self, X: np.ndarray, rd: str) -> np.ndarray:
        
        y = []
        
        channel = self.para.CHANNEL_INFO[rd].copy()
        channel.remove(self.para.ANCHOR_CHANNEL)
        
        codebook = self.para.CODEBOOK[rd]
        
        log.info('Preparing training set...')
        for x in tqdm(X, dynamic_ncols = True):
            
            g = self._get_training_one_spot(x, channel, codebook)
            y.append(g)

        assert len(y) == len(X)
        
        return np.array(y)
            
        
    def _train_random_forest(self, X: np.ndarray, y: list) -> RandomForestClassifier:
        
        clf = RandomForestClassifier(n_jobs = self.para.THREADS)
        
        clf.fit(X,y)
        
        return clf
    
    def _create_reverse_mapping(self, codebook):
        reverse_map = {}
        for outer_key, inner_dict in codebook.items():
            for inner_key, gene in inner_dict.items():
                reverse_map[gene] = (outer_key, inner_key)
        return reverse_map
        
        
    def _predict_all_spots(self, clf: RandomForestClassifier, X: np.ndarray, meta: np.ndarray, rd: str) -> pd.core.frame.DataFrame:
        
        channel = self.para.CHANNEL_INFO[rd].copy()
        channel.remove(self.para.ANCHOR_CHANNEL)
        
        codebook = self.para.CODEBOOK[rd]
        
        gene_to_channels = self._create_reverse_mapping(codebook)
        channel_to_idx = {ch: i + 2 for i, ch in enumerate(channel)}
        
        if len(codebook) == 1:
            
            valid_genes = clf.predict(X)
            valid_indices = np.arange(len(valid_genes))
            prob_diffs = np.mean(X[:,[0,4]], axis = 1)
            
        else:
        
            df_prob = clf.predict_proba(X)

            tmp = np.sort(df_prob, axis = 1)

            prob_array = np.array(df_prob)
            meta_array = np.array(meta)

            sorted_indices = np.argsort(prob_array, axis=1)
            first_indices = sorted_indices[:, -1]
            second_indices = sorted_indices[:, -2]
            first_probs = prob_array[np.arange(len(prob_array)), first_indices]
            second_probs = prob_array[np.arange(len(prob_array)), second_indices]
            prob_diffs = first_probs - second_probs

            valid_mask = prob_diffs > self.prob_diff_threshold
            valid_indices = np.where(valid_mask)[0]

            valid_genes = clf.classes_[first_indices[valid_indices]]
        
        meta_array = np.array(meta)
        meta_valid = meta_array[valid_indices]

        ch1_list = []
        ch2_list = []
        for gene in valid_genes:
            ch1, ch2 = gene_to_channels[gene]
            ch1_list.append(channel_to_idx[ch1])
            ch2_list.append(channel_to_idx[ch2])

        
        intermediate_df = pd.DataFrame({
            'x': meta_valid[:, 0],
            'y': meta_valid[:, 1],
            'gene': valid_genes,
            'ch1': meta_valid[np.arange(len(valid_indices)), ch1_list],
            'ch2': meta_valid[np.arange(len(valid_indices)), ch2_list],
            'qv': prob_diffs[valid_indices]
        })
        
        return intermediate_df
    
    
    def _remove_multi_used_spots(self, intermediate_df: pd.core.frame.DataFrame, qv_ratio_threshold: float = 0.9):
        
        indices = intermediate_df[['ch1', 'ch2']].to_numpy()
        uf = UnionFind()

        for sublist in indices:

            first_elem = sublist[0]
            uf.add(first_elem)
            for elem in sublist[1:]:
                uf.add(elem)
                uf.union(first_elem, elem)

        label_map = {}
        label_count = 0
        result = []

        for sublist in indices:

            root = uf.find(sublist[0])
            if root not in label_map:
                label_map[root] = label_count
                label_count += 1
            result.append(label_map[root])

        intermediate_df["label"] = result
        
        grouped = intermediate_df.groupby('label')
        keep_indices = []
        
        log.info('Removing multi used spots...')
        for label, group in tqdm(grouped, dynamic_ncols = True):
            if len(group) == 1:

                keep_indices.extend(group.index.tolist())
            else:

                sorted_idx = group['qv'].sort_values(ascending=False).index
                top_idx = sorted_idx[0]
                top_qv = group['qv'][sorted_idx[0]]
                top_gene = group.loc[top_idx, 'gene']

                conflict_found = False
                for idx in sorted_idx[1:]:
                    if group.loc[idx, 'qv'] > self.qv_ratio_threshold * top_qv and group.loc[idx, 'gene'] != top_gene:
                        conflict_found = True
                        break

                if not conflict_found:
                    keep_indices.append(top_idx)


        return intermediate_df.loc[keep_indices].reset_index(drop=True)
    
    
    def decoding(self, force = False):
        
        all_decoded_df = []
        
        for rd in self.para.CYCLES:
            
            log.info(f'Decoding {rd}...')
            
            PROGRESS_DICT = self.para._parse_progress_yaml()
            
            if not PROGRESS_DICT['decoding'][rd] or force:
            
                if all(
                        len(inner_dict) == 1 and ch in inner_dict
                        for ch, inner_dict in self.para.CODEBOOK[rd].items()
                    ):

                    final_df = self._all_single_decode(rd)
                    final_df.to_parquet(os.path.join(self.para.OUTPUT_PATH, f'tmp/{rd}_decoded.parquet'))

                else:


                    X, meta = self._get_feature(rd)

                    y = self._get_training_set(X, rd)

                    mask_index = (y != 'Ambiguous')

                    X_t = X[mask_index, :].copy()
                    y = y[mask_index]
                    
                    # meta_t = meta[mask_index,:].copy()

                    log.info(f'Training Random Forest Classifier...')
                    clf = self._train_random_forest(X_t, y)

                    log.info(f'Predicting spots...')
                    intermediate_df = self._predict_all_spots(clf, X, meta, rd)

                    final_df = self._remove_multi_used_spots(intermediate_df)

                    final_df.to_parquet(os.path.join(self.para.OUTPUT_PATH, f'tmp/{rd}_decoded.parquet'))
                
                PROGRESS_DICT['decoding'][rd] = True
                
                self.para.save_progress_yaml(PROGRESS_DICT)
                
            else:
                
                final_df = pd.read_parquet(os.path.join(self.para.OUTPUT_PATH, f'tmp/{rd}_decoded.parquet'))
            
            all_decoded_df.append(final_df)
            
            
        Final_df = pd.concat(all_decoded_df, axis = 0).reset_index(drop = True)
        Final_df['qv'] = Final_df['qv'] * 40
        Final_df.to_parquet(os.path.join(self.para.OUTPUT_PATH, f'tmp/Decoded_all.parquet'))
        
        return None
            
                        
                        
                    
                    
            