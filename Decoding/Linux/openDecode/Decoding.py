import numpy as np
import torch
import itertools
import pandas as pd
import random
from tqdm import tqdm
import pyro
from pyro.distributions import *
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
import os

def Decoding(Spots_Dict, Para):
    
    if Para.method_type == "10N":
        
        Decode_Dict = []
        
        for rd,channels in Para.Round_channel.items():
            
            info = f"Decoding {rd}..."
            print(info)
            
            barcodes_01 = Para.CodeBooks_01[rd]
            gene_names = np.concatenate([Para.CodeBooks_genes[rd], np.array(["background", "infeasible", "nan"])])
            
            # OUTPUT:
            # 'class_probs': posterior probabilities computed via e-step
            # 'class_ind': indices of different barcode classes (genes / background / infeasible / nan)
            #  class_ind = {'genes': np.arange(K), 'bkg': bkg_ind, 'inf': inf_ind_s, 'nan': nan_class_ind}
            # 'params': estimated model parameters
            # 'norm_const': constants used for normalization of spots prior to model fitting
                
            out = decoding_function(Spots_Dict[rd]['intensity'], barcodes_01, print_training_progress=True)
            middle_df = _make_dataframe(out, Spots_Dict[rd], gene_names, barcodes_01)
            middle_df = _move_multiused_spots(middle_df)
            Decode_Dict.append(middle_df)
            
        Final_df = pd.concat(Decode_Dict, axis = 0).reset_index(drop = True)
        Final_df.to_parquet(os.path.join(Para.output, "tmp/Decoded_raw.parquet"))
        Final_df = Final_df[~Final_df["gene"].isin(["background", "infeasible", "nan"])].reset_index(drop = True)
        Final_df = Final_df[Final_df['probs'] >= 0.7].reset_index(drop = True)
        Final_df.to_parquet(os.path.join(Para.output, "Decoded_filtered.parquet"))
            
    elif Para.method_type == "MultiCycle":
        
        info = f"Decoding..."
        print(info)
        
        barcodes_01 = Para.CodeBooks_01
        gene_names = np.concatenate([Para.CodeBooks_genes, np.array(["background", "infeasible", "nan"])])
        
        out = decoding_function(Spots_Dict['intensity'], barcodes_01, print_training_progress=True)
        Final_df = _make_dataframe(out, Spots_Dict, gene_names, barcodes_01)
        Final_df.to_parquet(os.path.join(Para.output, "tmp/Decoded_raw.parquet"))
        Final_df = Final_df[~Final_df["gene"].isin(["background", "infeasible", "nan"])].reset_index(drop = True)
    
        Final_df = _move_multiused_spots(Final_df)
        Final_df = Final_df[Final_df['probs'] >= 0.7].reset_index(drop = True)
        Final_df.to_parquet(os.path.join(Para.output, "Decoded_filtered.parquet"))
        
    return Final_df
    
    
    
def _make_dataframe(out, Spots_Dict, gene_names, barcodes_01):
    
    spots = Spots_Dict.copy()
    
    _,ii,jj = spots["index"].shape
    
    max_index = 0
    for i in range(ii):
        for j in range(jj):
            spots['index'][:,i,j] += max_index
            max_index = spots['index'][:,i,j].max()
    
    val = out['class_probs'].max(axis=1)
    ind = out['class_probs'].argmax(axis=1)
    genes = [gene_names[i] for i in ind]
    
    used_indices = []
    for i,idx in enumerate(ind):
        if np.isin(gene_names[idx], ["background", "infeasible", "nan"]):
            used_indices.append([])
        else:
            used_indices.append(list(spots['index'][i][np.where(barcodes_01[idx] == 1)]))
            
    
    decode_df = pd.DataFrame({
        "gene": genes,
        "probs": val,
        "indices": used_indices,
        "x": Spots_Dict['location'][:,0],
        "y": Spots_Dict['location'][:,1],
    })
    
    return decode_df

# def _label_filter(group):
    
#     import pandas as pd
#     import numpy as np
    
#     proba_list = np.array(group["probs"])
    
#     if len(proba_list) == 1:
#         return group
    
#     gene_list = np.array(group["gene"])
#     init_gene = gene_list[0]
#     for i,g in enumerate(gene_list[1:]):
#         if g != init_gene and proba_list[i+1] < 0.7:
#             return group.iloc[[0]]
#         else:
#             continue
    
#     return pd.DataFrame(columns=group.columns)


# class UnionFind:
#     def __init__(self):
#         self.parent = {}
    
#     def find(self, x):
#         if self.parent[x] != x:
#             self.parent[x] = self.find(self.parent[x])
#         return self.parent[x]
    
#     def union(self, x, y):
#         rootX = self.find(x)
#         rootY = self.find(y)
#         if rootX != rootY:
#             self.parent[rootY] = rootX
            
#     def add(self, x):
#         if x not in self.parent:
#             self.parent[x] = x

class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        # 首先找到根节点
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        
        # 路径压缩：将路径上的所有节点直接指向根
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
            

def _move_multiused_spots(df):
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = os.cpu_count() - 1)
    
    info = f"Removing multi used spots..."
    print(info)
    
    indices = df["indices"].to_list()
    uf = UnionFind()
    
    # 将每个子列表中的元素加入并查集
    for sublist in indices:
        if not sublist:  # 处理空子列表
            continue
        first_elem = sublist[0]
        uf.add(first_elem)
        for elem in sublist[1:]:
            uf.add(elem)
            uf.union(first_elem, elem)
    
    # 为每个子列表找到其根元素，并赋予标签
    label_map = {}
    label_count = 0
    result = []

    for sublist in indices:
        if not sublist:  # 处理空子列表
            result.append(-1)  # 使用 -1 表示空子列表
            continue
        root = uf.find(sublist[0])
        if root not in label_map:
            label_map[root] = label_count
            label_count += 1
        result.append(label_map[root])
    
    df["label"] = result
    
    # def _filter_group(group):
    #     if len(group) == 1:
    #         return group
    #     else:
    #         # 按probs降序排序
    #         sorted_group = group.sort_values('probs', ascending=False)
    #         # 检查第一名是否是第二名的1.357倍
    #         if sorted_group.iloc[0]['probs'] >= 1.357 * sorted_group.iloc[1]['probs']:
    #             return sorted_group.head(1)
    #         else:
    #             return pd.DataFrame()  # 返回空DataFrame表示不保留任何行
    
    df = df.loc[df.groupby('label')['probs'].idxmax()]
    # df = df.groupby('label', group_keys=False).parallel_apply(_label_filter)
    
    # df = df.groupby('label').parallel_apply(_filter_group).reset_index(drop=True)
    
    df = df.reset_index(drop = True)
    
    return df


# https://github.com/gerstung-lab/postcode/blob/master/source-code/postcode/decoding_functions.py

def torch_format(numpy_array):
    D = numpy_array.shape[1] * numpy_array.shape[2]
    return torch.tensor(numpy_array).float().transpose(1, 2).reshape(numpy_array.shape[0], D)


def barcodes_01_from_channels(barcodes_1234, C, R):
    K = barcodes_1234.shape[0]
    barcodes_01 = np.ones((K, C, R))
    for b in range(K):
        barcodes_01[b, :, :] = 1 * np.transpose(barcodes_1234[b, :].reshape(R, 1) == np.arange(1, C + 1))
    return barcodes_01


def kronecker_product(tr, tc):
    tr_height, tr_width = tr.size()
    tc_height, tc_width = tc.size()
    out_height = tr_height * tc_height
    out_width = tr_width * tc_width
    tiled_tc = tc.repeat(tr_height, tr_width)
    expanded_tr = (tr.unsqueeze(2).unsqueeze(3).repeat(1, tc_height, tc_width, 1).view(out_height, out_width))
    return expanded_tr * tiled_tc


def chol_sigma_from_vec(sigma_vec, D):
    L = torch.zeros(D, D)
    L[torch.tril(torch.ones(D, D)) == 1] = sigma_vec
    return torch.mm(L, torch.t(L))


def e_step(data, w, theta, sigma, N, K, print_training_progress):
    class_probs = torch.ones(N, K)
    if print_training_progress:
        for k in tqdm(range(K)):
            dist = MultivariateNormal(theta[k], sigma)
            class_probs[:, k] = w[k] * torch.exp(dist.log_prob(data))
    else:
        for k in range(K):
            dist = MultivariateNormal(theta[k], sigma)
            class_probs[:, k] = w[k] * torch.exp(dist.log_prob(data))

    class_prob_norm = class_probs.div(torch.sum(class_probs, dim=1, keepdim=True))
    # class_prob_norm[torch.isnan(class_prob_norm)] = 0
    return class_prob_norm


def mat_sqrt(A, D):
    try:
        U, S, V = torch.svd(A + 1e-3 * A.mean() * torch.rand(D, D))
    except:
        U, S, V = torch.svd(A + 1e-2 * A.mean() * torch.rand(D, D))
    S_sqrt = torch.sqrt(S)
    return torch.mm(torch.mm(U, torch.diag(S_sqrt)), V.t())

@config_enumerate
def model_constrained_tensor(data, N, D, C, R, K, codes, batch_size=None):
    w = pyro.param('weights', torch.ones(K) / K, constraint=constraints.simplex)

    # using tensor sigma
    sigma_ch_v = pyro.param('sigma_ch_v', torch.eye(C)[np.tril_indices(C, 0)])
    sigma_ch = chol_sigma_from_vec(sigma_ch_v, C)
    sigma_ro_v = pyro.param('sigma_ro_v', torch.eye(D)[np.tril_indices(R, 0)])
    sigma_ro = chol_sigma_from_vec(sigma_ro_v, R)
    sigma = kronecker_product(sigma_ro, sigma_ch)

    # codes_tr_v = pyro.param('codes_tr_v', 3 * torch.ones(1, D), constraint=constraints.positive)
    codes_tr_v = pyro.param('codes_tr_v', 3 * torch.ones(1, D), constraint=constraints.greater_than(1.))
    codes_tr_consts_v = pyro.param('codes_tr_consts_v', -1 * torch.ones(1, D))

    theta = torch.matmul(codes * codes_tr_v + codes_tr_consts_v, mat_sqrt(sigma, D))

    with pyro.plate('data', N, batch_size) as batch:
        z = pyro.sample('z', Categorical(w))
        pyro.sample('obs', MultivariateNormal(theta[z], sigma), obs=data[batch])
        
auto_guide_constrained_tensor = AutoDelta(poutine.block(model_constrained_tensor, expose=['weights', 'codes_tr_v', 'codes_tr_consts_v', 'sigma_ch_v', 'sigma_ro_v']))


def train(svi, num_iterations, data, N, D, C, R, K, codes, print_training_progress, batch_size):
    pyro.clear_param_store()
    losses = []
    if print_training_progress:
        for j in tqdm(range(num_iterations)):
            loss = svi.step(data, N, D, C, R, K, codes, batch_size)
            losses.append(loss)
    else:
        for j in range(num_iterations):
            loss = svi.step(data, N, D, C, R, K, codes, batch_size)
            losses.append(loss)
    return losses


# input - output decoding function
def decoding_function(spots, barcodes_01,
                      num_iter=60, batch_size=4000, up_prc_to_remove=99.95,
                      modify_bkg_prior=True, # should be False when there is a lot of background signal (eg pixel-wise decoding, a lot of noisy boundary tiles)
                      estimate_bkg=True, estimate_additional_barcodes=None, # controls adding additional barcodes during parameter estimation
                      add_remaining_barcodes_prior=0.05, # after model is estimated, infeasible barcodes are used in the e-step with given prior
                      print_training_progress=True, set_seed=1):
    # INPUT:
    # spots: a numpy array of dim N x C x R (number of spots x coding channels x rounds);
    # barcodes_01: a numpy array of dim K x C x R (number of barcodes x coding channels x rounds)
    # OUTPUT:
    # 'class_probs': posterior probabilities computed via e-step
    # 'class_ind': indices of different barcode classes (genes / background / infeasible / nan)
    # 'params': estimated model parameters
    # 'norm_const': constants used for normalization of spots prior to model fitting

    # if cuda available, runs on gpu
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
            
    N = spots.shape[0]
    if N == 0:
        print('There are no spots to decode.')
        return
    C = spots.shape[1]
    R = spots.shape[2]
    K = barcodes_01.shape[0]
    D = C * R
    data = torch_format(spots)
    codes = torch_format(barcodes_01)

    # include background / any additional barcode in codebook
    if estimate_bkg:
        bkg_ind = codes.shape[0]
        codes = torch.cat((codes, torch.zeros(1, D)))
    else:
        bkg_ind = np.empty((0,), dtype=np.int32)
    if np.any(estimate_additional_barcodes is not None):
        inf_ind = codes.shape[0] + np.arange(estimate_additional_barcodes.shape[0])
        codes = torch.cat((codes, torch_format(estimate_additional_barcodes)))
    else:
        inf_ind = np.empty((0,), dtype=np.int32)

    # normalize spot values
    ind_keep = np.where(np.sum(data.cpu().numpy() < np.percentile(data.cpu().numpy(), up_prc_to_remove, axis=0), axis=1) == D)[0] if up_prc_to_remove < 100 else np.arange(0, N)
    s = torch.tensor(np.percentile(data[ind_keep, :].cpu().numpy(), 60, axis=0))
    max_s = torch.tensor(np.percentile(data[ind_keep, :].cpu().numpy(), 99.9, axis=0))
    min_s = torch.min(data[ind_keep, :], dim=0).values
    log_add = (s ** 2 - max_s * min_s) / (max_s + min_s - 2 * s)
    log_add = torch.max(-torch.min(data[ind_keep, :], dim=0).values + 1e-10, other=log_add.float())
    data_log = torch.log10(data + log_add)
    data_log_mean = data_log[ind_keep, :].mean(dim=0, keepdim=True)
    data_log_std = data_log[ind_keep, :].std(dim=0, keepdim=True)
    data_norm = (data_log - data_log_mean) / data_log_std  # column-wise normalization
    
    # model training:
    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(model_constrained_tensor, auto_guide_constrained_tensor, optim, loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)
    losses = train(svi, num_iter, data_norm[ind_keep, :], len(ind_keep), D, C, R, codes.shape[0], codes, print_training_progress, min(len(ind_keep), batch_size))
    # collect estimated parameters
    w_star = pyro.param('weights').detach()
    sigma_ch_v_star = pyro.param('sigma_ch_v').detach()
    sigma_ro_v_star = pyro.param('sigma_ro_v').detach()
    sigma_ro_star = chol_sigma_from_vec(sigma_ro_v_star, R)
    sigma_ch_star = chol_sigma_from_vec(sigma_ch_v_star, C)
    sigma_star = kronecker_product(sigma_ro_star, sigma_ch_star)
    codes_tr_v_star = pyro.param('codes_tr_v').detach()
    codes_tr_consts_v_star = pyro.param('codes_tr_consts_v').detach()
    theta_star = torch.matmul(codes * codes_tr_v_star + codes_tr_consts_v_star, mat_sqrt(sigma_star, D))

    # computing class probabilities with appropriate prior probabilities
    if modify_bkg_prior and w_star.shape[0] > K:
        # making sure that the K barcode classes have higher prior in case there are more than K classes
        w_star_mod = torch.cat((w_star[0:K], w_star[0:K].min().repeat(w_star.shape[0] - K)))
        w_star_mod = w_star_mod / w_star_mod.sum()
    else:
        w_star_mod = w_star

    if add_remaining_barcodes_prior > 0:
        barcodes_1234 = np.array([p for p in itertools.product(np.arange(1, C + 1), repeat=R)])  # all possible barcodes
        codes_inf = np.array(torch_format(barcodes_01_from_channels(barcodes_1234, C, R)).cpu())  # all possible barcodes in the same format as codes
        codes_inf = np.concatenate((np.zeros((1, D)), codes_inf))  # add the bkg code at the beginning
        codes_cpu = codes.cpu()
        for b in range(codes_cpu.shape[0]):  # remove already existing codes
            r = np.array(codes_cpu[b, :], dtype=np.int32)
            if np.where(np.all(codes_inf == r, axis=1))[0].shape[0]!=0:
                i = np.reshape(np.where(np.all(codes_inf == r, axis=1)), (1,))[0]
                codes_inf = np.delete(codes_inf, i, axis=0)
        if not estimate_bkg:
            bkg_ind = codes_cpu.shape[0]
            inf_ind = np.append(inf_ind, codes_cpu.shape[0] + 1 + np.arange(codes_inf.shape[0]))
        else:
            inf_ind = np.append(inf_ind, codes_cpu.shape[0] + np.arange(codes_inf.shape[0]))
        codes_inf = torch.tensor(codes_inf).float()
        alpha = (1 - add_remaining_barcodes_prior)
        w_star_all = torch.cat((alpha * w_star_mod, torch.tensor((1 - alpha) / codes_inf.shape[0]).repeat(codes_inf.shape[0])))
        class_probs_star = e_step(data_norm, w_star_all,
                                  torch.matmul(torch.cat((codes, codes_inf)) * codes_tr_v_star + codes_tr_consts_v_star.repeat(w_star_all.shape[0], 1), mat_sqrt(sigma_star, D)), sigma_star, N, w_star_all.shape[0],
                                  print_training_progress)
    else:
        class_probs_star = e_step(data_norm, w_star_mod, theta_star, sigma_star, N, codes.shape[0], print_training_progress)

    # collapsing added barcodes
    class_probs_star_s = torch.cat((torch.cat((class_probs_star[:, 0:K], class_probs_star[:, bkg_ind].reshape((N, 1))), dim=1), torch.sum(class_probs_star[:, inf_ind], dim=1).reshape((N, 1))), dim=1)
    inf_ind_s = inf_ind[0]
    # adding another class if there are NaNs
    nan_spot_ind = torch.unique((torch.isnan(class_probs_star_s)).nonzero(as_tuple=False)[:, 0])
    if nan_spot_ind.shape[0] > 0:
        nan_class_ind = class_probs_star_s.shape[1]
        class_probs_star_s = torch.cat((class_probs_star_s, torch.zeros((class_probs_star_s.shape[0], 1))), dim=1)
        class_probs_star_s[nan_spot_ind, :] = 0
        class_probs_star_s[nan_spot_ind, nan_class_ind] = 1
    else:
        nan_class_ind = np.empty((0,), dtype=np.int32)
        
    class_probs = class_probs_star_s.cpu().numpy()
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.FloatTensor")

    class_ind = {'genes': np.arange(K), 'bkg': bkg_ind, 'inf': inf_ind_s, 'nan': nan_class_ind}
    torch_params = {'w_star': w_star_mod.cpu(), 'sigma_star': sigma_star.cpu(),
                    'sigma_ro_star': sigma_ro_star.cpu(), 'sigma_ch_star': sigma_ch_star.cpu(),
                    'theta_star': theta_star.cpu(), 'codes_tr_consts_v_star': codes_tr_consts_v_star.cpu(),
                    'codes_tr_v_star': codes_tr_v_star.cpu(), 'losses': losses}
    norm_const = {'log_add': log_add, 'data_log_mean': data_log_mean, 'data_log_std': data_log_std}

    return {'class_probs': class_probs, 'class_ind': class_ind, 'params': torch_params, 'norm_const': norm_const}