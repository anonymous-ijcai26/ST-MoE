import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict
from .utils import sliding_window_corr_cpu
import pickle
from .prepare_dynamic_subgraphs import prepare_dynamic_subgraphs_all

def get_comm_index():
    with open('xxx/dataset/data/DICE_CPAC200_&_Yeo-7-liberal_res-1x1x1.pkl', 'rb') as handle:
        node_clus_map = pickle.load(handle)
        print(type(node_clus_map))
    return list(node_clus_map.keys())  # 200个脑区分别属于哪个社团
def rearrange_node(node_feature, rearranged_indices):

    node_feature_rearranged = node_feature[rearranged_indices, :]
    node_feature_rearranged = node_feature_rearranged[:, rearranged_indices]
    return node_feature_rearranged
def load_abide_data(cfg: DictConfig):
    
    data = np.load(cfg.dataset.path, allow_pickle=True).item()

    final_timeseires = data["signals"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']
    final_timeseires = np.array(final_timeseires, dtype=np.float32)  # 确保是浮动类型

    d_final_pearson = sliding_window_corr_cpu(
        data_np=final_timeseires,   
        num_windows=8,
        stride=15,
        window_size=None,       
        chunk_b=32,                 
        set_threads=None            
    )
    final_pearson = np.array(final_pearson, dtype=np.float32)  # 确保是浮动类型
    d_final_pearson = np.array(d_final_pearson, dtype=np.float32)  # 确保是浮动类型
    labels = np.array(labels, dtype=np.int64)  # 如果标签是整数，使用 int64
    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)
    comm_index = get_comm_index()
    B, C, H, W = d_final_pearson.shape
    
    for b in range(B):
        for c in range(C):
            node = d_final_pearson[b, c, :, :]
            rearranged_nodes = rearrange_node(node, comm_index)           
            
            d_final_pearson[b, c, :, :] = rearranged_nodes
    final_timeseires, final_pearson, d_final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, d_final_pearson, labels)]
   

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
    
    
    
    
    num_per_class = 8

    idx0_all = (labels == 0).nonzero(as_tuple=True)[0]
    idx1_all = (labels == 1).nonzero(as_tuple=True)[0]

    if len(idx0_all) < num_per_class or len(idx1_all) < num_per_class:
        raise ValueError(
            f"样本不足：class0={len(idx0_all)}, class1={len(idx1_all)}"
        )

    perm0 = torch.randperm(len(idx0_all))[:num_per_class]
    perm1 = torch.randperm(len(idx1_all))[:num_per_class]

    idx0 = idx0_all[perm0]
    idx1 = idx1_all[perm1]

    batch_0 = {
        "timeseries": final_timeseires[idx0],   # (8, N, T)
        "dynamic_corr": d_final_pearson[idx0],  # (8, C, N, N)
        "static_corr": final_pearson[idx0],     # (8, N, N)
        "labels": labels[idx0],                 # (8,)
        "site": [site[i] for i in idx0.tolist()]
    }

    batch_1 = {
        "timeseries": final_timeseires[idx1],   # (8, N, T)
        "dynamic_corr": d_final_pearson[idx1],  # (8, C, N, N)
        "static_corr": final_pearson[idx1],     # (8, N, N)
        "labels": labels[idx1],                 # (8,)
        "site": [site[i] for i in idx1.tolist()]
    }


    return final_timeseires, d_final_pearson, final_pearson, labels, site


