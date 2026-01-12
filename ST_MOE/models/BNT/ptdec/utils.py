import torch
import math
def coalesce_sparse(ei, ew, num_nodes):
    sp = torch.sparse_coo_tensor(ei, ew, size=(num_nodes, num_nodes))
    return sp.coalesce()

def add_self_loops_sparse(sp):
    N = sp.size(0)
    eye_idx = torch.arange(N, device=sp.device)
    ei_eye = torch.stack([eye_idx, eye_idx], dim=0)  # (2,N)
    ew_eye = torch.ones(N, device=sp.device, dtype=sp.dtype)
    ei = torch.cat([sp.indices(), ei_eye], dim=1)
    ew = torch.cat([sp.values(),  ew_eye], dim=0)
    return torch.sparse_coo_tensor(ei, ew, size=(N, N)).coalesce()

@torch.no_grad()
def dense_batch_topk_edges(
    A_bn: torch.Tensor,
    k: int | None = None,
    pct: float | None = None,
    sym: bool = True,
    per_row: bool = True,      # True=按行百分比；False=全局百分比
    use_abs: bool = False,     # True=按 |权重| 选边
    min_k: int = 1
):

    assert (k is not None) ^ (pct is not None), "k 和 pct 必须二选一"
    B, N, _ = A_bn.shape
    A = A_bn.clone()

    # 去自环
    idx = torch.arange(N, device=A.device)
    A[:, idx, idx] = float("-inf")

    # 需要的话改为按绝对值排
    A_for_top = A.abs() if use_abs else A

    if per_row:
        # ---------- 按行百分比：算出 k ----------
        if pct is not None:
            k = max(min_k, int(math.ceil(pct * (N - 1))))
            k = min(k, N - 1)
        # 每行 top-k
        vals, idx_topk = torch.topk(A_for_top, k=k, dim=-1)        # (B,N,k)
        # 取原始权重（不是绝对值）
        rows = torch.arange(N, device=A.device).view(1, N, 1).expand(B, N, k).reshape(-1)
        cols = idx_topk.reshape(-1)
        w    = A.gather(dim=-1, index=idx_topk).reshape(-1)        # 用原矩阵的值

        # 批偏移
        offsets = (torch.arange(B, device=A.device) * N).view(B,1,1).expand(B,N,k).reshape(-1)
        rows, cols = rows + offsets, cols + offsets

    else:
        # ---------- 全局百分比：按批内所有边的阈值 ----------
        # 只取上三角/下三角的一半来避免重复（这里取上三角，不含对角）
        triu_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=A.device), diagonal=1)
        # (B, N*N) → 过滤为上三角
        A_lin = A_for_top.reshape(B, -1)                                     # 权重（用于找阈值）
        mask_lin = triu_mask.reshape(-1).expand(B, -1)                        # 同形掩码
        vals_all = A_lin[mask_lin]                                           # (B * N*(N-1)/2,)

        if pct is None:
            raise ValueError("全局模式必须给 pct")
        keep_num = max(1, int(math.ceil(vals_all.numel() * pct)))
        thr = torch.topk(vals_all, k=keep_num, largest=True, sorted=False).values.min()

        # ≥ 阈值的上三角边全部保留
        keep_triu = (A_for_top >= thr) & triu_mask                           # (B,N,N)

        # 取出 (i,j) 与 (j,i) 的坐标与权重
        b_idx, r_idx, c_idx = keep_triu.nonzero(as_tuple=True)               # 三元组索引
        rows = (r_idx + b_idx * N).to(torch.long)
        cols = (c_idx + b_idx * N).to(torch.long)
        w    = A[b_idx, r_idx, c_idx]                                       # 用原矩阵的值

        if sym:
            # 镜像一份 (j,i)
            rows = torch.cat([rows, (c_idx + b_idx * N)], dim=0)
            cols = torch.cat([cols, (r_idx + b_idx * N)], dim=0)
            w    = torch.cat([w,    A[b_idx, c_idx, r_idx]], dim=0)

    # coalesce 去重/合并
    ei = torch.stack([rows, cols], dim=0)
    sp = torch.sparse_coo_tensor(ei, w, size=(B * N, B * N)).coalesce()
    return sp.indices(), sp.values()

@torch.no_grad()
def fast_subgraph(edge_index, edge_weight, mask):
    """快速筛边 + 重标号"""
    keep = mask[edge_index[0]] & mask[edge_index[1]]
    ei, ew = edge_index[:, keep], edge_weight[keep]
    old_ids = mask.nonzero(as_tuple=False).flatten()
    new_id = -torch.ones(mask.size(0), dtype=torch.long, device=mask.device)
    new_id[old_ids] = torch.arange(old_ids.numel(), device=mask.device)
    return new_id[ei], ew, old_ids

@torch.no_grad()
def make_subgraph_batch(B, n_sub, device):
    return torch.arange(B, device=device).repeat_interleave(n_sub)

@torch.no_grad()
def prepare_dynamic_subgraphs_all(
    d_final_pearson: torch.Tensor,
    node_rearranged_len: list[int],
    pct: int = 0.20,
    sym: bool = True
):
    """
    一次性生成所有时间窗/子网的子图。
    输入:
      d_final_pearson: (B, C, N, N)
      node_rearranged_len: [0,23,59,...]
    返回:
      data_dict[(c, net_id)] = {edge_index, edge_weight, batch, orig_idx, ...}
    """
    B, C, N, _ = d_final_pearson.shape
    device = d_final_pearson.device
    data = {}

    # 预计算子网掩码 (S, B*N)
    intervals = [(node_rearranged_len[i], node_rearranged_len[i+1]) for i in range(len(node_rearranged_len)-1)]
    sizes = [e - s for s, e in intervals]
    S = len(intervals)
    base_mask = torch.zeros(S, N, dtype=torch.bool, device=device)
    for i, (s, e) in enumerate(intervals):
        base_mask[i, s:e] = True
    sub_masks = base_mask.unsqueeze(1).expand(S, B, N).reshape(S, B*N)

    # 主循环
    for c in range(C):
        A_bn = d_final_pearson[:, c]                   # (B,N,N)
        edge_index, edge_weight = dense_batch_topk_edges(
            A_bn, pct=pct, per_row=True, sym=True, use_abs=False
        )

        for net_id, n_sub in enumerate(sizes):
            mask = sub_masks[net_id]
            sub_ei, sub_ew, orig_idx = fast_subgraph(edge_index, edge_weight, mask)
            sub_batch = make_subgraph_batch(B=B, n_sub=n_sub, device=device)
            data[(c, net_id)] = dict(
                edge_index=sub_ei,
                edge_weight=sub_ew,
                batch=sub_batch,
                orig_idx=orig_idx,
                num_nodes_sub=n_sub,
                B=B, N=N
            )
    return data
