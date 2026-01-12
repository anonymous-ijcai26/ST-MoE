import torch

@torch.no_grad()
def dense_batch_topk_edges(A_bn: torch.Tensor, k: int, sym: bool = True):

    B, N, _ = A_bn.shape
    A = A_bn.clone()
    idx = torch.arange(N, device=A.device)
    A[:, idx, idx] = float("-inf")
    vals, idx_topk = torch.topk(A, k=k, dim=-1)
    rows = torch.arange(N, device=A.device).view(1, N, 1).expand(B, N, k).reshape(-1)
    cols = idx_topk.reshape(-1)
    w = vals.reshape(-1)
    offsets = (torch.arange(B, device=A.device) * N).view(B,1,1).expand(B,N,k).reshape(-1)
    rows, cols = rows + offsets, cols + offsets
    if sym:
        rows = torch.cat([rows, cols], dim=0)
        cols = torch.cat([cols, rows[:rows.numel()//2]], dim=0)
        w = torch.cat([w, w[:w.numel()//2]], dim=0)
    ei = torch.stack([rows, cols], dim=0)
    sp = torch.sparse_coo_tensor(ei, w, size=(B*N, B*N)).coalesce()
    return sp.indices(), sp.values()

@torch.no_grad()
def fast_subgraph(edge_index, edge_weight, mask):

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
    k_topk: int = 20,
    sym: bool = True
):

    B, C, N, _ = d_final_pearson.shape
    device = d_final_pearson.device
    data = {}


    intervals = [(node_rearranged_len[i], node_rearranged_len[i+1]) for i in range(len(node_rearranged_len)-1)]
    sizes = [e - s for s, e in intervals]
    S = len(intervals)
    base_mask = torch.zeros(S, N, dtype=torch.bool, device=device)
    for i, (s, e) in enumerate(intervals):
        base_mask[i, s:e] = True
    sub_masks = base_mask.unsqueeze(1).expand(S, B, N).reshape(S, B*N)

    for c in range(C):
        A_bn = d_final_pearson[:, c]                   # (B,N,N)
        edge_index, edge_weight = dense_batch_topk_edges(A_bn, k=k_topk, sym=sym)

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
