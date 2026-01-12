import os
import torch
from contextlib import suppress

@torch.no_grad()
def sliding_window_corr_cpu(
    data_np,                 # numpy: (B,N,T)
    num_windows: int,
    stride: int,
    window_size: int | None = None,
    eps: float = 1e-6,
    zero_diag: bool = True,
    chunk_b: int = 32,      
    set_threads: int | None = None,  
):

    if set_threads is not None:
        with suppress(Exception):
            torch.set_num_threads(set_threads)
            torch.set_num_interop_threads(max(1, set_threads // 2))
        os.environ["OMP_NUM_THREADS"] = str(set_threads)
        os.environ["MKL_NUM_THREADS"] = str(set_threads)


    x = torch.as_tensor(data_np, device="cpu", dtype=torch.float32)  # (B,N,T)
    B, N, T = x.shape
    if T==100:
        stride=10

    if window_size is None:
        window_size = T - (num_windows - 1) * stride
    assert window_size > 0 and T >= window_size
    W = ((T - window_size) // stride) + 1
    assert W == num_windows, f"{W=} != {num_windows=}"


    out = torch.empty((B, W, N, N), device="cpu", dtype=torch.float32)


    windows = x.unfold(2, window_size, stride)

    for w in range(W):
        Xw = windows[:, :, w, :]          
        L = Xw.shape[-1]
        for s in range(0, B, chunk_b):
            e = min(s + chunk_b, B)
            Xc = Xw[s:e]                  


            mean = Xc.mean(dim=-1, keepdim=True)
            std  = Xc.std(dim=-1, unbiased=True, keepdim=True).clamp_min(eps)
            Xc = (Xc - mean) / std


            corr = torch.bmm(Xc, Xc.transpose(1, 2)) / (L - 1)

            if zero_diag:
                idx = torch.arange(N)
                corr[:, idx, idx] = 0

            out[s:e, w] = corr
            del Xc, corr

    return out  