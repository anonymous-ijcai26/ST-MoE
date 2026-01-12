import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC, prepare_dynamic_subgraphs_all, coalesce_sparse, add_self_loops_sparse, StateEx
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
from einops import rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
import numpy as np
class SubnetInteraction(nn.Module):
    def __init__(self, hidden_dim, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # 用 batch_first 更直观
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, M, return_attn=False):

        B, E, T, H = M.shape
        out_list = []
        attn_list = []

        for t in range(T):
            x = M[:, :, t, :]                 # (B,E,H)
            y, attn_w = self.attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )
            x = self.norm1(x + self.drop(y))
            y2 = self.ffn(x)
            x = self.norm2(x + self.drop(y2))

            out_list.append(x)        # (B,E,H)
            attn_list.append(attn_w)  # (B,nhead,E,E)

        M_out = torch.stack(out_list, dim=2)            # (B,E,T,H)
        if return_attn:
            attn_all = torch.stack(attn_list, dim=1)    # (B,T,nhead,E,E)
            return M_out, attn_all
        return M_out
class Percentile(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, q):
        k = int(x.numel() * q / 100.0)
        if k <= 0:
            return x.min()
        return x.view(-1).kthvalue(k).values


def flatten_upper_triangle_batch(a):
    """
    a: (B, N, N) 对称矩阵，取上三角（不含对角）展开成向量
    返回: (B, N*(N-1)/2)
    """
    B, N, _ = a.shape

    triu_idx = torch.triu_indices(N, N, offset=1, device=a.device)
    out = a[:, triu_idx[0], triu_idx[1]]  
    return out
class LayerTemporal(nn.Module):
    """
    一个时间 expert：对序列每个时间点做同一个 MLP
    x: (B*E, T, H) -> (B*E, T, H)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (BE, T, H)
        BE, T, H = x.shape
        x2 = x.reshape(BE*T, H)
        y2 = self.mlp(x2)
        return y2.reshape(BE, T, H)
class LayerTemporal_MoE(nn.Module):

    def __init__(self, hidden_dim, num_experts, topk=1):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk


        self.gate = nn.Linear(hidden_dim, num_experts)


        self.temporal_experts = nn.ModuleList([
            LayerTemporal(hidden_dim)
            for _ in range(num_experts)
        ])

    def gating(self, x):

        ctx = x.mean(dim=1)                
        logits = self.gate(ctx)            
        probs = F.softmax(logits, dim=-1)  # (BE, K)

        values, indexes = torch.topk(probs, k=self.topk, dim=-1)  # (BE, topk)
        zeros = torch.zeros_like(probs)
        gates = zeros.scatter(-1, indexes, values)                # (BE, K)
        return gates, probs

    def forward(self, M):
        """
        M: (B, E, T, H)
        """
        B, E, T, H = M.shape
        x = rearrange(M, 'b e t h -> (b e) t h')  # (BE, T, H)

        gates, probs = self.gating(x)            # (BE, K)

 
        expert_outs = []
        for expert in self.temporal_experts:
            expert_outs.append(expert(x))       # (BE, T, H)

        expert_outs = torch.stack(expert_outs, dim=-2)  # (BE, K, T, H)

        gates_expanded = gates.unsqueeze(-1).unsqueeze(-1)  # (BE, K, 1, 1)
        x_out = torch.sum(expert_outs * gates_expanded, dim=-2)  # (BE, T, H)

        M_out = rearrange(x_out, '(b e) t h -> b e t h', b=B, e=E)
        return M_out


class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, epsilon=True):
        super().__init__()
        if epsilon:
            self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))
        else:
            self.epsilon = 0.0
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, v, a):
        # v: (btn, c)  a: sparse (btn, btn)
        v_agg = torch.sparse.mm(a, v)
        v_agg += (1 + self.epsilon) * v
        return self.mlp(v_agg)

class LayerGIN_MoE(nn.Module):
    def __init__(self, num_features, hidden_dim, num_experts,
                 ):
        super().__init__()
        self.num_experts = num_experts
       
        self.gate = nn.Linear(num_features, num_experts)

        self.gin_experts = nn.ModuleList([
            LayerGIN(num_features, hidden_dim)
            for _ in range(num_experts)
        ])
    def gating(self, x):
        logits = self.gate(x)       
        probs = F.softmax(logits, dim=-1)   
        values, indexes = torch.topk(probs, k=1, dim=-1)   
        zeros = torch.zeros_like(probs)                   
        gates = zeros.scatter(-1, indexes, values)     
        return gates, probs   

    def forward(self, v, a, b, t, n):

        x = rearrange(v, '(b t n) c -> b t n c', b=b, t=t, n=n)
        gates, probs = self.gating(x)  # gates: (b,t,n,e)
        expert_outs = []
        for expert in self.gin_experts:
            expert_outs.append(expert(v, a))      # (btn,c)
        expert_outs = torch.stack(expert_outs, dim=-2)  # (btn, e, c)
        gates_flat = rearrange(gates, 'b t n e -> (b t n) e')  # (btn, e)
        out = torch.sum(expert_outs * gates_flat.unsqueeze(-1), dim=-2)  # (btn,c)

        return out   

class STMoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_experts, node_rearranged_len, drop_ratio):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_ratio = drop_ratio
        self.node_rearranged_len = node_rearranged_len
        self.num_subnets = len(node_rearranged_len) - 1


        self.spatial = LayerGIN_MoE(in_dim, hidden_dim, num_experts)

 
        self.temporal = LayerTemporal_MoE(hidden_dim, num_experts, topk=1)
      
        self.subnet_interaction = SubnetInteraction(
            hidden_dim=hidden_dim,
            nhead=4,
            dropout=drop_ratio
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def subnet_pool_over_nodes(self, v_bt):
        """
        v_bt: (B, T, N, H)
        return M: (B, E, T, H)
        """
        B, T, N, H = v_bt.shape
        subnet_feats = []
        for s in range(self.num_subnets):
            st = self.node_rearranged_len[s]
            ed = self.node_rearranged_len[s+1]
            v_sub = v_bt[:, :, st:ed, :]     # (B,T,n_s,H)
            v_sub = v_sub.mean(dim=2)       # pool nodes -> (B,T,H)
            subnet_feats.append(v_sub)
        M = torch.stack(subnet_feats, dim=1)  # (B,E,T,H)
        return M

    def inject_back_to_nodes(self, v_bt, M_moe):

        B, T, N, H = v_bt.shape
        v_new = v_bt

        for s in range(self.num_subnets):
            st = self.node_rearranged_len[s]
            ed = self.node_rearranged_len[s+1]

            # M_moe[:, s] shape: (B,T,H)
            msg = M_moe[:, s, :, :]                 # (B,T,H)
            msg = msg.unsqueeze(2)                  # (B,T,1,H) -> broadcast to nodes
            v_new[:, :, st:ed, :] = v_new[:, :, st:ed, :] + self.alpha * msg

        return v_new

    def forward(self, v_flat, a_sparse, B, T, N):
 

        v_flat = self.spatial(v_flat, a_sparse, B, T, N)     
        v_flat = F.dropout(F.relu(v_flat), self.drop_ratio, training=self.training)

        v_bt = rearrange(v_flat, '(b t n) h -> b t n h', b=B, t=T, n=N) # 8*8*200*128 

  
        M = self.subnet_pool_over_nodes(v_bt)               


        M_moe = self.temporal(M)                            
        M_inter, attn_subnet = self.subnet_interaction(M_moe, return_attn=True)         

        v_bt = self.inject_back_to_nodes(v_bt, M_inter)       

        v_flat = rearrange(v_bt, 'b t n h -> (b t n) h')
        return v_flat, M_moe
class ModularityMoEGIN(nn.Module):
    def __init__(self, gin_type, num_features, gnn_hidden, num_layers, sparsity,
                 drop_ratio, graph_pooling,
                 num_experts=8, node_rearranged_len=None,
                 fc_hidden=256, out_dim=2):
        super().__init__()
        self.sparsity = sparsity
        self.drop_ratio = drop_ratio
        self.num_layers = num_layers
        self.percentile = Percentile()

        assert node_rearranged_len is not None
        self.node_rearranged_len = node_rearranged_len
        self.num_subnets = len(node_rearranged_len) - 1


        if graph_pooling == 'sum':
            self.pool = lambda x: x.sum(-2)
        elif graph_pooling == 'mean':
            self.pool = lambda x: x.mean(-2)
        elif graph_pooling == 'max':
            self.pool = lambda x: x.max(-2)[0]
        else:
            raise ValueError


        self.st_layers = nn.ModuleList()
        in_dim = num_features
        for _ in range(num_layers):
            self.st_layers.append(
                STMoELayer(
                    in_dim=in_dim,
                    hidden_dim=gnn_hidden,
                    num_experts=num_experts,
                    node_rearranged_len=node_rearranged_len,
                    drop_ratio=drop_ratio
                )
            )
            in_dim = gnn_hidden

        self.num_nodes = node_rearranged_len[-1]
        tri_dim = int(self.num_nodes * (self.num_nodes - 1) / 2)
        tri_dim = self.num_nodes * (self.num_nodes - 1) // 2


        self.tri_proj = nn.Sequential(
            nn.Linear(tri_dim, gnn_hidden),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
        )


        self.fuse_gate = nn.Linear(gnn_hidden * 2, gnn_hidden)
        self.fuse_out  = nn.Linear(gnn_hidden * 2, gnn_hidden)

        self.bn_tri = nn.BatchNorm1d(tri_dim)

    def _collate_adjacency(self, a, sparsity):
        i_list, v_list = [], []
        B, T, N, _ = a.shape
        for sample in range(B):
            for timepoint in range(T):
                _a = a[sample, timepoint]
                thr = self.percentile(_a, 100 - sparsity)
                mask = (_a > thr)
                _i = mask.nonzero(as_tuple=False)
                _v = torch.ones(len(_i), device=a.device)
                _i += sample * T * N + timepoint * N
                i_list.append(_i); v_list.append(_v)
        _i = torch.cat(i_list).T
        _v = torch.cat(v_list)
        size = (B*T*N, B*T*N)
        return torch.sparse.FloatTensor(_i, _v, size)

    def forward(self, a, v):
        """
        a: (B,T,N,N)
        v: (B,T,N,C)
        """
        B, T, N, C = v.shape
        v_flat = rearrange(v, 'b t n c -> (b t n) c')
        a_sparse = self._collate_adjacency(a, self.sparsity)

        M_last = None
        for st in self.st_layers:
            v_flat, M_last = st(v_flat, a_sparse, B, T, N)  # (B,E,T,H)
        
        B_, E, T_, H = M_last.shape

        tri_list = []
        for t_i in range(T):
            x_tri = flatten_upper_triangle_batch(a[:, t_i])  # (B, tri_dim)
            x_tri = self.bn_tri(x_tri)                       
            tri_list.append(x_tri)
        tri_feat = torch.stack(tri_list, dim=1)              # (B, T, tri_dim)

        # ---- 2) tri 压到 H 维（共享）----
        tri_emb = self.tri_proj(rearrange(tri_feat, 'b t d -> (b t) d'))  # (B*T, H)
        tri_emb = rearrange(tri_emb, '(b t) h -> b t h', b=B, t=T)        # (B, T, H)

        # ---- 3) 广播到每个子网 token，并做门控融合 ----
        tri_emb = tri_emb.unsqueeze(1).expand(-1, E, -1, -1)              # (B, E, T, H)

        z = torch.cat([M_last, tri_emb], dim=-1)                          # (B, E, T, 2H)
        gate = torch.sigmoid(self.fuse_gate(z))                           # (B, E, T, H)
        msg  = self.fuse_out(z)                                           # (B, E, T, H)
        fused = M_last + gate * msg                                       # (B, E, T, H)

        token_seq = rearrange(fused, 'b e t h -> b (e t) h')               # (B, E*T, H)

        return token_seq
        # return logits

class STMoE(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz 
        self.pos_encoding = config.model.pos_encoding
        self.gnn_layers = nn.ModuleList()
        gin_type = config.model.gin_type
        num_layers = config.model.num_gin_layers
        num_experts = config.model.num_gin_experts  
        num_features =config.model.num_features
        gnn_hidden = config.model.gin_hidden

        self.node_rearranged_len = [0, 23, 59, 82, 101, 127, 144, 169, 200]
        sizes = config.model.sizes 
        sizes[0] = config.dataset.node_sz 
        do_pooling = config.model.pooling

        self.m_ex = ModularityMoEGIN(gin_type=config.model.gin_type,
                                num_features=200,
                                gnn_hidden=config.model.gin_hidden,
                                fc_hidden=config.model.fc_hidden,
                                num_layers=config.model.num_gin_layers,
                                sparsity=config.model.sparsity,
                                drop_ratio=config.model.dropout,
                                graph_pooling=config.model.graph_pooling,
                                node_rearranged_len=self.node_rearranged_len,
                                num_experts=config.model.num_gin_experts)
        self.s_ex = StateEx(hidden_dim=config.model.fc_hidden,
                            num_states=config.model.num_states, 
                            orthogonal=config.model.orthogonal,
                            freeze_center=config.model.freeze_center, 
                            project_assignment=config.model.project_assignment)
        if num_layers>0:
            if gin_type == 'gin':
                self.gnn_layers.append(LayerGIN(num_features, gnn_hidden))
            elif gin_type == 'moe_gin':
                self.gnn_layers.append(LayerGIN_MoE(num_features, gnn_hidden, num_experts))

            for _ in range(0, num_layers - 1):
                if gin_type == 'gin':
                    self.gnn_layers.append(LayerGIN(gnn_hidden, gnn_hidden))
                elif gin_type == 'moe_gin':
                    self.gnn_layers.append(LayerGIN_MoE(gnn_hidden, gnn_hidden, num_experts))

        self.predict = nn.Sequential(
            nn.Linear(config.model.fc_hidden, config.model.fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.model.fc_hidden, config.model.fc_hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((config.model.fc_hidden//2), 2),
        )


    def forward(self,
                time_seires: torch.tensor,
                d_final_pearson: torch.tensor,
                node_feature: torch.tensor):
        bz, _, _, = node_feature.shape  
        node_feature = node_feature.cuda()
        a = d_final_pearson.cuda()  # (B,T,N,N)
        v = a
        x = self.m_ex(a=a, v=v)      # (B,2)
        state_repr, state_assignments = self.s_ex(x) #x变成了8*8*128的形式 8个样本 8个窗口
        logits = self.predict(state_repr) # state_repr成了8*6*128
        return logits


