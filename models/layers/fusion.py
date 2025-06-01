from math import ceil

import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, reduce

class GraphFusion(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
    
        
    def forward(self, x):
        # print(x,shape)
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]
        
        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        # attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        
        cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        cross_attn_pathways = cross_attn_pathways.softmax(dim=-1)
        
        out_pathways = cross_attn_histology @ v[:, :, :self.num_pathways]
        out_histology = cross_attn_pathways @ v[:, :, self.num_pathways:]
        cross_token = torch.cat((out_pathways, out_histology), dim=2)
        cross_token = rearrange(cross_token, 'b h n d -> b n (h d)', h = h)
        #print(cross_token.shape)
        
        return cross_token
class AlignFusion(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim=512, num_pathways=6):
    
        super().__init__()
    
        self.num_pathways = num_pathways
        
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_pathway_to_histology = Attention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_histology_to_pathway = Attention(embedding_dim, num_heads)
        self.norm4 = nn.LayerNorm(embedding_dim)
        
        self.final_cross_attn = Attention(embedding_dim, num_heads)
        self.final_norm = nn.LayerNorm(embedding_dim)
    def forward(self, token):
        """
        1.self-attention for gene
        2.cross-attention Q:gene K,V:wsi
        3.MLP for gene
        4.cross-attention Q:wsi  K,V:gene
        """
        # Align Block
        
        # Self attention block
        keys_p = token[:,self.num_pathways:,:]
        queries_p = token[:,:self.num_pathways,:]
        # print(queries_p)
        queries = queries_p + self.self_attn(queries_p, queries_p, queries_p)
        # print(queries)
        queries = self.norm1(queries)
        
        keys=keys_p
        attn_out = self.cross_attn_pathway_to_histology(q=queries, k=keys, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        
        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        attn_out = self.cross_attn_histology_to_pathway(q=keys, k=queries, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        # 1.add align result to orignal features
        # 2.finally update the gene features 
        k = keys  + keys_p
        q = queries + queries_p
        q = q+self.final_cross_attn(q=q,k=k,v=k)
        q = self.final_norm(q)

        return torch.cat((q, k),dim=1)
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x