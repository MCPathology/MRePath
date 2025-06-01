#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains layers used in AllSet and all other tested methods.
"""

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional

import torch_sparse

# This part is for PMA.
# Modified from GATConv in pyg.
# Method for initialization
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def sparse_diagonal(diag, shape):
    r,c = shape
    assert r == c
    indexes = torch.arange(r).to(diag.device)
    indexes = torch.stack([indexes, indexes], dim=0)
    return torch.sparse.FloatTensor(indexes, diag)
    
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def generate_indices_general(indexes, d):
    d_range = torch.arange(d)
    d_range_edges = d_range.repeat(d).view(-1,1) #0,1..d,0,1..d..   d*d elems
    d_range_nodes = d_range.repeat_interleave(d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
    indexes = indexes.unsqueeze(1) 

    large_indexes_0 = d * indexes[0] + d_range_nodes
    large_indexes_0 = large_indexes_0.permute((1,0)).reshape(1,-1)
    large_indexes_1 = d * indexes[1] + d_range_edges
    large_indexes_1 = large_indexes_1.permute((1,0)).reshape(1,-1)
    large_indexes = torch.concat((large_indexes_0, large_indexes_1), 0)

    return large_indexes

def normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, d, norm_type='degree_norm'):
    #this will return either D^-1/B^-1 or D^(-1/2)/B^-1
    if norm_type == 'degree_norm':
        #return D_inv and B_inv used to normalised the laplacian (/propagation)
        #normalise using node/hyperedge degrees D_e and D_v in the paper
        D = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    elif norm_type == 'sym_degree_norm':
        #normalise using node/hyperedge degrees D_e and D_v in the paper
        D = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
        D = D ** (-0.5)
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    elif norm_type == 'block_norm':
        #normalise using diag(HHT) and deg_e <- this take into account the values predicted in H as oposed to 0/1 as in the degree
        # this way of computing the normalisation tensor is only valid for diagonal sheaf
        D = scatter_add(alpha*alpha, hyperedge_index[0], dim=0, dim_size=num_nodes*d)
        D = 1.0 / D #can compute inverse like this because the matrix is diagonal
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B
    
    elif norm_type == 'sym_block_norm':
        #normalise using diag(HHT) and deg_e <- this take into account the values predicted in H as oposed to 0/1 as in the degree
        # this way of computing the normalisation tensor is only valid for diagonal sheaf
        D = scatter_add(alpha*alpha, hyperedge_index[0], dim=0, dim_size=num_nodes*d)
        D = D ** (-0.5) #can compute inverse like this because the matrix is diagonal
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

# One layer of Sheaf Diffusion with diagonal Laplacian Y = (I-D^-1/2LD^-1) with L normalised with B^-1
class HyperDiffusionDiagSheafConv(MessagePassing):
    r"""
    
    """
    def __init__(self, in_channels, out_channels, d, device, dropout=0, bias=True, norm_type='degree_norm', 
                left_proj=None, norm=None, residual = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm_type = norm_type
        self.left_proj = left_proj
        self.norm = norm
        self.residual = residual

        if self.left_proj:
            self.lin_left_proj = MLP(in_channels=d, 
                        hidden_channels=d,
                        out_channels=d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        
           

        self.lin = MLP(in_channels=in_channels, 
                        hidden_channels=out_channels,
                        out_channels=out_channels,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
                        
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.device = device

        self.I_mask = None
        self.Id = None

        self.reset_parameters()

    #to allow multiple runs reset all parameters used
    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()

        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                alpha, 
                num_nodes,
                num_edges) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix {Nd x F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix Nd x Md} from nodes to edges.
            alpha (Tensor, optional): restriction maps
        """ 
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1,num_nodes * self.d).t()
        x = self.lin(x)
        data_x = x

        #depending on norm_type D^-1 or D^-1/2
        D_inv, B_inv = normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, self.d, self.norm_type)


        if self.norm_type in ['sym_degree_norm', 'sym_block_norm']:
            # compute D^(-1/2) @ X
            x = D_inv.unsqueeze(-1) * x

        H = torch.sparse.FloatTensor(hyperedge_index, alpha, size=(num_nodes*self.d, num_edges*self.d))
        H_t = torch.sparse.FloatTensor(hyperedge_index.flip([0]), alpha, size=(num_edges*self.d, num_nodes*self.d))

        #this is because spdiags does not support gpu
        B_inv =  sparse_diagonal(B_inv, shape = (num_edges*self.d, num_edges*self.d))
        D_inv =  sparse_diagonal(D_inv, shape = (num_nodes*self.d, num_nodes*self.d))

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        minus_L = torch_sparse.spspmm(B_inv.indices(), B_inv.values(), H_t.indices(), H_t.values(), B_inv.shape[0],B_inv.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(H.indices(), H.values(), minus_L[0], minus_L[1], H.shape[0],H.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(D_inv.indices(), D_inv.values(), minus_L[0], minus_L[1], D_inv.shape[0],D_inv.shape[1], H_t.shape[1])
        minus_L = torch.sparse_coo_tensor(minus_L[0], minus_L[1], size=(num_nodes*self.d, num_nodes*self.d)).to(self.device)

        #negate the diagonal blocks and add eye matrix
        if num_nodes != 4096 or self.I_mask is None: #prepare these in advance
            I_mask_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            I_mask_indices = generate_indices_general(I_mask_indices, self.d)
            I_mask_values = torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse_coo_tensor(I_mask_indices, I_mask_values).to(self.device)
            self.Id = sparse_diagonal(torch.ones(num_nodes*self.d), shape = (num_nodes*self.d, num_nodes * self.d)).to(self.device)

        minus_L = minus_L.coalesce()
        #this help us changing the sign of the elements in the block diagonal
        #with an efficient lower=memory mask 
        minus_L = torch.sparse_coo_tensor(minus_L.indices(), minus_L.values(), minus_L.size())
        # print(minus_L)
        # print(self.I_mask)
        minus_L = minus_L - 2 * minus_L.mul(self.I_mask)
        minus_L = self.Id + minus_L

        minus_L = minus_L.coalesce()
        out = torch_sparse.spmm(minus_L.indices(), minus_L.values(), minus_L.shape[0], minus_L.shape[1], x)
        if self.bias is not None:
            out = out + self.bias
        if self.residual:
            out = out + data_x
        if num_nodes !=4096:
            self.I_mask = None
        return out

#  One layer of Sheaf Diffusion with orthogonal Laplacian Y = (I-D^-1/2LD^-1) with L normalised with B^-1
class HyperDiffusionOrthoSheafConv(MessagePassing):
    r"""
    
    """
    def __init__(self, in_channels, out_channels, d, device, dropout=0, bias=True, norm_type='degree_norm', 
                left_proj=None, norm=None, residual = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm_type=norm_type
        self.norm=norm

        #for ortho matrix block <=> degree
        if self.norm_type == 'block_norm':
            self.norm_type = 'degree_norm'
        elif self.norm_type == 'sym_block_norm':
            self.norm_type = 'sym_degree_norm'
        
        self.left_proj = left_proj
        self.residual = residual

        if self.left_proj:
            self.lin_left_proj = MLP(in_channels=d, 
                        hidden_channels=d,
                        out_channels=d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        self.lin = MLP(in_channels=in_channels, 
                        hidden_channels=d,
                        out_channels=out_channels,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.device = device

        self.I_mask = None
        self.Id = None
        self.reset_parameters()

    #to allow multiple runs reset all parameters used
    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Tensor, hyperedge_index: Tensor,
                alpha, 
                num_nodes,
                num_edges) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix {Nd x F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix Nd x Md} from nodes to edges.
            alpha (Tensor, optional): restriction maps
        """ 
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1,num_nodes * self.d).t()
        x = self.lin(x)    
        data_x = x

        if self.I_mask is None: #prepare these in advance
            I_mask_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            I_mask_indices = utils.generate_indices_general(I_mask_indices, self.d)
            I_mask_values = -1 * torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse.FloatTensor(I_mask_indices, I_mask_values).to(self.device)
            self.Id = utils.sparse_diagonal(torch.ones(num_nodes*self.d), shape = (num_nodes*self.d, num_nodes * self.d)).to(self.device)
        
        D_inv, B_inv = normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, self.d, norm_type=self.norm_type)

        if self.norm_type in ['sym_degree_norm', 'sym_block_norm']:
            # compute D^(-1/2) @ X
            x = D_inv.unsqueeze(-1) * x

        H = torch.sparse.FloatTensor(hyperedge_index, alpha, size=(num_nodes*self.d, num_edges*self.d))
        H_t = torch.sparse.FloatTensor(hyperedge_index.flip([0]), alpha, size=(num_edges*self.d, num_nodes*self.d))

        #these are still diagonal because of ortho
        B_inv =  utils.sparse_diagonal(B_inv, shape = (num_edges*self.d, num_edges*self.d))
        D_inv = utils.sparse_diagonal(D_inv, shape = (num_nodes*self.d, num_nodes*self.d))

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        minus_L = torch_sparse.spspmm(B_inv.indices(), B_inv.values(), H_t.indices(), H_t.values(), B_inv.shape[0],B_inv.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(H.indices(), H.values(), minus_L[0], minus_L[1], H.shape[0],H.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(D_inv.indices(), D_inv.values(), minus_L[0], minus_L[1], D_inv.shape[0],D_inv.shape[1], H_t.shape[1])
        minus_L = torch.sparse_coo_tensor(minus_L[0], minus_L[1], size=(num_nodes*self.d, num_nodes*self.d)).to(self.device)



        minus_L = minus_L * self.I_mask
        minus_L = self.Id + minus_L

        minus_L = minus_L.coalesce()
        out = torch_sparse.spmm(minus_L.indices(), minus_L.values(), minus_L.shape[0],minus_L.shape[1], x)

        if self.bias is not None:
            out = out + self.bias

        if self.residual:
            out = out + data_x
        return out

#  One layer of Sheaf Diffusion with general/lowrank Laplacian Y = (I-D^-1/2LD^-1) with L normalised with B^-1
class HyperDiffusionGeneralSheafConv(MessagePassing):
    r"""
    
    """
    def __init__(self, in_channels, out_channels, d, device, dropout=0, bias=True, norm_type='degree_norm', 
                left_proj=None, norm=None, residual=False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm=norm

        self.left_proj = left_proj
        self.residual = residual

        if self.left_proj:
            self.lin_left_proj = MLP(in_channels=d, 
                        hidden_channels=d,
                        out_channels=d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)

        self.lin = MLP(in_channels=in_channels, 
                        hidden_channels=d,
                        out_channels=out_channels,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.device = device

        self.I_mask = None
        self.Id = None

        self.norm_type = norm_type

        self.reset_parameters()

    #to allow multiple runs reset all parameters used
    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    #this is just for block and sym_block normalisation since the matrices D^-1 is a proper inverse that need to be computed
    def normalise(self, h_general_sheaf, hyperedge_index, norm_type, num_nodes, num_edges):
        #this is just for block and sym_block normalisation

        #index correspond to the small matrix
        row_small = hyperedge_index[0].view(-1,self.d,self.d)[:,0,0] // self.d
        h_general_sheaf_1 = h_general_sheaf.reshape(row_small.shape[0], self.d, self.d)

        to_be_inv_nodes = torch.bmm(h_general_sheaf_1, h_general_sheaf_1.permute(0,2,1)) 
        to_be_inv_nodes = scatter_add(to_be_inv_nodes, row_small, dim=0, dim_size=num_nodes)

      
        if norm_type in ['block_norm']:
            d_inv_nodes = utils.batched_sym_matrix_pow(to_be_inv_nodes, -1.0) #n_nodes x d x d   
            return d_inv_nodes

        elif norm_type in ['sym_block_norm']:
            d_sqrt_inv_nodes = utils.batched_sym_matrix_pow(to_be_inv_nodes, -0.5) #n_nodes x d x d 
            return d_sqrt_inv_nodes

       

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                alpha, 
                num_nodes,
                num_edges) -> Tensor:
        r"""
        Args:
            Args:
            x (Tensor): Node feature matrix {Nd x F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix Nd x Md} from nodes to edges.
            alpha (Tensor, optional): restriction maps
        """ 
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1,num_nodes * self.d).t()

        x = self.lin(x)
        data_x = x

        if self.I_mask is None or num_nodes != self.I_mask.shape[0]: #prepare these in advance
            # I_block = torch.block_diag(*[torch.ones((self.d, self.d)) for i in range(num_nodes)]).to(self.device)
            I_mask_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            I_mask_indices = generate_indices_general(I_mask_indices, self.d)
            I_mask_values = -1 * torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse.FloatTensor(I_mask_indices, I_mask_values).to(self.device)
            self.Id = sparse_diagonal(torch.ones(num_nodes*self.d), shape = (num_nodes*self.d, num_nodes * self.d)).to(self.device)

        if self.norm_type in ['block_norm', 'sym_block_norm']:
            # NOTE: the normalisation is specific to general sheaf
            #D_e is the same as before
            B_inv_flat = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*self.d)
            B_inv_flat = 1.0 / B_inv_flat
            B_inv_flat[B_inv_flat == float("inf")] = 0
            B_inv =  utils.sparse_diagonal(B_inv_flat, shape = (num_edges*self.d, num_edges*self.d))

            #D_v is a dxd matrix than needs to be inverted
            D_inv = self.normalise(alpha, hyperedge_index, self.norm_type, num_nodes, num_edges) # num_nodes x d x d
            diag_indices_D = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            D_inv_indices = utils.generate_indices_general(diag_indices_D, self.d).to(x.device)
            D_inv_flat = D_inv.reshape(-1)
            D_inv = torch.sparse.FloatTensor(D_inv_indices, D_inv_flat)

        else:
            D_inv, B_inv = normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, self.d, norm_type=self.norm_type)
        
        # compute D^(-1/2) @ X for the sym case
        # x: (num_nodes*d) x f
        if self.norm_type  == 'sym_degree_norm':
            x = D_inv.unsqueeze(-1) * x
        elif self.norm_type  == 'sym_block_norm':
            D_inv = D_inv.coalesce()
            x = torch_sparse.spmm(D_inv.indices(), D_inv.values(), D_inv.shape[0],D_inv.shape[1], x)

        if self.norm_type in ['sym_degree_norm', 'degree_norm']:
            #these are still diagonal because of ortho
            B_inv =  sparse_diagonal(B_inv, shape = (num_edges*self.d, num_edges*self.d))
            D_inv = sparse_diagonal(D_inv, shape = (num_nodes*self.d, num_nodes*self.d))

        H = torch.sparse.FloatTensor(hyperedge_index, alpha, size=(num_nodes*self.d, num_edges*self.d))
        H_t = torch.sparse.FloatTensor(hyperedge_index.flip([0]), alpha, size=(num_edges*self.d, num_nodes*self.d))

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()


        minus_L = torch_sparse.spspmm(B_inv.indices(), B_inv.values(), H_t.indices(), H_t.values(), B_inv.shape[0],B_inv.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(H.indices(), H.values(), minus_L[0], minus_L[1], H.shape[0],H.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(D_inv.indices(), D_inv.values(), minus_L[0], minus_L[1], D_inv.shape[0],D_inv.shape[1], H_t.shape[1])
        minus_L = torch.sparse_coo_tensor(minus_L[0], minus_L[1], size=(num_nodes*self.d, num_nodes*self.d)).to(self.device)
        minus_L = minus_L * self.I_mask
        minus_L = self.Id + minus_L

        minus_L = minus_L.coalesce()
        out = torch_sparse.spmm(minus_L.indices(), minus_L.values(), minus_L.shape[0],minus_L.shape[1], x)
        
        if self.bias is not None:
            out = out + self.bias

        if self.residual:
            out = out + data_x

        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        F = self.out_channels
        out = norm_i.view(-1, 1) * x_j.view(-1, F)
        if alpha is not None:
            out = alpha.view(-1, 1) * out

        return out


class PMA(MessagePassing):
    """
        PMA part:
        Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
        i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
        In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """
    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):
        #         kwargs.setdefault('aggr', 'add')
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.
        self.aggr = 'add'
#         self.input_seed = input_seed

#         This is the encoder part. Where we use 1 layer NN (Theta*x_i in the GATConv description)
#         Now, no seed as input. Directly learn the importance weights alpha_ij.
#         self.lin_O = Linear(heads*self.hidden, self.hidden) # For heads combining
        # For neighbor nodes (source side, key)
        self.lin_K = Linear(in_channels, self.heads*self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_channels, self.heads*self.hidden)
        self.att_r = Parameter(torch.Tensor(
            1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads*self.hidden,
                       hidden_channels=self.heads*self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=.0, Normalization='None',)
        self.ln0 = nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = nn.LayerNorm(self.heads*self.hidden)
#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:

#         Always no bias! (For now)
        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
#         glorot(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
#         zeros(self.bias)

    def forward(self, x, edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.hidden

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)
#         else:
#             x_l, x_r = x[0], x[1]
#             assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
#             x_l = self.lin_l(x_l).view(-1, H, C)
#             alpha_l = (x_l * self.att_l).sum(dim=-1)
#             if x_r is not None:
#                 x_r = self.lin_r(x_r).view(-1, H, C)
#                 alpha_r = (x_r * self.att_r).sum(dim=-1)

#         assert x_l is not None
#         assert alpha_l is not None

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
#         ipdb.set_trace()
        out = self.propagate(edge_index, x=x_V,
                             alpha=alpha_r, aggr=self.aggr)

        alpha = self._alpha
        self._alpha = None

#         Note that in the original code of GMT paper, they do not use additional W^O to combine heads.
#         This is because O = softmax(QK^T)V and V = V_in*W^V. So W^O can be effectively taken care by W^V!!!
        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out+F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j,
                index, ptr,
                size_j):
        #         ipdb.set_trace()
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max()+1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.
        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.
        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
#         ipdb.set_trace()
        if aggr is None:
            raise ValeuError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.lin = Linear(in_ft, out_ft, bias=bias)
#         self.weight = Parameter(torch.Tensor(in_ft, out_ft))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_ft))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        #         x = data.x
        #         G = data.edge_index

        x = self.lin(x)
#         x = x.matmul(self.weight)
#         if self.bias is not None:
#             x = x + self.bias
        x = torch.matmul(G, x)
        return x

        
class HNHNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, nonlinear_inbetween=True,
                 concat=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HNHNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.nonlinear_inbetween = nonlinear_inbetween

        # preserve variable heads for later use (attention)
        self.heads = heads
        self.concat = True
        # self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_v2e = Linear(in_channels, hidden_channels)
        self.weight_e2v = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_v2e.reset_parameters()
        self.weight_e2v.reset_parameters()
        # glorot(self.weight_v2e)
        # glorot(self.weight_e2v)
        # zeros(self.bias)

    def forward(self, x, data):
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """
        hyperedge_index = data.edge_index
        hyperedge_weight = None
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.weight_v2e(x)

#         ipdb.set_trace()
#         x = torch.matmul(torch.diag(data.D_v_beta), x)
        x = data.D_v_beta.unsqueeze(-1) * x

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=data.D_e_beta_inv,
                             size=(num_nodes, num_edges))
        
        if self.nonlinear_inbetween:
            out = F.relu(out)
        
        # sanity check
        out = torch.squeeze(out, dim=1)
        
        out = self.weight_e2v(out)
        
#         out = torch.matmul(torch.diag(data.D_e_alpha), out)
        out = data.D_e_alpha.unsqueeze(-1) * out

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=data.D_v_alpha_inv,
                             size=(num_edges, num_nodes))
        
        return out

    def message(self, x_j, norm_i):

        out = norm_i.view(-1, 1) * x_j

        return out

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.hidden_channels, self.out_channels)



class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True, residual = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        self.residual = residual
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False)
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)


        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)


    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        num_nodes = hyperedge_index[0].max().item() + 1

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)
        data_x = x
        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # pdb.set_trace()
        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if self.residual:
            out = out + data_x
        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='ln', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        elif Normalization == 'ln':
            print("using LN")
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                # print(self.normalizations[0].device)
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class HalfNLHconv(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn',
                 InputNorm=False,
                 heads=1,
                 attention=True
                 ):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, Normalization, InputNorm)
                self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm)
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    def reset_parameters(self):

        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ is 'Identity'):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ is 'Identity'):
                self.f_dec.reset_parameters()
#         self.bn.reset_parameters()

    def forward(self, x, edge_index, norm, aggr='add'):
        """
        input -> MLP -> Prop
        """
        
        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr)
            x = F.relu(self.f_dec(x))
            
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index,
                  dim_size=None, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.
        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.
        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
#         ipdb.set_trace()
        if aggr is None:
            raise ValeuError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

