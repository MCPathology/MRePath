from models.layers.layers import MLP
import torch.nn as nn
import torch
from torch_scatter import scatter, scatter_mean, scatter_add
import utils

import torch.nn.functional as F
import numpy as np


# helper functions to predict sigma(MLP(x_v || h_e)) varying how thw attributes for hyperedge are computed
def predict_blocks(x, e, hyperedge_index, sheaf_lin, args):
    # e_j = avg(x_v)
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    return h_sheaf

def predict_blocks_var2(x, hyperedge_index, sheaf_lin, args):
    # e_j = avg(h_v)
    row, col = hyperedge_index
    e = scatter_mean(x[row],col, dim=0)
    
    xs = torch.index_select(x, dim=0, index=row)
    es= torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    
    return h_sheaf

def predict_blocks_var3(x, hyperedge_index, sheaf_lin, sheaf_lin2, args):
    # universal approx according to  Equivariant Hypergraph Diffusion Neural Operators
    # # e_j = sum(φ(x_v))

    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    #φ(x_v)
    x_e = sheaf_lin2(x)
    #sum(φ(x_v)
    e = scatter_add(x_e[row],col, dim=0)  
    es= torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse v ersion of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if args.sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif args.sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    
    return h_sheaf

def predict_blocks_cp_decomp(x, hyperedge_index, cp_W, cp_V, sheaf_lin,sheaf_act='sigmoid'):
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    xs_ones = torch.cat((xs, torch.ones(xs.shape[0],1).to(xs.device)), dim=-1) #nnz x f+1
    xs_ones_proj = torch.tanh(cp_W(xs_ones)) #nnz x r
    xs_prod =  scatter(xs_ones_proj, col, dim=0, reduce="mul") #edges x r
    e = torch.relu(cp_V(xs_prod))#edges x f
    e = e + torch.relu(scatter_add(x[row],col, dim=0))
    es= torch.index_select(e, dim=0, index=col)

    # sigma(MLP(x_v || h_e))
    h_sheaf = torch.cat((xs,es), dim=-1) #sparse version of an NxEx2f tensor
    h_sheaf = sheaf_lin(h_sheaf)  #sparse version of an NxExd tensor
    if sheaf_act == 'sigmoid':
        h_sheaf = F.sigmoid(h_sheaf) # output d numbers for every entry in the incidence matrix
    elif sheaf_act == 'tanh':
        h_sheaf = F.tanh(h_sheaf) # output d numbers for every entry in the incidence matrix
    return h_sheaf

#One class for each type of sheaf. We will need to merge them.

# Build the restriction maps for the Diagonal Case
class SheafBuilderDiag(nn.Module):
    def __init__(self, sheaf_pred_block='cp_decomp', sheaf_dropout=True, sheaf_special_head=False, heads=1, MLP_hidden=256, dropout=0.0, AllSet_input_norm=True):
        super(SheafBuilderDiag, self).__init__()
        self.prediction_type = sheaf_pred_block # pick the way hyperedge feartures are computed
        self.sheaf_dropout = sheaf_dropout
        self.special_head = sheaf_special_head # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = heads # stalk dinension
        self.MLP_hidden = MLP_hidden
        self.norm = AllSet_input_norm
        self.dropout = dropout
        
        self.sheaf_lin = MLP(
                    in_channels=2*self.MLP_hidden, 
                    hidden_channels=self.MLP_hidden,
                    out_channels=self.d,
                    num_layers=1,
                    dropout=0.0,
                    Normalization='ln',
                    InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=self.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        elif self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(
                        in_channels=self.MLP_hidden+1, 
                        hidden_channels=self.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
            self.cp_V = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=self.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.MLP_hidden)
                        

    def reset_parameters(self):
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.sheaf_lin.reset_parameters()
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        

    # this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        #predict (_ x d) elements
        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin)
        
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        
        if self.special_head:
            new_head_mask = [1]*(self.d-1) + [0]
            new_head = [0]*(self.d-1) + [1]
            h_sheaf = h_sheaf * torch.tensor(new_head_mask, device=x.device) + torch.tensor(new_head, device=x.device)
        
        self.h_sheaf = h_sheaf #this is stored in self for testing purpose
        h_sheaf_attributes = h_sheaf.reshape(-1) #(d*K)

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # We need to modify indices from the NxE matrix 
        # to correspond to the large Nd x Ed matrix, but restrict only on the element of the diagonal of each block
        # indices: scalar [i,j] -> block dxd with indices [d*i, d*i+1.. d*i+d-1; d*j, d*j+1 .. d*j+d-1]
        # attributes: reshape h_sheaf

        d_range = torch.arange(self.d, device=x.device).view(1,-1,1).repeat(2,1,1) #2xdx1
        hyperedge_index = hyperedge_index.unsqueeze(1) #2x1xK
        hyperedge_index = self.d * hyperedge_index + d_range #2xdxK
        hyperedge_index = hyperedge_index.permute(0,2,1).reshape(2,-1) #2x(d*K)
        h_sheaf_index = hyperedge_index

        #the resulting (index, values) pair correspond to the diagonal of each block sub-matrix
        return h_sheaf_index, h_sheaf_attributes
        
    
#Build the restriction maps for the General Case
class SheafBuilderGeneral(nn.Module):
    def __init__(self, sheaf_pred_block='cp_decomp', sheaf_dropout=True, sheaf_special_head=False, heads=1, MLP_hidden=256, dropout=0.25, AllSet_input_norm=True):
        super(SheafBuilderGeneral, self).__init__()
        self.prediction_type = sheaf_pred_block # pick the way hyperedge feartures are computed
        self.sheaf_dropout = sheaf_dropout
        self.special_head = sheaf_special_head # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = heads #stalk dimension
        self.MLP_hidden = MLP_hidden
        self.norm = AllSet_input_norm
        self.norm_type = 'sym_degree_norm'
        self.dropout = dropout

        
        self.general_sheaf_lin = MLP(
                in_channels=2*self.MLP_hidden, 
                hidden_channels=MLP_hidden,
                out_channels=self.d*self.d,
                num_layers=1,
                dropout=0.0,
                Normalization='ln',
                InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(
                        in_channels=self.MLP_hidden+1, 
                        hidden_channels=MLP_hidden,
                        out_channels=MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
            self.cp_V = MLP(
                        in_channels=MLP_hidden, 
                        hidden_channels=MLP_hidden,
                        out_channels=MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        if self.prediction_type == 'transformer':
            self.transformer_lin_layer.reset_parameters()
            self.transformer_layer.reset_parameters()
        elif self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin.reset_parameters()
            self.general_sheaf_lin2.reset_parameters()
        else:
            self.general_sheaf_lin.reset_parameters()
        if self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

        

    def forward(self, x, e, hyperedge_index, debug=False):
        """ 
        x: N x f
        e: N x f 
        -> (concat) N x E x 2f -> (linear project) N x E x d*d
        -> (reshape) (Nd x Ed) with each block dxd being unconstrained

        """
        num_nodes =  x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # N x d x f -> N x f

        row, col = hyperedge_index
        x_row = torch.index_select(x, dim=0, index=row)
        e_col = torch.index_select(e, dim=0, index=col)

        if self.prediction_type == 'MLP_var1':
            h_general_sheaf = predict_blocks(x, e, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_general_sheaf = predict_blocks_var2(x, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_general_sheaf = predict_blocks_var3(x, hyperedge_index, self.general_sheaf_lin, self.general_sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_general_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.general_sheaf_lin)
        
        if debug:
            self.h_general_sheaf = h_general_sheaf #for debug purpose

        if self.sheaf_dropout:
            h_general_sheaf = F.dropout(h_general_sheaf, p=self.dropout, training=self.training)

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # modify indices to correspond to the big matrix and assign the weights
        # indices: [i,j] -> [d*i, d*i.. d*i+d-1, d*i+d-1; d*j, d*j+1 .. d*j, d*j+1,..d*j+d-1]
        
        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(-1,1) #0,1..d,0,1..d..   d*d elems
        d_range_nodes = d_range.repeat_interleave(self.d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
        hyperedge_index = hyperedge_index.unsqueeze(1) 

        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
        h_general_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)
        h_general_sheaf_attributes = h_general_sheaf.reshape(-1)
        
        #create the big matrix from the dxd blocks  
        return h_general_sheaf_index, h_general_sheaf_attributes


#Build the restriction maps for the Orthogonal Case


#Build the restriction maps for the LowRank Case
class SheafBuilderLowRank(nn.Module):
    def __init__(self, args):
        super(SheafBuilderLowRank, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block # pick the way hyperedge feartures are computed
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = args.heads # stalk dimension
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm
        self.norm_type = args.sheaf_normtype

        self.rank = args.rank # rank for the block matrices
        self.dropout = args.dropout

        self.general_sheaf_lin = MLP(
                    in_channels=2*self.MLP_hidden, 
                    hidden_channels=args.MLP_hidden,
                    out_channels=2*self.d*self.rank+self.d,
                    num_layers=1,
                    dropout=0.0,
                    Normalization='ln',
                    InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(
                        in_channels=self.MLP_hidden+1, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
            self.cp_V = MLP(
                        in_channels=args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        if self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin.reset_parameters()
            self.general_sheaf_lin2.reset_parameters()
        else:
            self.general_sheaf_lin.reset_parameters()
        if self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index, debug=False):
        """ 
        x: N x f
        e: N x f 
        -> (concat) N x E x 2f -> (linear project) N x E x d*d
        -> (reshape) (Nd x Ed) with each block dxd being unconstrained

        """
        num_nodes =  x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # N x d x f -> N x f

        row, col = hyperedge_index
        x_row = torch.index_select(x, dim=0, index=row)
        e_col = torch.index_select(e, dim=0, index=col)

        if self.prediction_type == 'MLP_var1':
            h_general_sheaf = predict_blocks(x, e, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_general_sheaf = predict_blocks_var2(x, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_general_sheaf = predict_blocks_var3(x, hyperedge_index, self.general_sheaf_lin, self.general_sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_general_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.general_sheaf_lin, self.args)

        # compute AB^T + diag(c)
        # h_general_sheaf is nnz x (2*d*r)
        h_general_sheaf_A = h_general_sheaf[:, :self.d*self.rank].reshape(h_general_sheaf.shape[0], self.d, self.rank) #nnz x d x r
        h_general_sheaf_B = h_general_sheaf[:, self.d*self.rank:2*self.d*self.rank].reshape(h_general_sheaf.shape[0], self.d, self.rank) #nnz x d x r
        h_general_sheaf_C = h_general_sheaf[:, 2*self.d*self.rank:].reshape(h_general_sheaf.shape[0], self.d) #nnz x d x r
        
        h_general_sheaf = torch.bmm(h_general_sheaf_A,h_general_sheaf_B.transpose(2,1)) #rank-r matrix    
        #add elements on the diagonal
        diag = torch.diag_embed(h_general_sheaf_C)
        h_general_sheaf = h_general_sheaf + diag

        h_general_sheaf = h_general_sheaf.reshape(h_general_sheaf.shape[0], self.d*self.d)
        
        if debug:
            self.h_general_sheaf = h_general_sheaf #for debug purpose
        if self.sheaf_dropout:
            h_general_sheaf = F.dropout(h_general_sheaf, p=self.dropout, training=self.training)

        # from a d-dim tensor assoc to every entrence in edge_index
        # create a sparse incidence Nd x Ed

        # modify indices to correspond to the big matrix and assign the weights
        # indices: [i,j] -> [d*i, d*i.. d*i+d-1, d*i+d-1; d*j, d*j+1 .. d*j, d*j+1,..d*j+d-1]
        
        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(-1,1) #0,1..d,0,1..d..   d*d elems
        d_range_nodes = d_range.repeat_interleave(self.d).view(-1,1) #0,0..0,1,1..1..d,d..d  d*d elems
        hyperedge_index = hyperedge_index.unsqueeze(1) 
   

        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
        h_general_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)

        if self.norm_type == 'block_norm':
            # pass
            h_general_sheaf_1 = h_general_sheaf.reshape(h_general_sheaf.shape[0], self.d, self.d)
            num_nodes = hyperedge_index[0].max().item() + 1
            num_edges = hyperedge_index[1].max().item() + 1

            to_be_inv_nodes = torch.bmm(h_general_sheaf_1, h_general_sheaf_1.permute(0,2,1)) 
            to_be_inv_nodes = scatter_add(to_be_inv_nodes, row, dim=0, dim_size=num_nodes)

            to_be_inv_edges = torch.bmm(h_general_sheaf_1.permute(0,2,1), h_general_sheaf_1)
            to_be_inv_edges = scatter_add(to_be_inv_edges, col, dim=0, dim_size=num_edges)


            d_sqrt_inv_nodes = utils.batched_sym_matrix_pow(to_be_inv_nodes, -1.0) #n_nodes x d x d
            d_sqrt_inv_edges = utils.batched_sym_matrix_pow(to_be_inv_edges, -1.0) #n_edges x d x d
            

            d_sqrt_inv_nodes_large = torch.index_select(d_sqrt_inv_nodes, dim=0, index=row)
            d_sqrt_inv_edges_large = torch.index_select(d_sqrt_inv_edges, dim=0, index=col)


            alpha_norm = torch.bmm(d_sqrt_inv_nodes_large, h_general_sheaf_1)
            alpha_norm = torch.bmm(alpha_norm, d_sqrt_inv_edges_large)
            h_general_sheaf = alpha_norm.clamp(min=-1, max=1)
            h_general_sheaf = h_general_sheaf.reshape(h_general_sheaf.shape[0], self.d*self.d)

        h_general_sheaf_attributes = h_general_sheaf.reshape(-1)
        #create the big matrix from the dxd blocks  
        return h_general_sheaf_index, h_general_sheaf_attributes


# the hidden dimensiuon are a bit differently computed
# That's why the classes are separated. 
# We shoul merge them at some point
# Moreover, the function only return the values, not the indices in this case

class HGCNSheafBuilderDiag(nn.Module):
    def __init__(self, args, hidden_dim):
        """
        hidden_dim overwrite the args.MLP_hidden used in the normal sheaf HNN
        """
        super(HGCNSheafBuilderDiag, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block # pick the way hyperedge feartures are computed
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = args.heads # stalk dimension 
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout


        if self.prediction_type == 'MLP_var1':
            self.sheaf_lin = MLP(
                        in_channels=self.MLP_hidden + args.MLP_hidden,  
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        else:
            self.sheaf_lin = MLP(
                        in_channels=2*self.MLP_hidden,  
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(
                        in_channels=self.MLP_hidden+1, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
            self.cp_V = MLP(
                        in_channels=args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.MLP_hidden)             

    def reset_parameters(self):
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    #this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1

        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

       
        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

       
        return h_sheaf

class HGCNSheafBuilderGeneral(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNSheafBuilderGeneral, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block # pick the way hyperedge feartures are computed
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = args.heads #stalk dimension
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout
        
        if self.prediction_type == 'MLP_var1':
            self.sheaf_lin = MLP(
                        in_channels=self.MLP_hidden + args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        else:
            self.sheaf_lin = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(
                        in_channels=self.MLP_hidden+1, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
            self.cp_V = MLP(
                        in_channels=args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.MLP_hidden)              

    def reset_parameters(self):
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    #this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        # h_sheaf = self.predict_blocks(x, e, hyperedge_index, sheaf_lin)
        # h_sheaf = self.predict_blocks_var2(x, hyperedge_index, sheaf_lin)
        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        
        return h_sheaf
        
class HGCNSheafBuilderOrtho(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNSheafBuilderOrtho, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block # pick the way hyperedge feartures are computed
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = args.heads
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map='householder') #method applied to transform params into ortho dxd matrix
        self.dropout = args.dropout

        if self.prediction_type == 'MLP_var1':
            self.sheaf_lin = MLP(
                        in_channels=self.MLP_hidden + args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*(self.d-1)//2,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        else:
            self.sheaf_lin = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.d*(self.d-1)//2,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(
                        in_channels=self.MLP_hidden+1, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
            self.cp_V = MLP(
                        in_channels=args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.MLP_hidden)             

    def reset_parameters(self):
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    #this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)

        h_sheaf = self.orth_transform(h_sheaf) #sparse version of a NxExdxd tensor
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf
        
class HGCNSheafBuilderLowRank(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNSheafBuilderLowRank, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block # pick the way hyperedge feartures are computed
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head # add a head having just 1 on the diagonal. this should be similar to the normal hypergraph conv
        self.d = args.heads #stalk dimension
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm

        self.rank = args.rank
        self.dropout = args.dropout
        
        if self.prediction_type == 'MLP_var1':
            self.sheaf_lin = MLP(
                        in_channels=self.MLP_hidden + args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=2*self.d*self.rank+self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        else:
            self.sheaf_lin = MLP(
                        in_channels=2*self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=2*self.d*self.rank+self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(
                        in_channels=self.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(
                        in_channels=self.MLP_hidden+1, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.norm)
            self.cp_V = MLP(
                        in_channels=args.MLP_hidden, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=self.MLP_hidden)                

    def reset_parameters(self):
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin.reset_parameters()
            self.sheaf_lin2.reset_parameters()
        else:
            self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    #this is exclusively for diagonal sheaf
    def forward(self, x, e, hyperedge_index):
        """ tmp
        x: Nd x f -> N x f
        e: Ed x f -> E x f
        -> (concat) N x E x (d+1)F -> (linear project) N x E x d (the elements on the diagonal of each dxd block)
        -> (reshape) (Nd x Ed) with NxE diagonal blocks of dimension dxd

        """

        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) # N x d x f -> N x f
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1) # # x d x f -> E x f

        # h_sheaf = self.predict_blocks(x, e, hyperedge_index, sheaf_lin)
        # h_sheaf = self.predict_blocks_var2(x, hyperedge_index, sheaf_lin)
        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin,  self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)

        # h_general_sheaf is nnz x (2*d*r)
        h_general_sheaf_A = h_sheaf[:, :self.d*self.rank].reshape(h_sheaf.shape[0], self.d, self.rank) #nnz x d x r
        h_general_sheaf_B = h_sheaf[:, self.d*self.rank:2*self.d*self.rank].reshape(h_sheaf.shape[0], self.d, self.rank) #nnz x d x r
        h_general_sheaf_C = h_sheaf[:, 2*self.d*self.rank:].reshape(h_sheaf.shape[0], self.d) #nnz x d x r

        h_sheaf = torch.bmm(h_general_sheaf_A,h_general_sheaf_B.transpose(2,1)) #rank-r matrix

        diag = torch.diag_embed(h_general_sheaf_C)
        h_sheaf = h_sheaf + diag

        h_sheaf = h_sheaf.reshape(h_sheaf.shape[0], self.d*self.d)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf
        