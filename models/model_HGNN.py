import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from models.layers.cross_attention import FeedForward, MMAttentionLayer
from models.layers.fusion import GraphFusion, AlignFusion
from models.layers.layers import *
from models.layers.sheaf_builder import *
from torch_scatter import scatter_mean
from .util import initialize_weights
from .util import NystromAttention
from .util import SNN_Block
from .util import MultiheadAttention
import dhg
from dhg.nn import HGNNPConv
from collections import defaultdict
    
class MRePath(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion="concat", model_size="small",graph_type="HGNN"):
        super(MRePath, self).__init__()

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion

        ###
        self.size_dict = {
            "pathomics": {"small": [768, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU6())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        self.graph_type = graph_type
        
        if self.graph_type == "HGNN":
            self.sheaf_builder = SheafBuilderGeneral()
            self.convs=nn.ModuleList()
            # Sheaf Diffusion layers
            for _ in range(3):
                self.convs.append(HyperDiffusionGeneralSheafConv(256, 256, d=1, device='cuda'))
        elif self.graph_type == "GCN":
            from torch_geometric.nn.models import GCN
            self.graph = GCN(in_channels=256, hidden_channels=512, out_channels=256, num_layers=3, dropout=0.25).to("cuda")
        elif self.graph_type == "GAT":
            from torch_geometric.nn.models import GAT
            self.graph = GAT(in_channels=256, hidden_channels=512, out_channels=256, num_layers=3, dropout=0.25).to("cuda")                           
        
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)
       
        
        # modality balance
        g_dim = self.size_dict["genomics"][model_size][-1]
        p_dim = self.size_dict["pathomics"][model_size][-1]
        g_num = 6
        p_num = 4096
        
        # self.gene_multiattention = nn.MultiheadAttention(embed_dim=g_dim, num_heads=2*self.n_classes)
        self.ConfidNet_g = nn.Sequential(
            nn.Linear(g_num*g_dim, g_num*g_dim*2),
            nn.Linear(g_num*g_dim*2, g_num*g_dim),
            nn.Linear(g_num*g_dim, 1),
            nn.Sigmoid()
        )
        self.ConfiNet_p_pre=nn.Linear(p_dim,1)
        self.ConfidNet_p = nn.Sequential(
            nn.Linear(p_num, p_num*2),
            nn.Linear(p_num*2, p_num),
            nn.Linear(p_num, 1),
            nn.Sigmoid()
        )
        
        self.attention_fusion = AlignFusion(
            embedding_dim=g_dim,
            num_heads = 4,
            num_pathways = g_num
        )

        # Classification Layer
        self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1]*2, hidden[-1]//2), nn.ReLU()]
            )
        self.feed_forward = FeedForward(g_dim, dropout=0.25)
        self.layer_norm = nn.LayerNorm(g_dim )

        self.classifier = nn.Linear(hidden[-1]//2, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]
        
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)  # [1, 6, 1024]
        pathomics_features = self.pathomics_fc(x_path)[0]
        
        # graph structure
        graph = kwargs["graph"]
        edge_index = graph.edge_index
        edge_latent = graph.edge_latent
        
        # sheaf hypergraph
        if self.graph_type == "HGNN":
          if edge_index.shape[1]+edge_latent.shape[1]>0:
              hyper_index = self.get_hyperedge(edge_index)
              hyper_latent = self.get_hyperedge(edge_latent)
            
              hg = dhg.Hypergraph(num_v=pathomics_features.shape[0], e_list=hyper_index+hyper_latent)

              H = hg.H.coalesce().indices().long().to('cuda')
              hyperedge_attr = self.init_hyperedge_attr(x=pathomics_features, hyperedge_index=H.to('cuda'))
              num_nodes=pathomics_features.shape[0]
              num_edges=H[1].max().item() + 1
        
        # calculate modality rebalance
        wsi_f = pathomics_features.clone()
        wsi_f = wsi_f.unsqueeze(0)
        gene_f = genomics_features.clone()
        
        if wsi_f.shape[1] == 4096:
                tmp = self.ConfiNet_p_pre(wsi_f).view(1,-1)
                path_tcp = self.ConfidNet_p(tmp)
                gene_tcp = self.ConfidNet_g(gene_f.view(1,-1))
                
                path_holo = torch.log(gene_tcp)/(torch.log(path_tcp*gene_tcp)+1e-8)
                gene_holo = torch.log(path_tcp)/(torch.log(path_tcp*gene_tcp)+1e-8)
                
                cb_path = path_tcp.detach() + path_holo.detach()
                cb_gene = gene_tcp.detach() + gene_holo.detach()
                
                w_all = torch.stack((cb_path, cb_gene),1)
                softmax = nn.Softmax(1)
                w_all = softmax(w_all)
                
                w_path = w_all[:,0]
                w_gene = w_all[:,1]
        else:
                path_tcp = self.ConfiNet_p_pre(wsi_f).view(1,-1)
                path_tcp = nn.Sigmoid()(path_tcp.mean(dim=1).unsqueeze(0))
                gene_tcp = self.ConfidNet_g(gene_f.view(1,-1))
        
                path_holo = torch.log(gene_tcp)/(torch.log(path_tcp*gene_tcp)+1e-8)
                gene_holo = torch.log(path_tcp)/(torch.log(path_tcp*gene_tcp)+1e-8)
                cb_path = path_tcp.detach() + path_holo.detach()
                cb_gene = gene_tcp.detach() + gene_holo.detach()
                w_all = torch.stack((cb_path, cb_gene),1)
                softmax = nn.Softmax(1)
                w_all = softmax(w_all)
                w_path = w_all[:,0]
                w_gene = w_all[:,1]
                
        # three layers hypergraph convolution
        if self.graph_type=="HGNN":
            for i, conv in enumerate(self.convs[:-1]):
                if i == 0 :
                    h_sheaf_index, h_sheaf_attributes = self.sheaf_builder(pathomics_features, hyperedge_attr, H)
                # Sheaf Laplacian Diffusion
                pathomics_features = conv(pathomics_features, hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges)
                pathomics_features = F.dropout(pathomics_features, p=0.25, training=True)

            
            # Sheaf Laplacian Diffusion
            pathomics_features = self.convs[-1](pathomics_features,  hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges)
            pathomics_features = pathomics_features.view(-1, 256).unsqueeze(0) # Nd x out_channels -> Nx(d*out_channels)
        else: 
            pathomics_features = self.graph(pathomics_features, edge_total).unsqueeze(0)
        
        token = torch.cat((genomics_features, pathomics_features), dim=1)
        
        token_cross = self.attention_fusion(token=token)
        
        token_cross = self.feed_forward(token_cross)
        token_cross = self.layer_norm(token_cross)
        
        paths_postSA_embed = token_cross[:, :6, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = token_cross[:, 6:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)
        
        fusion = self.mm(
                torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)
            )
        # predict
        logits = self.classifier(fusion)  
        
        return logits
        
    def get_hyperedge(self, edge):
        adj_matrix = edge.cpu().numpy()
        hyperedges = defaultdict(set)
        for start, end in adj_matrix.T:
            hyperedges[start].add(end)
        hypergraph_edges = []

        for start_node, end_nodes in hyperedges.items():
            edge = {start_node}.union(end_nodes)
            hypergraph_edges.append(list(edge))

        return hypergraph_edges
        
    def init_hyperedge_attr(self, type='avg', num_edges=None, x=None, hyperedge_index=None):
        #initialize hyperedge attributes either random or as the average of the node
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr
    def hyperedge_to_incidence_matrix(self,hyperedge_index, num_nodes, num_hyperedges):
        hyperedge_index = hyperedge_index.coalesce()

        # COO indices
        node_indices = hyperedge_index.indices()[0]
        edge_indices = hyperedge_index.indices()[1] 

        values = torch.ones(node_indices.size(0), dtype=torch.float32)

        incidence_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([node_indices, edge_indices]),
            values=values,
            size=(num_nodes, num_hyperedges)
        )

        return incidence_matrix
        

    
