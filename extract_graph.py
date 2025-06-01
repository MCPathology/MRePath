import os, sys
import argparse
from os.path import join
import h5py
import math
from math import floor
import pdb
from time import time
from tqdm import tqdm

### Numerical Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import percentileofscore

### Graph Network Packages
import nmslib
import networkx as nx

### PyTorch / PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import convert


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

def pt2graph(wsi_h5, radius=5):
    from torch_geometric.data import Data as geomData
    from itertools import chain
    coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]
    
    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),dtype=int)
    edge_spatial = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)
    
    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(features[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),dtype=int)
    edge_latent = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)

    G = geomData(x = torch.Tensor(features),
                 edge_index = edge_spatial,
                 edge_latent = edge_latent,
                 centroid = torch.Tensor(coords))
    return G

def createDir_h5toPyG(h5_path, save_path):
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        pbar.set_description('%s - Creating Graph' % (h5_fname[:12]))

        try:
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            G = pt2graph(wsi_h5,radius=9)
            torch.save(G, os.path.join(save_path, h5_fname[:-3]+'.pt'))
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')

def main(args):
    h5_path = args.h5_path
    save_path = args.graph_save_path
    os.makedirs(save_path, exist_ok=True)
    createDir_h5toPyG(h5_path, save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type = str,default='',
                        help='path to FFPE h5 files proceeded by CLAM')
    parser.add_argument('--graph_save_path', type = str,default='',
                        help='path to store the generated graph')
    args = parser.parse_args()
    results = main(args)
    print("finished!")