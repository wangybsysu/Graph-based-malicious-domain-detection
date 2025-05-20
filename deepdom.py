
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import math
import torch.optim as optim
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Convert domain names to ASCII-encoded features.
def f_eone1(domainlist):
    fe_al = []
    for domain in domainlist:
        fe = [0]*128
        for y in domain:
            fe[ord(y)] +=1
        fe1 = [0]*64
        fe1[0:2] = fe[45:47]
        fe1[2:12] = fe[48:58]
        fe1[12:38] = fe[65:91]
        fe1[38:64] = fe[97:123]
        fe_al.append(fe1)
    scaler = MinMaxScaler()
    scaler = scaler.fit(fe_al)
    fe_al = scaler.transform(fe_al)
    return fe_al


# Generate node neighbor sampling based on heuristic relational graph structures.
def adjgen(dom_w, nodelist):
    dom_edges = pd.DataFrame({'start': (dom_w['start'].to_list()+dom_w['end'].to_list()),
                            'end':(dom_w['end'].to_list()+dom_w['start'].to_list())})
    dom_adjdic = {} # get domain adj relations
    domadjgropu = dom_edges.groupby('start')
    for name, group in domadjgropu:
        dom_adjdic[name] = group['end'].values
    nodes = nodelist[(nodelist['label']==1) | (nodelist['label']==0)] # all domain nodes
    nodes_adj = np.array(np.zeros((1,15)))
    #nodes_adjfeat = np.array(np.zeros((1,64)))
    for i in range(nodes.shape[0]):
        print(i)
        adjdom = dom_adjdic.get(i, [])
        if len(adjdom)!=0:
            if adjdom.shape[0]>=5:
                adj=np.hstack((np.random.choice(adjdom, size=5, replace=False),
                            np.random.choice(adjdom, size=5, replace=False),
                            np.random.choice(adjdom, size=5, replace=False)))
                nodes_adj = np.vstack((nodes_adj,adj))
            else:
                adjlist=np.hstack((adjdom, np.array([nodes.shape[0]]*(5-adjdom.shape[0]))))
                adj=np.hstack((adjlist,adjlist,adjlist))
                nodes_adj = np.vstack((nodes_adj,adj))
        else:
            nodes_adj=np.vstack((nodes_adj,np.array([nodes.shape[0]]*15)))
    nodes_adj = np.vstack((nodes_adj[1:],nodes_adj[0]))
    return nodes_adj


# load input data， same as bp
# Please note that the number of sampling iterations should match the number of model layers.
# Please note that the sampled data should be updated in each training round (though the actual impact is minor).
dom_ipadj = adjgen(dom_ipw, nodelist)
dom_sufadj = adjgen(dom_suffix, nodelist)
dom_subadj = adjgen(dom_subnetw, nodelist)
dom_asnadj = adjgen(dom_asnw, nodelist)

# Please load the labels, features, and train/test masks by yourself.
data = Data(x = torch.tensor(nodes_feat, dtype=torch.float32),
            y = torch.tensor(labels, dtype=torch.int64),
            train_mask = torch.tensor(train_mask,dtype=torch.bool),
            test_mask = torch.tensor(test_mask,dtype=torch.bool),
            dom_ipadj = torch.tensor(dom_ipadj,dtype=torch.int64),
            dom_sufadj = torch.tensor(dom_sufadj,dtype=torch.int64),
            dom_subadj = torch.tensor(dom_subadj,dtype=torch.int64),
            dom_asnadj = torch.tensor(dom_asnadj,dtype=torch.int64),
            )

#%%
# attention layer
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

# deepdom model
class Deepdom(nn.Module):
    def __init__(self, nhid, drpout):
        super(Deepdom, self).__init__()

        self.attention1 = Attention(nhid)        
        self.linear1 = nn.Linear(nhid*2, nhid)
        self.attention2 = Attention(nhid)        
        self.linear2 = nn.Linear(nhid*2, nhid)
        self.attention3 = Attention(nhid)        
        self.linear3 = nn.Linear(nhid*2, nhid)        
        self.dropout = nn.Dropout(drpout)
        self.relu = nn.ReLU()

        self.MLP = nn.Sequential(
            nn.Linear(nhid, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 2)
        )

    def forward(self, x, ip_adj=None, sub_adj=None, suf_adj=None, asn_adj=None):
        # Generate the matrixes of random walk neighbors
        # Please note that the number of sampling iterations should match the number of model layers.
        # Take the average of the neighbor nodes generated for each meta-path, respectively.
        ipfeat = x[ip_adj[0]].mean(1)
        subfeat = x[sub_adj[0]].mean(1)
        suffeat = x[suf_adj[0]].mean(1)
        asnfeat = x[asn_adj[0]].mean(1)
        #y = torch.stack([ipfeat,subfeat], dim=1)
        #y = torch.stack([ipfeat,subfeat,suffeat], dim=1)
        y = torch.stack([ipfeat,subfeat,suffeat,asnfeat], dim=1)
        # utilize attention mechanism aggregates neighbors.
        y, att = self.attention1(y)
        x = torch.cat((x,y),dim=1)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)

        # Update the matrixes of random walk neighbors
        ipfeat = x[ip_adj[1]].mean(1)
        subfeat = x[sub_adj[1]].mean(1)
        suffeat = x[suf_adj[1]].mean(1)
        asnfeat = x[asn_adj[1]].mean(1)
        #y = torch.stack([ipfeat,subfeat], dim=1)
        #y = torch.stack([ipfeat,subfeat,suffeat], dim=1)
        y = torch.stack([ipfeat,subfeat,suffeat,asnfeat], dim=1)
        y, att = self.attention2(y)
        x = torch.cat((x,y),dim=1)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)        

        # Update the matrixes of random walk neighbors
        ipfeat = x[ip_adj[2]].mean(1)
        subfeat = x[sub_adj[2]].mean(1)
        suffeat = x[suf_adj[2]].mean(1)
        asnfeat = x[asn_adj[2]].mean(1)
        #y = torch.stack([ipfeat,subfeat], dim=1)
        #y = torch.stack([ipfeat,subfeat,suffeat], dim=1)
        y = torch.stack([ipfeat,subfeat,suffeat,asnfeat], dim=1)
        y, att = self.attention3(y)
        x = torch.cat((x,y),dim=1)
        x = self.linear3(x)
        x = self.dropout(x)

        output = self.MLP(x)
        return output


# Please note that the sampled data should be updated in each training round (though the actual impact is minor).
# The basic training process is a loop of: sampling → feeding into the model → optimizing the model.

