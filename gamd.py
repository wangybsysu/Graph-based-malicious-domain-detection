import argparse
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler



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

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        #self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.attention = Attention(hid_size*4)
        self.attention1 = Attention(out_size)
        self.gat1 = dglnn.GATConv(in_size,hid_size,heads[0],feat_drop=0.2,
                attn_drop=0.2,activation=F.relu,
            )
        self.gat2 = dglnn.GATConv(in_size,hid_size,heads[0],feat_drop=0.2,
                attn_drop=0.2,activation=F.relu,
            )
        self.gat3 = dglnn.GATConv(in_size,hid_size,heads[0],feat_drop=0.2,
                attn_drop=0.2,activation=F.relu,
            )
        self.gat4 = dglnn.GATConv(in_size,hid_size,heads[0],feat_drop=0.2,
                attn_drop=0.2,activation=F.relu,
            )
        
        self.gat11 = dglnn.GATConv(hid_size * heads[0],hid_size,heads[1],feat_drop=0.2,
                attn_drop=0.2,activation=F.relu,
            )
        self.gat22 = dglnn.GATConv(hid_size * heads[0],hid_size,heads[1],feat_drop=0.2,
                attn_drop=0.2,activation=F.relu,
            )
        self.gat33 = dglnn.GATConv(hid_size * heads[0],hid_size,heads[1],feat_drop=0.2,
                attn_drop=0.2,activation=F.relu,
            )
        self.gat44 = dglnn.GATConv(hid_size * heads[0],hid_size,heads[1],feat_drop=0.2,
                attn_drop=0.2,activation=F.relu,
            )
        
        self.gat111 = dglnn.GATConv(hid_size * heads[1],out_size,heads[2],feat_drop=0.2,
                attn_drop=0.2,activation=None,
            )
        self.gat222 = dglnn.GATConv(hid_size * heads[1],out_size,heads[2],feat_drop=0.2,
                attn_drop=0.2,activation=None,
            )
        self.gat333 = dglnn.GATConv(hid_size * heads[1],out_size,heads[2],feat_drop=0.2,
                attn_drop=0.2,activation=None,
            )
        self.gat444 = dglnn.GATConv(hid_size * heads[1],out_size,heads[2],feat_drop=0.2,
                attn_drop=0.2,activation=None,
            )

        
        self.MLP = nn.Sequential(
            nn.Linear(out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 2)
        )        

# gip, gsub represent heuristic relational graph structures (in DGL graph format).
# inputs is the node features 
    def forward(self, gip, gsub, gsuf, gasn, inputs):
        h = inputs
        hip = self.gat1(gip,h).flatten(1)
        hsub = self.gat2(gsub,h).flatten(1)
        hip = (hip+h)/2
        hsub = (hsub+h)/2
        #hsuf = self.gat3(gsuf,h).flatten(1)
        #hasn = self.gat4(gasn,h).flatten(1)
        h = torch.stack([hip,hsub], dim=1)
        #h = torch.stack([hip,hsub,hsuf], dim=1)
        #h = torch.stack([hip,hsub,hsuf,hasn], dim=1)
        h, att = self.attention(h)
        
        hip = self.gat11(gip,h).flatten(1)
        hsub = self.gat22(gsub,h).flatten(1)
        hip = (hip+h)/2
        hsub = (hsub+h)/2
        #hsuf = self.gat33(gsuf,h).flatten(1)
        #hasn = self.gat44(gasn,h).flatten(1)
        h = torch.stack([hip,hsub], dim=1)
        #h = torch.stack([hip,hsub,hsuf], dim=1)
        #h = torch.stack([hip,hsub,hsuf,hasn], dim=1)
        h, att = self.attention(h)

        hip = self.gat11(gip,h).flatten(1)
        hsub = self.gat22(gsub,h).flatten(1)
        hip = (hip+h)/2
        hsub = (hsub+h)/2
        #hsuf = self.gat33(gsuf,h).flatten(1)
        #hasn = self.gat44(gasn,h).flatten(1)
        h = torch.stack([hip,hsub], dim=1)
        #h = torch.stack([hip,hsub,hsuf], dim=1)
        #h = torch.stack([hip,hsub,hsuf,hasn], dim=1)
        h, att = self.attention(h)

        hip = self.gat111(gip,h).mean(1)
        hsub = self.gat222(gsub,h).mean(1)
        hip = (hip+h)/2
        hsub = (hsub+h)/2
        #hsuf = self.gat333(gsuf,h).mean(1)
        #hasn = self.gat444(gasn,h).mean(1)
        h = torch.stack([hip,hsub], dim=1)
        #h = torch.stack([hip,hsub,hsuf], dim=1)
        #h = torch.stack([hip,hsub,hsuf,hasn], dim=1)
        h, att = self.attention1(h)

        h = self.MLP(h)
        return h        
    


