#%%
from functools import reduce
import signal
import time
from joblib import Parallel, delayed
import math
import itertools
import pandas as pd
import networkx as nx
import numpy as np
import factorgraph as fg
from multiprocessing import Process



# load heuristic association data
nodelist = pd.read_csv('phishingnode.csv') #Original node

trainnode = pd.read_csv('trainnode.csv') #Training data
testnode = pd.read_csv('testnode.csv') #Testing  data
dom_ipw = pd.read_csv('dom_ipw.csv') #IP heuristic relation association graph
dom_subnetw = pd.read_csv('dom_subnetw.csv') #subnet heuristic relation association graph
dom_asnw = pd.read_csv('dom_asnw.csv') #ASN heuristic relation association graph
dom_suffix = pd.read_csv('dom_suffix.csv') #suffix heuristic relation association graph


trainnode = trainnode[(trainnode['label']==0)|(trainnode['label']==1)]
testnode = testnode[(testnode['label']==0)|(testnode['label']==1)]


# Split into subgraphs to speed up operations
def cnsubg(dom_asnw):
    G = nx.Graph()
    A = dom_asnw['start'].to_list()
    B = dom_asnw['end'].to_list()
    percetage = 0
    for i in range(dom_asnw.shape[0]):
        if percetage != math.ceil(i/dom_asnw.shape[0]*100):
            percetage = math.ceil(i/dom_asnw.shape[0]*100)
            print('Percentage of completed operations: '+str(percetage)+'%')
        G.add_edge(A[i], B[i])
    connected_subgraphs = list(nx.connected_components(G))
    return connected_subgraphs

def subd(connected_subgraphs):
    i=0
    subgdic = {}
    for subgraph in connected_subgraphs:
        #print(i)
        subgraph = list(subgraph)
        for sub in subgraph:
            subgdic[sub] = i
        i+=1
    return subgdic

# Only the subgraph with labeled nodes is kept. 
# If all the nodes in the subgraph have no labels, the inference cannot operate.
def upsubg(dom_ipw, testnode_all):
    connected_subgraphs = cnsubg(dom_ipw)
    subgdic = subd(connected_subgraphs)
    mark = []
    for nd in testnode_all:
        mark.append(subgdic.get(nd))
    mark = list(set(mark))

    connected_subgraphsnew = []
    for subg in mark:
        connected_subgraphsnew.append(connected_subgraphs[subg])
    return connected_subgraphsnew

# Perform inference computations
def malcaulate(subg, edges, trainnode_bad, trainnode_good, i):
    signal.signal(signal.SIGALRM, timeout_handler)
    # timmer
    signal.alarm(3600)
    result = []
    try:
        g = fg.Graph()
        edges = edges[['start','end']].values

        for node in subg:
            if node in trainnode_bad:
                g.rv(str(node), 2)
                g.factor([str(node)], potential=np.array([0.01, 0.99]))
            else:
                if node in trainnode_good:
                    g.rv(str(node), 2)
                    g.factor([str(node)], potential=np.array([0.99, 0.01]))
                else:
                    g.rv(str(node), 2)
                    g.factor([str(node)], potential=np.array([0.5, 0.5]))
        for edge in edges:
            g.factor([str(edge[0]), str(edge[1])], potential=np.array([
                [0.51, 0.49],
                [0.49, 0.51]
            ]))
        iters, converged = g.lbp(normalize=True)
        print('the '+str(i)+'-th subgraph calculate done!!')
        malresults = g.rv_marginals()
        
        for res in malresults:
            result.append([str(res[0]), np.argmax(res[1])])
    except:
        print('The '+str(i)+'-th subgraph operation failed!!!')
        result = []
    finally:
        signal.alarm(0)
    return result


# %%
#Multi-threaded execution generates inference results
dom_w = dom_ipw
nodes = dom_w['start'].to_list()+dom_w['end'].to_list()
nodes = list(set(nodes))
testnode_all = testnode[(testnode['index'].isin(nodes))]['index'].to_list()
trainnode_bad = trainnode[(trainnode['index'].isin(nodes)) & trainnode['label']==1]['index'].to_list()
trainnode_good = trainnode[(trainnode['index'].isin(nodes)) & trainnode['label']==0]['index'].to_list()
connected_subgraphsnew = upsubg(dom_w, testnode_all)

results = Parallel(n_jobs=70, timeout=3600)(delayed(malcaulate)(subg, 
                                                  dom_w[(dom_w['start'].isin(subg)) | (dom_w['end'].isin(subg))], 
                                                  trainnode_bad, trainnode_good, connected_subgraphsnew.index(subg)) for subg in connected_subgraphsnew)

# %%
