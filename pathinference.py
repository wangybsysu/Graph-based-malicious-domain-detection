
#%%
from functools import reduce
import signal
import time
import pandas as pd
import networkx as nx

# Input data, it is in dataframe format, each entry is: domain-suffix-ip-subnet-asn-operator
edge_up 

#%% 

# Grouping, generating dictionaries according to groups, computing edge weights(Jaccard similarity coefficient)
def calcuw(edge_up):
    grp_dom = edge_up.groupby('start')
    dom_d,dom_ip,dom_subnet,dom_asn = [],{},{},{}
    for dom, group in grp_dom:
        dom_ip[dom] = set(group['end'].to_list())
        dom_subnet[dom] = set(group['subnet'].to_list())
        dom_asn[dom] = set(group['asn'].to_list())
        dom_d.append(dom)
    dom_ipw,dom_subnetw,dom_asnw = [],[],[]
    for i in range(len(dom_d)):
        print(i)
        if i < len(dom_d)-1:
            for j in range(i+1, len(dom_d)):
                ipw = 1-1/(1+len(dom_ip[dom_d[i]].intersection(dom_ip[dom_d[j]])))
                if ipw!=0:
                    dom_ipw.append([dom_d[i],dom_d[j], ipw])
                subnetw = 1-1/(1+len(dom_subnet[dom_d[i]].intersection(dom_subnet[dom_d[j]])))
                if subnetw!=0:
                    dom_subnetw.append([dom_d[i],dom_d[j], subnetw])
                asnw = 1-1/(1+len(dom_asn[dom_d[i]].intersection(dom_asn[dom_d[j]])))
                if asnw!=0:
                    dom_asnw.append([dom_d[i],dom_d[j], asnw]) 
    return dom_ipw, dom_subnetw, dom_asnw
dom_ipw, dom_subnetw, dom_asnw = calcuw(edge_up)

dom_ipw = pd.DataFrame(dom_ipw, columns=['start','end','w'])
dom_subnetw = pd.DataFrame(dom_subnetw, columns=['start','end','w'])
dom_asnw = pd.DataFrame(dom_asnw, columns=['start','end','w'])

trainnode # training node
testnode # testnode

# Calculating path weights
def jisuanmal(w):
    result = 0
    if len(w)==1:
        return w[0]
    else:
        result = w[0]
        res = 0
        for i in range(1,len(w)):
            res+= (1/2**i)*w[i]
        result+= (1-result)*res
        return result

# Calculating the Shortest Path
def malcaul(dom_ipw, trainnode, testnode):
    nodes = dom_ipw['start'].to_list()+dom_ipw['end'].to_list()
    nodes = list(set(nodes))
    testnode_all = testnode[(testnode['index'].isin(nodes))]['index'].to_list()
    trainnode_bad = trainnode[(trainnode['index'].isin(nodes)) & trainnode['label']==1]['index'].to_list()

    G = nx.Graph()
    A = dom_ipw['start'].to_list()
    B = dom_ipw['end'].to_list()
    C = dom_ipw['w'].to_list()
    for i in range(dom_ipw.shape[0]):
        G.add_edge(A[i], B[i], weight=C[i])
    connected_subgraphs = list(nx.connected_components(G))

    path_w = []
    for node in testnode_all:
        w = []
        print('正在计算节点：')
        print(node)
        for nd in trainnode_bad:
            pathw = 0
            for subgraph in connected_subgraphs:
                subgraph = list(subgraph)
                if (node in subgraph) and (nd in subgraph): 
                    data = dom_ipw[(dom_ipw['start'].isin(subgraph))|(dom_ipw['end'].isin(subgraph))]
                    G = nx.Graph()
                    A = data['start'].to_list()
                    B = data['end'].to_list()
                    C = data['w'].to_list()
                    for i in range(data.shape[0]):
                        G.add_edge(A[i], B[i], weight=C[i])
                    try:
                        shortest_path = nx.shortest_path(G, node, nd)
                    except:
                        shortest_path = []
                    if len(shortest_path)>0:
                    #if path is not None:
                        path_weight = []
                        for i in range(len(shortest_path) - 1):
                            path_weight.append(G[shortest_path[i]][shortest_path[i+1]]['weight'])
                        product = reduce((lambda x, y: x * y), path_weight)
                        pathw=product
                        #print("Path:", path, "Weight:", path_weight)
            if pathw>0:
                w.append(pathw)
        if len(w)>0:
            w = sorted(w,reverse=True)
            result = jisuanmal(w)
            print(result)
            if result>0:
                path_w.append([node, result])
    return path_w
#%%
# Multi-threaded execution generates inference results

from joblib import Parallel, delayed
import math
def malcalall(dom_ipw, trainnode, testnode, batch, worker):
    pichinum = math.ceil(testnode.shape[0]/batch)
    A = []
    for i in range(pichinum-1):
        A.append(testnode.iloc[(batch*i):(batch*i+batch)])
    A.append(testnode.iloc[(batch*i+batch):(testnode.shape[0])])

    results = Parallel(n_jobs=worker)(delayed(malcaul)(dom_ipw, trainnode, test) for test in A)
    return results

results = malcalall(dom_ipw, trainnode, testnode, 50, 20)
