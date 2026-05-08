import contextlib 
import time 
import torch 
import numpy as np  
import pandas as pd 
import time 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score 
import scipy.io as sio  
import numpy as np 
from sklearn.model_selection import train_test_split  
from scipy.io import savemat  
import random  

def compute_Helmholtzians_Hodge_1_Laplacian(edge_index, B2, Nm=None, directed=True):
    """
    Compute the Hodge 1-Laplacian in Helmholtz–Hodge decomposition.
    Args:
    edge_index: edge index matrix, shape [2, num_edges]; each column is one edge (source, target).
    B2: boundary operator B2 mapping 2-simplices (triangles) to 1-simplices (edges).
    Nm: number of nodes; if None, inferred from edge_index.
    directed: whether the graph is directed (unused in this function).
    Returns:
    Hodge 1-Laplacian L and boundary operator B1.
    """
 
    edge_index = edge_index  # assign input edge_index (redundant line)
    print(edge_index.shape)  # debug: print edge_index shape
    if Nm is None:  # if node count not given
        Nm = torch.max(edge_index) + 1  # max node id + 1

    B1_row=[]  # row indices for sparse B1
    B1_column=[]  # column indices for sparse B1
    values=[]  # nonzero values for B1
    for i in range(edge_index.shape[1]):  # each column is one edge
        B1_row.extend([edge_index[0, i],edge_index[1, i]])  # source and target node ids
        B1_column.extend([i,i])  # edge column index twice
        values.extend([-1,1])  # incidence signs: -1 at source, +1 at target
    B1_indices = torch.tensor([B1_row, B1_column], dtype=torch.long)  # COO indices
    B1_values = torch.tensor(values, dtype=torch.float)  # COO values
    B1_shape = torch.Size([Nm, edge_index.shape[1]])  # B1 shape [num_nodes, num_edges]

    B1= torch.sparse_coo_tensor(B1_indices, B1_values, B1_shape)  # sparse boundary B1: nodes -> edges

    print(B1.shape, B2.shape)  # debug: B1 and B2 shapes


    return torch.sparse.mm(B1.permute(1, 0), B1) + torch.sparse.mm(B2, B2.permute(1, 0)),B1  # L = B1^T B1 + B2 B2^T; return L and B1


if __name__ == "__main__":  # main entry: run when executed as script
    datasets=['soc-sign-bitcoinalpha','soc-sign-bitcoinotc','Slashdot','epinions']  # dataset names to test
  
    M=[50,100,150,200,250,300]  # embedding dimensions M to try
   
    for k in range(len(datasets)):  # loop over datasets
        AUCs,macro_f1s,F1s,Accuracys, Costs = np.zeros((5,5,6)) ,np.zeros((5,5,6)) , np.zeros((5,5,6)), np.zeros((5,5,6)), np.zeros((5,5,6))  # result arrays: 5 runs x 5 time steps x 6 M values
        for turn in range(5):  # 5 runs per dataset for stability / averaging
            data_name=datasets[k]  # current dataset name
            data = pd.read_csv(f"./data/{data_name}.csv",header=None)  # load CSV; header=None: no header row
            if data_name in ['soc-sign-bitcoinalpha','soc-sign-bitcoinotc']:  # these two have timestamp column
                data.columns = ['SOURCE', 'TARGET', 'RATING','t']  # source, target, rating, time
            else:  # other datasets
                data.columns = ['SOURCE', 'TARGET', 'RATING']  # source, target, rating (no time)
            num_edges = len(data)  # number of rows = edges
            edges_index = {}  # map (u,v) -> original row index
            nodes = set(data['SOURCE']) | set(data['TARGET'])  # union of endpoints
            node_dict = {node: index for index, node in enumerate(nodes)}  # node id -> 0..n-1
            num_nodes = len(nodes)  # node count
            classes=[]  # sign label per kept edge (+1 / -1)
            graph = {}  # adjacency: node -> set of neighbors
            pos_in_degree = torch.zeros(num_nodes, 1, dtype=torch.float)  # positive in-degree counts
            neg_in_degree = torch.zeros(num_nodes, 1, dtype=torch.float)  # negative in-degree counts
            pos_out_degree = torch.zeros(num_nodes, 1, dtype=torch.float)  # positive out-degree counts
            neg_out_degree = torch.zeros(num_nodes, 1, dtype=torch.float)  # negative out-degree counts
           
            random_numbers = np.random.randint(1, 201, size=20)  # unused random ints 1..200
    
            for j in range(len(M)):  # loop over embedding dim M
                m=M[j]  # current M
                start=time.time()  # wall-clock start
                for index, row in data.iterrows():  # each row = one edge
                    source = row['SOURCE']  # source id
                    target = row['TARGET']  # target id
                    rating = row['RATING']  # rating value
                    #category = row['RATING']  # commented: use rating as class directly
                    if rating > 0:  # positive rating
                        category = 1  # positive edge
                    else :  # non-positive
                        category = -1  # negative edge
                
                    source_index = node_dict[source]  # remap source to index
                    target_index = node_dict[target]  # remap target to index
                    if source_index not in graph:  # init adjacency for source
                        graph[source_index] = set()  # empty neighbor set
                    graph[source_index].add(target_index)  # undirected: add v to u's neighbors
                    if target_index not in graph:  # init adjacency for target
                        graph[target_index] = set()  # empty neighbor set
                    graph[target_index].add(source_index)  # undirected: add u to v's neighbors
                    if (source_index, target_index) in edges_index:  # duplicate directed edge
                        # print(f"Duplicate edge found: ({source_index}, {target_index})")  # commented debug
                        continue  # skip duplicate
                    else:  # new edge
                        edges_index[(source_index, target_index)] = index  # map to CSV row index
                       
                    if category == 1:  # positive edge
                        pos_out_degree[source_index] += 1  # ++ positive out at source
                        pos_in_degree[target_index] += 1  # ++ positive in at target
                    else:  # negative edge
                        neg_out_degree[source_index] += 1  # ++ negative out at source
                        neg_in_degree[target_index] += 1  # ++ negative in at target
                    classes.append(category)  # append label for this edge
                edge_index = torch.from_numpy(np.array(list(edges_index.keys())))  # stack edge pairs as tensor
                triangles = set()  # discovered triangles
                triangle_num=0  # triangle id counter
                B2_row=[]  # B2 row indices (edge ids)
                B2_column=[]  # B2 col indices (triangle ids)
                values=[]  # B2 nonzero values
            
                for u in range(num_nodes):  # each node u
                    neighbors_u = graph[u]  # neighbors of u
            
                    for v in neighbors_u:  # neighbor v
                        if u < v:  # list each triangle once
                            neighbors_v = graph[v]  # neighbors of v
                            common_neighbors = neighbors_u.intersection(neighbors_v)  # shared neighbors of u,v
                            for w in common_neighbors:  # candidate third vertex
                                if u < v < w:  # canonical order
                                    
                                    triangles.add((u, v, w))  # add triangle
                                    edge_index_uv = edges_index.get((u, v), None)  # edge (u,v) row id
                                    if edge_index_uv is None:  # try reverse orientation
                                        edge_index_uv = edges_index.get((v, u), None)  # (v,u)
                                        if edge_index_uv is None:  # missing edge
                                         
                                            continue  # skip triangle

                                    edge_index_uw = edges_index.get((u, w), None)  # edge (u,w)
                                    if edge_index_uw is None:  # try reverse
                                        edge_index_uw = edges_index.get((w, u), None)  # (w,u)
                                        if edge_index_uw is None:  # missing
                                            continue  # skip
                                    edge_index_vw = edges_index.get((v, w), None)  # edge (v,w)
                                    if edge_index_vw is None:  # try reverse
                                        edge_index_vw = edges_index.get((w, v), None)  # (w,v)
                                        if edge_index_vw is None:  # missing
                                            continue  # skip

                                    B2_row.extend([edge_index_uv, edge_index_uw, edge_index_vw])  # three edges of triangle
                                    B2_column.extend([triangle_num,triangle_num,triangle_num])  # same triangle column
                                    values.extend([1,-1,1])  # boundary coeffs for (u,v),(u,w),(v,w)
                                    triangle_num = triangle_num + 1  # next triangle id
                B2_indices = torch.tensor([B2_row, B2_column], dtype=torch.long)  # B2 COO indices
                B2_values = torch.tensor(values, dtype=torch.float)  # B2 values
                B2_shape = torch.Size([num_edges, triangle_num])  # B2 shape [num_edges, num_triangles]
                B2= torch.sparse_coo_tensor(B2_indices, B2_values, B2_shape)  # sparse B2: triangles -> edges        
                L,B1=compute_Helmholtzians_Hodge_1_Laplacian(edge_index.T,B2,Nm=num_nodes,directed=True)  # L, B1; .T -> [2, num_edges]
                node_feature_tensor = torch.cat((pos_in_degree, neg_in_degree, pos_out_degree, neg_out_degree), dim=1)  # node feats [N, 4]
                edge_feature_tensor = torch.zeros((num_edges, 8), dtype=torch.float)  # edge feats [E, 8] = concat two node feats
                for i in range(num_edges):  # each edge row
                    u, v = edge_index[i]  # endpoints
              
                    edge_feature_tensor[i] = torch.cat((node_feature_tensor[u],node_feature_tensor[v]),dim=0)  # 4+4=8

                triange_feature_tensor = torch.zeros((len(triangles), 12), dtype=torch.float)  # triangle feats [T, 12] = 3x4
            
                for i, (u, v, w) in enumerate(triangles):  # each triangle
                
                    triange_feature_tensor[i] = torch.cat((node_feature_tensor[u], node_feature_tensor[v], node_feature_tensor[w]), dim=0)  # 4+4+4=12
         
                for t in range(1,2):  # time steps (only t=1 here)
                    
                    start_time = time.time()  # propagation start time
                    for i in range(t):  # i propagation steps (here 1)
                        if i == 0:  # first step
                            edge_feature_tensor_tmp = edge_feature_tensor.clone()  # copy edge feats
                            node_feature_tensor_tmp = node_feature_tensor.clone()  # copy node feats
                            triange_feature_tensor_tmp = triange_feature_tensor.clone()  # copy triangle feats
                        W1 = torch.randn((node_feature_tensor_tmp.size(1), m), dtype=torch.float)  # random W1: node_dim -> m
                        W2 = torch.randn((edge_feature_tensor_tmp.size(1), m), dtype=torch.float)  # random W2: edge_dim -> m
                        W3 = torch.randn((triange_feature_tensor_tmp.size(1), m), dtype=torch.float)  # random W3: tri_dim -> m

                        node_feature_tensor_tmp = (torch.mm(node_feature_tensor_tmp, W1)>0).float()  # ReLU-like binarize node embed
                        triange_feature_tensor_tmp =  (torch.mm(triange_feature_tensor_tmp, W3)>0).float()  # ReLU-like triangle embed
                        node2edge = torch.sparse.mm(B1.t(),node_feature_tensor_tmp)  # node -> edge via B1^T
                        edge_feature_tensor_tmp = (torch.sparse.mm(L, torch.mm(edge_feature_tensor_tmp, W2))>0).float()  # Laplacian on edge embed + ReLU
                        triangle2edge = torch.mm(B2,triange_feature_tensor_tmp)  # triangle -> edge via B2
            
                        edge_feature_tensor_tmp = ((node2edge + edge_feature_tensor_tmp + triangle2edge)>0).float()  # fuse + ReLU
                        embeds=edge_feature_tensor_tmp.numpy()  # edge embeddings numpy
                        
                    negated_array = (1 - embeds)  # bitwise complement for augmented features
                    
                    conta= np.hstack((embeds, negated_array))  # concat embed and negated (2x dim)
                    train_X, test_X, train_Y, test_Y = train_test_split(conta, classes, test_size=0.2)  # 80/20 split
                    train_time=time.time()-start  # elapsed since load (unused)
                    clf = LogisticRegression()  # logistic regression
                    clf.fit(train_X, train_Y)  # fit
                    y_pred = clf.predict(test_X)  # predict
                    macro_f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')  # macro F1
                    binary_f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='binary')  # binary F1
                    accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)  # accuracy
                    auc = roc_auc_score(test_Y, clf.predict_proba(test_X)[:, 1])  # AUC positive class
                   
                    cost = time.time() - start_time  # time for propagation + train block 
                    F1s[turn,t-1,j]=binary_f1  # store binary F1 [run, time-1, M idx]
                    macro_f1s[turn,t-1,j]=macro_f1  # store macro F1
                    Accuracys[turn,t-1,j]=accuracy  # store accuracy
                    AUCs[turn,t-1,j]=auc  # store AUC
                    Costs[turn,t-1,j] = cost  # store runtime
                    print(data_name,auc,macro_f1,binary_f1,accuracy)  # print metrics
  
                    # save_path = f'/data/disk2/xuantan/S3KECTCH copy/result/{data_name}.mat'  # old path (commented)
                    # sio.savemat(save_path, {'embeds': conta, 'classes': classes})  # commented: save .mat
                
        results_dict = {  # bundle all metrics for .mat export
        'F1s': F1s,  # binary F1 over runs (3D)
        "AUCS":AUCs,  # AUC over runs (3D)
        "macro_f1s":macro_f1s,  # macro F1 over runs (3D)
        'Accuracys': Accuracys,  # accuracy over runs (3D)
        'Costs': Costs,  # runtime over runs (3D)
        'f1_mean': np.mean(F1s,axis=0),  # mean over runs axis 0
        'accuracy_mean':np.mean(Accuracys,axis=0),  # mean accuracy
        'cost_mean':np.mean(Costs,axis=0),  # mean time
        'auc_mean': np.mean(AUCs,axis=0),  # mean AUC
        'macro_mean':np.mean(macro_f1s,axis=0),  # mean macro F1
        
    }
        savemat(f'./result/{data_name}.mat', results_dict)  # write results_dict to ./result/<dataset>.mat
                
