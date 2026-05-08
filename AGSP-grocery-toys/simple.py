import contextlib  # Context managers (unused in this script)
import time  # Timing
import torch  # Tensors and sparse ops
import numpy as np  # Arrays
import pandas as pd  # CSV / tables
import time  # duplicate import (harmless)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import scipy.io as sio  # MATLAB I/O (unused)
import numpy as np  # duplicate import (harmless)
from sklearn.model_selection import train_test_split  # duplicate (harmless)
from scipy.io import savemat
import random  # unused

def compute_Helmholtzians_Hodge_1_Laplacian(edge_index, B2, Nm=None, directed=True):
    """
    Build the Hodge 1-Laplacian used in Helmholtz–Hodge decomposition.

    Args:
        edge_index: Edge indices, shape [2, num_edges]; column = one edge (src, dst).
        B2: Boundary operator mapping 2-simplices (triangles) to 1-simplices (edges).
        Nm: Number of nodes; if None, inferred from edge_index.
        directed: Directed flag (unused in this function).

    Returns:
        Hodge 1-Laplacian L and boundary operator B1.
    """
 
    edge_index = edge_index
    print(edge_index.shape)
    if Nm is None:
        Nm = torch.max(edge_index) + 1

    B1_row=[]
    B1_column=[]
    values=[]
    for i in range(edge_index.shape[1]):
        B1_row.extend([edge_index[0, i],edge_index[1, i]])
        B1_column.extend([i,i])
        values.extend([-1,1])
    B1_indices = torch.tensor([B1_row, B1_column], dtype=torch.long)
    B1_values = torch.tensor(values, dtype=torch.float)
    B1_shape = torch.Size([Nm, edge_index.shape[1]])

    B1= torch.sparse_coo_tensor(B1_indices, B1_values, B1_shape)

    print(B1.shape, B2.shape)


    return torch.sparse.mm(B1.permute(1, 0), B1) + torch.sparse.mm(B2, B2.permute(1, 0)),B1  # L = B1^T B1 + B2 B2^T


if __name__ == "__main__":
    datasets=['soc-sign-bitcoinalpha','soc-sign-bitcoinotc','Slashdot','epinions']
  
    M=[50,100,150,200,250,300]
   
    for k in range(len(datasets)):
        AUCs,macro_f1s,F1s,Accuracys, Costs = np.zeros((5,5,6)) ,np.zeros((5,5,6)) , np.zeros((5,5,6)), np.zeros((5,5,6)), np.zeros((5,5,6))  # 5 runs x 5 time steps x 6 embedding dims
        for turn in range(5):
            data_name=datasets[k]
            data = pd.read_csv(f"./data/{data_name}.csv",header=None)
            if data_name in ['soc-sign-bitcoinalpha','soc-sign-bitcoinotc']:
                data.columns = ['SOURCE', 'TARGET', 'RATING','t']
            else:
                data.columns = ['SOURCE', 'TARGET', 'RATING']
            num_edges = len(data)
            edges_index = {}
            nodes = set(data['SOURCE']) | set(data['TARGET'])
            node_dict = {node: index for index, node in enumerate(nodes)}
            num_nodes = len(nodes)
            classes=[]
            graph = {}
            pos_in_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
            neg_in_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
            pos_out_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
            neg_out_degree = torch.zeros(num_nodes, 1, dtype=torch.float)
           
            random_numbers = np.random.randint(1, 201, size=20)  # unused
    
            for j in range(len(M)):
                m=M[j]
                start=time.time()
                for index, row in data.iterrows():
                    source = row['SOURCE']
                    target = row['TARGET']
                    rating = row['RATING']
                    # category = row['RATING']  # could use rating directly as label
                    if rating > 0:
                        category = 1
                    else :
                        category = -1
                
                    source_index = node_dict[source]
                    target_index = node_dict[target]
                    if source_index not in graph:
                        graph[source_index] = set()
                    graph[source_index].add(target_index)
                    if target_index not in graph:
                        graph[target_index] = set()
                    graph[target_index].add(source_index)
                    if (source_index, target_index) in edges_index:
                        # print(f"Duplicate edge found: ({source_index}, {target_index})")
                        continue
                    else:
                        edges_index[(source_index, target_index)] = index
                       
                    if category == 1:
                        pos_out_degree[source_index] += 1
                        pos_in_degree[target_index] += 1
                    else:
                        neg_out_degree[source_index] += 1
                        neg_in_degree[target_index] += 1
                    classes.append(category)
                edge_index = torch.from_numpy(np.array(list(edges_index.keys())))
                triangles = set()
                triangle_num=0
                B2_row=[]
                B2_column=[]
                values=[]
            
                for u in range(num_nodes):
                    neighbors_u = graph[u]
            
                    for v in neighbors_u:
                        if u < v:
                            neighbors_v = graph[v]
                            common_neighbors = neighbors_u.intersection(neighbors_v)
                            for w in common_neighbors:
                                if u < v < w:
                                    
                                    triangles.add((u, v, w))
                                    edge_index_uv = edges_index.get((u, v), None)
                                    if edge_index_uv is None:
                                        edge_index_uv = edges_index.get((v, u), None)
                                        if edge_index_uv is None:
                                         
                                            continue

                                    edge_index_uw = edges_index.get((u, w), None)
                                    if edge_index_uw is None:
                                        edge_index_uw = edges_index.get((w, u), None)
                                        if edge_index_uw is None:
                                            continue
                                    edge_index_vw = edges_index.get((v, w), None)
                                    if edge_index_vw is None:
                                        edge_index_vw = edges_index.get((w, v), None)
                                        if edge_index_vw is None:
                                            continue

                                    B2_row.extend([edge_index_uv, edge_index_uw, edge_index_vw])
                                    B2_column.extend([triangle_num,triangle_num,triangle_num])
                                    values.extend([1,-1,1])
                                    triangle_num = triangle_num + 1
                B2_indices = torch.tensor([B2_row, B2_column], dtype=torch.long)
                B2_values = torch.tensor(values, dtype=torch.float)
                B2_shape = torch.Size([num_edges, triangle_num])
                B2= torch.sparse_coo_tensor(B2_indices, B2_values, B2_shape)
                L,B1=compute_Helmholtzians_Hodge_1_Laplacian(edge_index.T,B2,Nm=num_nodes,directed=True)
                node_feature_tensor = torch.cat((pos_in_degree, neg_in_degree, pos_out_degree, neg_out_degree), dim=1)
                edge_feature_tensor = torch.zeros((num_edges, 8), dtype=torch.float)
                for i in range(num_edges):
                    u, v = edge_index[i]
              
                    edge_feature_tensor[i] = torch.cat((node_feature_tensor[u],node_feature_tensor[v]),dim=0)

                triange_feature_tensor = torch.zeros((len(triangles), 12), dtype=torch.float)
            
                for i, (u, v, w) in enumerate(triangles):
                
                    triange_feature_tensor[i] = torch.cat((node_feature_tensor[u], node_feature_tensor[v], node_feature_tensor[w]), dim=0)
         
                for t in range(1,2):
                    
                    start_time = time.time()
                    for i in range(t):
                        if i == 0:
                            edge_feature_tensor_tmp = edge_feature_tensor.clone()
                            node_feature_tensor_tmp = node_feature_tensor.clone()
                            triange_feature_tensor_tmp = triange_feature_tensor.clone()
                        W1 = torch.randn((node_feature_tensor_tmp.size(1), m), dtype=torch.float)
                        W2 = torch.randn((edge_feature_tensor_tmp.size(1), m), dtype=torch.float)
                        W3 = torch.randn((triange_feature_tensor_tmp.size(1), m), dtype=torch.float)

                        node_feature_tensor_tmp = (torch.mm(node_feature_tensor_tmp, W1)>0).float()
                        triange_feature_tensor_tmp =  (torch.mm(triange_feature_tensor_tmp, W3)>0).float()
                        node2edge = torch.sparse.mm(B1.t(),node_feature_tensor_tmp)
                        edge_feature_tensor_tmp = (torch.sparse.mm(L, torch.mm(edge_feature_tensor_tmp, W2))>0).float()
                        triangle2edge = torch.mm(B2,triange_feature_tensor_tmp)
            
                        edge_feature_tensor_tmp = ((node2edge + edge_feature_tensor_tmp + triangle2edge)>0).float()
                        embeds=edge_feature_tensor_tmp.numpy()
                        
                    negated_array = (1 - embeds)
                    
                    conta= np.hstack((embeds, negated_array))
                    train_X, test_X, train_Y, test_Y = train_test_split(conta, classes, test_size=0.2)
                    train_time=time.time()-start  # unused
                    clf = LogisticRegression()
                    clf.fit(train_X, train_Y)
                    y_pred = clf.predict(test_X)
                    macro_f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
                    binary_f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='binary')
                    accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)
                    auc = roc_auc_score(test_Y, clf.predict_proba(test_X)[:, 1])
                   
                    cost = time.time() - start_time
                    F1s[turn,t-1,j]=binary_f1
                    macro_f1s[turn,t-1,j]=macro_f1
                    Accuracys[turn,t-1,j]=accuracy
                    AUCs[turn,t-1,j]=auc
                    Costs[turn,t-1,j] = cost
                    print(data_name,auc,macro_f1,binary_f1,accuracy)
  
                    # save_path = f'/data/disk2/xuantan/S3KECTCH copy/result/{data_name}.mat'
                    # sio.savemat(save_path, {'embeds': conta, 'classes': classes})
        results_dict = {
        'F1s': F1s,
        "AUCS":AUCs,
        "macro_f1s":macro_f1s,
        'Accuracys': Accuracys,
        'Costs': Costs,
        'f1_mean': np.mean(F1s,axis=0),
        'accuracy_mean':np.mean(Accuracys,axis=0),
        'cost_mean':np.mean(Costs,axis=0),
        'auc_mean': np.mean(AUCs,axis=0),
        'macro_mean':np.mean(macro_f1s,axis=0),
        
    }
        savemat(f'./result/{data_name}.mat', results_dict)
