import os
import sys
import copy
# Set before importing torch to reduce CUDA fragmentation and help avoid OOM
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import argparse
import numpy as np
import time
import warnings
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning
from model import SimplicialGNN, build_node_to_triangles_dict, compute_loss, evaluate
from magb_data_loader import load_magb_grocery_data


def calculate_metrics(y_true, y_pred, y_proba=None, num_classes=None):
    """
    Compute classification metrics.

    Args:
        y_true: Ground-truth labels (numpy or torch).
        y_pred: Predicted labels (numpy or torch).
        y_proba: Probabilities or logits (numpy or torch, shape [n_samples, n_classes]).
        num_classes: Number of classes.

    Returns:
        dict with accuracy, precision, recall, f1, auc.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    metrics = {}
    
    accuracy = float((y_true == y_pred).mean())
    metrics['accuracy'] = accuracy
    
    precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    if y_proba is not None:
        try:
            if isinstance(y_proba, torch.Tensor):
                y_proba_np = y_proba.cpu().numpy()
            else:
                y_proba_np = np.array(y_proba).copy()
            
            if y_proba_np.size == 0:
                metrics['auc'] = 0.0
                return metrics
            
            if len(y_proba_np.shape) != 2:
                raise ValueError(f"y_proba must be 2D [n_samples, n_classes], got shape {y_proba_np.shape}")
            
            if len(y_true) != y_proba_np.shape[0]:
                raise ValueError(f"y_true length ({len(y_true)}) != y_proba rows ({y_proba_np.shape[0]})")
            
            if np.any(np.isnan(y_proba_np)) or np.any(np.isinf(y_proba_np)):
                raise ValueError("y_proba contains NaN or Inf; cannot compute AUC")
            
            if num_classes is None:
                num_classes = y_proba_np.shape[1]
            
            if y_proba_np.shape[1] != num_classes:
                raise ValueError(f"y_proba columns ({y_proba_np.shape[1]}) != num_classes ({num_classes})")
            
            row_sums = np.sum(y_proba_np, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                exp_scores = np.exp(y_proba_np - np.max(y_proba_np, axis=1, keepdims=True))
                y_proba_np = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                if np.any(np.isnan(y_proba_np)):
                    raise ValueError("NaN after softmax on y_proba; cannot compute AUC")
            
            unique_labels = np.unique(y_true)
            if len(unique_labels) == 0:
                metrics['auc'] = 0.0
                return metrics
            
            min_label = np.min(unique_labels)
            max_label = np.max(unique_labels)
            
            if min_label < 0 or max_label >= num_classes:
                raise ValueError(f"Labels in [{min_label}, {max_label}] outside [0, {num_classes-1}]")
            
            if len(unique_labels) < 2:
                metrics['auc'] = 0.0
                return metrics
            
            if num_classes == 2:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                    auc = float(roc_auc_score(y_true, y_proba_np[:, 1]))
            else:
                existing_classes = unique_labels
                
                if len(existing_classes) < num_classes:
                    y_true_filtered_binarized = label_binarize(y_true, classes=existing_classes)
                    y_proba_filtered = y_proba_np[:, existing_classes]
                    
                    proba_sums = np.sum(y_proba_filtered, axis=1, keepdims=True)
                    proba_sums = np.where(proba_sums == 0, 1.0, proba_sums)
                    y_proba_filtered = y_proba_filtered / proba_sums
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                        try:
                            auc = float(roc_auc_score(y_true_filtered_binarized, y_proba_filtered, 
                                                     multi_class='ovr', average='weighted'))
                        except ValueError as e:
                            try:
                                auc = float(roc_auc_score(y_true_filtered_binarized, y_proba_filtered, 
                                                         multi_class='ovr', average='macro'))
                            except ValueError as e2:
                                print(f"Warning: multiclass AUC failed (classes present: {existing_classes}). Error: {e2}")
                                auc = 0.0
                else:
                    y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                        try:
                            auc = float(roc_auc_score(y_true_binarized, y_proba_np, 
                                                     multi_class='ovr', average='macro'))
                        except ValueError as e:
                            try:
                                auc = float(roc_auc_score(y_true_binarized, y_proba_np, 
                                                         multi_class='ovr', average='weighted'))
                            except ValueError as e2:
                                print(f"Warning: multiclass AUC failed. Error: {e2}")
                                auc = 0.0
            
            if np.isnan(auc) or np.isinf(auc):
                print(f"Warning: invalid AUC {auc}, setting to 0.0")
                auc = 0.0
            
            metrics['auc'] = auc
        except Exception as e:
            print(f"Warning: could not compute AUC: {e}")
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics


def log_label_distribution(labels, train_idx, val_idx, test_idx, num_classes, name="dataset"):
    """
    Print per-split label counts and ratios for class imbalance analysis.
    labels and *_idx may be numpy or torch tensors.
    """
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels).astype(np.int64)
    train_idx = np.asarray(train_idx) if hasattr(train_idx, 'cpu') else np.asarray(train_idx)
    val_idx = np.asarray(val_idx) if hasattr(val_idx, 'cpu') else np.asarray(val_idx)
    test_idx = np.asarray(test_idx) if hasattr(test_idx, 'cpu') else np.asarray(test_idx)

    def _count_and_ratio(idx):
        if len(idx) == 0:
            return {}, {}
        y = labels[idx]
        cnt = {}
        for c in range(num_classes):
            cnt[c] = int((y == c).sum())
        total = len(y)
        ratio = {c: cnt[c] / total for c in range(num_classes)}
        return cnt, ratio

    print(f"\n[{name}] label distribution ({num_classes} classes)")
    print("=" * 70)
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        cnt, ratio = _count_and_ratio(idx)
        total = sum(cnt.values())
        print(f"  {split_name}: total samples = {total}")
        for c in range(num_classes):
            pct = ratio.get(c, 0) * 100
            print(f"    class {c:3d}: count = {cnt.get(c, 0):6d}, pct = {pct:6.2f}%")
        if total > 0:
            max_c = max(range(num_classes), key=lambda k: cnt.get(k, 0))
            min_c = min(range(num_classes), key=lambda k: cnt.get(k, 0) if cnt.get(k, 0) > 0 else float('inf'))
            print(f"    (max: class {max_c} n={cnt.get(max_c, 0)}, min: class {min_c} n={cnt.get(min_c, 0)})")
    print("=" * 70)


def log_per_class_metrics(y_true, y_pred, num_classes, logits=None):
    """
    Per-class test counts, per-class accuracy, and predicted label histogram.
    y_true/y_pred are numpy or torch; logits reserved for extensions.
    """
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    print("\n[test] per-class metrics and prediction histogram")
    print("=" * 70)
    for c in range(num_classes):
        mask = y_true == c
        n = mask.sum()
        if n == 0:
            print(f"  class {c:3d}: no test samples")
            continue
        correct = (y_pred[mask] == c).sum()
        acc_c = correct / n
        print(f"  class {c:3d}: n = {n:6d}, correct = {correct:6d}, acc = {acc_c:.4f}")

    pred_counts = {}
    for c in range(num_classes):
        pred_counts[c] = int((y_pred == c).sum())
    print("  predicted label counts:")
    for c in range(num_classes):
        print(f"    pred class {c:3d}: {pred_counts[c]:6d}")
    print("=" * 70)


def load_raw_dataset(
    dataset='grocery',
    device=None,
    data_path=None,
    feat_path=None,
    train_ratio=0.6,
    val_ratio=0.2,
):
    """
    Load raw MAGB-format data (Grocery / Toys).

    Args:
        dataset (str): 'grocery' or 'toys'; default 'grocery'.
        device: torch.device for tensors, or None (CPU).
        data_path (str): If set, overrides dataset and uses this directory.
        feat_path (str): Optional merged feature file (preferred if set).
        train_ratio (float): Train fraction; default 0.6.
        val_ratio (float): Val fraction; default 0.2.

    Returns:
        dict: edges, labels, splits, v_feat, t_feat, num_nodes, num_classes, data_path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')

    if data_path is not None:
        DATA_PATH = os.path.abspath(data_path)
        print(f"Using data path: {DATA_PATH}")
    else:
        if dataset == 'toys':
            DATA_PATH = os.path.join(project_root, "Data", "Toys")
        else:
            DATA_PATH = os.path.join(project_root, "Data", "Grocery")
        DATA_PATH = os.path.abspath(DATA_PATH)
        print(f"Dataset: {dataset}, data path: {DATA_PATH}")

    if feat_path is not None and not os.path.isabs(feat_path) and not os.path.exists(feat_path):
        feat_path = os.path.join(DATA_PATH, feat_path)

    edges, labels, splits, v_feat, t_feat, num_classes = load_magb_grocery_data(
        DATA_PATH,
        feat_path=feat_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        fewshots=None,
        undirected=True,
    )
    num_nodes = labels.shape[0]

    if device is not None:
        v_feat = v_feat.to(device)
        t_feat = t_feat.to(device)

    print(f"num_nodes: {num_nodes}, num_classes: {num_classes}")
    print("Raw data loaded.")

    return {
        'edges': edges,
        'labels': labels,
        'splits': splits,
        'v_feat': v_feat,
        't_feat': t_feat,
        'num_nodes': num_nodes,
        'num_classes': num_classes,
        'data_path': DATA_PATH,
    }



def build_triangles_and_B2(edges, num_nodes):
    """
    Build triangle set and B2 boundary operator from edge data.

    Args:
        edges: torch.Tensor [num_edges, 2], numpy array, or list of (u, v).
        num_nodes: Number of nodes.

    Returns:
        dict with 'triangles', 'B2', 'edges_index', 'num_edges', 'num_triangles',
        'edge_index_tensor'.

    Example:
        >>> edges = torch.tensor([[0, 1], [1, 2], [0, 2]])
        >>> num_nodes = 3
        >>> result = build_triangles_and_B2(edges, num_nodes)
        >>> print(result['num_triangles'])  # 1
        >>> print(result['B2'].shape)  # [3, 1]
    """
    
    if isinstance(edges, list):
        if len(edges) > 0 and isinstance(edges[0], tuple):
            edges_array = np.array(edges)
            edge_index_tensor = torch.from_numpy(edges_array)
        else:
            edge_index_tensor = torch.tensor(edges)
    elif isinstance(edges, np.ndarray):
        edge_index_tensor = torch.from_numpy(edges)
    elif isinstance(edges, torch.Tensor):
        edge_index_tensor = edges.clone()
    else:
        raise TypeError(f"Unsupported edge type: {type(edges)}")
    
    if edge_index_tensor.dim() != 2 or edge_index_tensor.shape[1] != 2:
        raise ValueError(f"Edges must be [num_edges, 2], got {edge_index_tensor.shape}")
    
    num_edges = edge_index_tensor.shape[0]
    
    edge_np = edge_index_tensor.cpu().numpy()
    graph = defaultdict(set)
    edges_index = {}
    for i in range(num_edges):
        u, v = int(edge_np[i, 0]), int(edge_np[i, 1])
        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            continue
        graph[u].add(v)
        graph[v].add(u)
        key = (min(u, v), max(u, v))
        if key not in edges_index:
            edges_index[key] = i
    
    triangles = set()
    triangle_num = 0
    B2_row = []
    B2_column = []
    values = []
    
    for u in range(num_nodes):
        if u not in graph:
            continue
            
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
                            triangles.discard((u, v, w))
                            continue
                        
                        edge_index_uw = edges_index.get((u, w), None)
                        if edge_index_uw is None:
                            edge_index_uw = edges_index.get((w, u), None)
                        if edge_index_uw is None:
                            triangles.discard((u, v, w))
                            continue
                        
                        edge_index_vw = edges_index.get((v, w), None)
                        if edge_index_vw is None:
                            edge_index_vw = edges_index.get((w, v), None)
                        if edge_index_vw is None:
                            triangles.discard((u, v, w))
                            continue
                        
                        B2_row.extend([edge_index_uv, edge_index_uw, edge_index_vw])
                        B2_column.extend([triangle_num, triangle_num, triangle_num])
                        values.extend([1, -1, 1])
                        triangle_num += 1
    
    num_triangles = len(triangles)
    if num_triangles == 0:
        B2 = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float),
            torch.Size([num_edges, 0])
        )
    else:
        B2_indices = torch.tensor([B2_row, B2_column], dtype=torch.long)
        B2_values = torch.tensor(values, dtype=torch.float)
        B2_shape = torch.Size([num_edges, num_triangles])
        B2 = torch.sparse_coo_tensor(B2_indices, B2_values, B2_shape)
        B2 = B2.coalesce()
    
    return {
        'triangles': triangles,
        'B2': B2,
        'edges_index': edges_index,
        'num_edges': num_edges,
        'num_triangles': num_triangles,
        'edge_index_tensor': edge_index_tensor
    }


def extract_k_hop_subgraph(center_nodes, edge_index, num_nodes, k=2, return_node_mapping=True):
    """
    Extract a k-hop subgraph around center nodes to save memory.

    Args:
        center_nodes: Center node indices (often train nodes).
        edge_index: [num_edges, 2].
        num_nodes: Total nodes in the full graph.
        k: Number of hops; default 2 (typical for 2-layer GNN).
        return_node_mapping: If True, include original->local index map.

    Returns:
        dict with subgraph_nodes, subgraph_edge_index, reverse_mapping,
        num_subgraph_nodes, num_subgraph_edges, and optionally node_mapping.
    """
    device = edge_index.device
    
    if not isinstance(center_nodes, torch.Tensor):
        if isinstance(center_nodes, (list, np.ndarray)):
            center_nodes = torch.tensor(center_nodes, dtype=torch.long, device=device)
        else:
            center_nodes = torch.tensor([center_nodes], dtype=torch.long, device=device)
    else:
        center_nodes = center_nodes.to(device)
    
    center_nodes = center_nodes.unique()
    
    subgraph_nodes_set = set(center_nodes.cpu().tolist())
    
    current_nodes = center_nodes.clone()
    for hop in range(k):
        mask_forward = torch.isin(edge_index[:, 0], current_nodes)
        neighbors_forward = edge_index[mask_forward, 1].unique()
        
        mask_backward = torch.isin(edge_index[:, 1], current_nodes)
        neighbors_backward = edge_index[mask_backward, 0].unique()
        
        all_neighbors = torch.cat([neighbors_forward, neighbors_backward]).unique()
        
        new_neighbors = all_neighbors[~torch.isin(all_neighbors, torch.tensor(list(subgraph_nodes_set), device=device))]
        
        if len(new_neighbors) == 0:
            break
        
        subgraph_nodes_set.update(new_neighbors.cpu().tolist())
        current_nodes = new_neighbors
    
    subgraph_nodes_list = sorted(list(subgraph_nodes_set))
    subgraph_nodes = torch.tensor(subgraph_nodes_list, dtype=torch.long, device=device)
    num_subgraph_nodes = len(subgraph_nodes_list)
    
    node_mapping = {orig_idx: subgraph_idx for subgraph_idx, orig_idx in enumerate(subgraph_nodes_list)}
    reverse_mapping = {subgraph_idx: orig_idx for orig_idx, subgraph_idx in node_mapping.items()}
    
    subgraph_tensor = torch.tensor(subgraph_nodes_list, dtype=torch.long, device=device)
    edge_mask = torch.isin(edge_index[:, 0], subgraph_tensor) & torch.isin(edge_index[:, 1], subgraph_tensor)
    subgraph_edge_index_raw = edge_index[edge_mask]
    num_subgraph_edges = subgraph_edge_index_raw.shape[0]
    
    mapping_tensor = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    mapping_tensor[subgraph_nodes] = torch.arange(num_subgraph_nodes, dtype=torch.long, device=device)
    subgraph_edge_index = mapping_tensor[subgraph_edge_index_raw].clone()
    
    result = {
        'subgraph_nodes': subgraph_nodes,
        'subgraph_edge_index': subgraph_edge_index,
        'reverse_mapping': reverse_mapping,
        'num_subgraph_nodes': num_subgraph_nodes,
        'num_subgraph_edges': num_subgraph_edges
    }
    
    if return_node_mapping:
        result['node_mapping'] = node_mapping
    
    return result


def compute_0_simplex_features(v_feat, t_feat):
    """
    0-simplex (node) features: concatenate text and vision along dim=1.

    Args:
        v_feat: [num_nodes, v_feat_dim]
        t_feat: [num_nodes, t_feat_dim]

    Returns:
        h_0: [num_nodes, v_feat_dim + t_feat_dim]
    """
    h_0 = torch.cat([t_feat, v_feat], dim=1)
    return h_0


def compute_1_simplex_features(h_0, edge_index_tensor):
    """
    1-simplex (edge) features: Hadamard product of endpoint features, h_e = h_u * h_v.

    Args:
        h_0: [num_nodes, feat_dim]
        edge_index_tensor: [num_edges, 2], rows [u, v].

    Returns:
        h_1: [num_edges, feat_dim]
    """
    num_edges = edge_index_tensor.shape[0]
    
    u_indices = edge_index_tensor[:, 0]
    v_indices = edge_index_tensor[:, 1]
    
    h_u = h_0[u_indices]
    h_v = h_0[v_indices]
    
    h_1 = h_u * h_v
    
    return h_1


def compute_2_simplex_features(h_1, triangles, B2, is_binary=False):
    """
    2-simplex (triangle) features via max-pool over incident edges (or OR if binary).

    Args:
        h_1: [num_edges, feat_dim]
        triangles: set of (u, v, w) with u < v < w
        B2: sparse COO [num_edges, num_triangles]
        is_binary: if True, treat as binary / OR path inside aggregation
    """
    num_triangles = len(triangles)
    feat_dim = h_1.shape[1]
    device = h_1.device
    dtype = h_1.dtype

    if num_triangles == 0:
        return torch.empty((0, feat_dim), dtype=dtype, device=device)

    edge_idx = B2.indices()[0].to(device)   # [nnz]
    tri_idx = B2.indices()[1].to(device)    # [nnz]
    h_1_edges = h_1[edge_idx]    # [nnz, feat_dim]
    if is_binary:
        h_1_edges = (h_1_edges > 0).to(dtype)
    index_2d = tri_idx.unsqueeze(1).expand(-1, feat_dim)
    out = torch.full((num_triangles, feat_dim), float('-inf') if not is_binary else 0.0, dtype=dtype, device=device)
    out.scatter_reduce_(0, index_2d, h_1_edges, reduce='amax')
    if not is_binary:
        out[out == float('-inf')] = 0.0
    return out


def compute_all_simplex_features(v_feat, t_feat, edge_index_tensor, triangles, B2, 
                                  is_binary=False, verbose=True):
    """
    Compute 0/1/2-simplex features: node concat, edge Hadamard, triangle pool.

    Returns dict with 'h_0', 'h_1', 'h_2'.
    """
    if verbose:
        print("\nComputing simplex features...")
    
    h_0 = compute_0_simplex_features(v_feat, t_feat)
    if verbose:
        print(f"0-simplex (node) shape: {h_0.shape}")
    
    h_1 = compute_1_simplex_features(h_0, edge_index_tensor)
    if verbose:
        print(f"1-simplex (edge) shape: {h_1.shape}")
    
    h_2 = compute_2_simplex_features(
        h_1, 
        triangles, 
        B2,
        is_binary=is_binary
    )
    if verbose:
        feat_type = "binary" if is_binary else "continuous"
        print(f"2-simplex (triangle) shape ({feat_type}): {h_2.shape}")
    
    return {
        'h_0': h_0,
        'h_1': h_1,
        'h_2': h_2
    }


# Helper: convert indices to torch.Tensor on global `device` (set in main)
def to_tensor(idx):
    if isinstance(idx, torch.Tensor):
        return idx.to(device)
    elif isinstance(idx, list):
        return torch.LongTensor(idx).to(device)
    elif isinstance(idx, np.ndarray):
        return torch.LongTensor(idx).to(device)
    else:
        return torch.LongTensor(idx).to(device)


def train_model(model, h_0, edge_index, triangles_dict, h_triangles, labels, 
                train_idx, val_idx, device, args):
    """
    Train with optional micro-batching (loss only) and mixed precision.

    Returns dict with best_model_state, best_val_acc, train_history.
    """
    print("\nStarting training...", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    use_amp = args.mixed_precision and torch.cuda.is_available()
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed precision (AMP) enabled", flush=True)
    else:
        scaler = None
    
    if not use_amp and torch.cuda.is_available():
        print("Tip: enable --mixed_precision to save GPU memory", flush=True)
    
    features_on_cpu = h_0.device.type == 'cpu' or (hasattr(h_0, 'device') and h_0.device.type == 'cpu')
    
    train_batch_size = args.train_batch_size
    if train_batch_size is None and features_on_cpu:
        num_nodes = h_0.shape[0]
        if num_nodes > 90000:
            train_batch_size = 200
        elif num_nodes > 50000:
            train_batch_size = 300
        elif len(train_idx) > 50000:
            train_batch_size = 400
        elif len(train_idx) > 20000:
            train_batch_size = 600
        elif len(train_idx) > 10000:
            train_batch_size = 800
        else:
            train_batch_size = 1000
        print(f"\nAuto train_batch_size={train_batch_size} (num_nodes={num_nodes:,}, save GPU mem)", flush=True)
    
    use_batch_training = train_batch_size is not None and train_batch_size < len(train_idx)
    
    if use_batch_training:
        print(f"Micro-batch training (grad accumulation), train_batch_size={train_batch_size}", flush=True)
        print(f"  Note: train_batch_size only affects which nodes contribute to loss; forward is still full graph.", flush=True)
        print(f"  To cut peak VRAM: --batch_size_triangles, --keep_data_cpu, --use_subgraph, --mixed_precision", flush=True)
        num_train_batches = (len(train_idx) + train_batch_size - 1) // train_batch_size
        print(f"Train batches per epoch: {num_train_batches}", flush=True)
        if num_train_batches > 1000:
            print(f"Warning: many train batches ({num_train_batches}); epochs may be slow", flush=True)
            print(f"  Try larger --train_batch_size (e.g. --train_batch_size {min(1000, len(train_idx) // 10)})", flush=True)
    else:
        num_train_batches = 1
        print("Full-graph training (single loss pass)", flush=True)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    train_history = {
        'loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    oom_count = 0
    current_train_batch_size = train_batch_size if use_batch_training else len(train_idx)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}...", flush=True)
        if use_batch_training:
            print(f"  {num_train_batches} train batches (size {current_train_batch_size})", flush=True)
        
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        try:
            total_loss = 0.0
            
            for batch_idx in range(num_train_batches):
                if use_batch_training:
                    progress_interval = max(100, num_train_batches // 20)
                    if (batch_idx + 1) % progress_interval == 0 or batch_idx == 0:
                        progress_pct = (batch_idx + 1) / num_train_batches * 100
                        elapsed = time.time() - epoch_start_time
                        print(f"  Epoch {epoch+1} progress: {batch_idx+1}/{num_train_batches} ({progress_pct:.1f}%) | "
                              f"elapsed {elapsed:.1f}s", flush=True)
                if use_batch_training:
                    start_idx = batch_idx * current_train_batch_size
                    end_idx = min((batch_idx + 1) * current_train_batch_size, len(train_idx))
                    batch_train_idx = train_idx[start_idx:end_idx]
                else:
                    batch_train_idx = train_idx
                
                if batch_idx > 0:
                    torch.cuda.empty_cache()
                
                if features_on_cpu:
                    if batch_idx > 0:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    h_0_gpu = h_0.to(device, non_blocking=False)
                    h_triangles_gpu = h_triangles
                else:
                    h_0_gpu = h_0
                    h_triangles_gpu = h_triangles
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        if args.gradient_checkpointing:
                            logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
                        else:
                            logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
                        batch_loss = compute_loss(
                            logits,
                            labels,
                            batch_train_idx,
                            weight_decay=0.0,
                            model=None
                        )
                        if use_batch_training:
                            batch_loss = batch_loss / num_train_batches
                    
                    scaler.scale(batch_loss).backward()
                    total_loss += batch_loss.item() * (num_train_batches if use_batch_training else 1)
                else:
                    logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
                    batch_loss = compute_loss(
                        logits,
                        labels,
                        batch_train_idx,
                        weight_decay=0.0,
                        model=None
                    )
                    if use_batch_training:
                        batch_loss = batch_loss / num_train_batches
                    
                    batch_loss.backward()
                    total_loss += batch_loss.item() * (num_train_batches if use_batch_training else 1)
                
                del logits, batch_loss
                
                if features_on_cpu:
                    del h_0_gpu
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                torch.cuda.empty_cache()
                
                if features_on_cpu and batch_idx > 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            
            if args.weight_decay > 0:
                l2_reg = torch.tensor(0.0, device=device)
                for param in model.parameters():
                    l2_reg += torch.sum(param ** 2)
                l2_reg = args.weight_decay * l2_reg
                if use_amp:
                    scaler.scale(l2_reg).backward()
                else:
                    l2_reg.backward()
                total_loss += l2_reg.item()
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            loss_value = total_loss
            
            if epoch < 20 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                train_acc = evaluate_batched(
                    model, h_0, edge_index, triangles_dict, h_triangles, labels, 
                    train_idx, device, args.eval_batch_size, features_on_cpu
                )
                val_acc = evaluate_batched(
                    model, h_0, edge_index, triangles_dict, h_triangles, labels, 
                    val_idx, device, args.eval_batch_size, features_on_cpu
                )
            else:
                train_acc = train_history['train_acc'][-1] if len(train_history['train_acc']) > 0 else 0.0
                val_acc = best_val_acc
            
            train_history['loss'].append(loss_value)
            train_history['train_acc'].append(train_acc)
            train_history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            if epoch < 20 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                print(f"Epoch {epoch+1:4d}/{args.epochs} | Loss: {loss_value:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Time: {epoch_time:.2f}s | Best Val: {best_val_acc:.4f}", flush=True)
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Progress: Epoch {epoch+1}/{args.epochs} (Loss: {loss_value:.4f}, Val Acc: {val_acc:.4f})", flush=True)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    total_time = time.time() - start_time
                    print(f"\nEarly stopping at epoch {epoch+1} (total time: {total_time:.2f}s)", flush=True)
                    break
        
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                oom_count += 1
                print(f"\nEpoch {epoch+1}: CUDA OOM: {e}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                if not use_batch_training:
                    use_batch_training = True
                    current_train_batch_size = min(400, len(train_idx))
                    num_train_batches = (len(train_idx) + current_train_batch_size - 1) // current_train_batch_size
                    print(f"Full-graph OOM; switching to micro-batches: size={current_train_batch_size}, num_batches={num_train_batches}", flush=True)
                    oom_count = 0
                elif use_batch_training and current_train_batch_size > 100:
                    old_batch_size = current_train_batch_size
                    current_train_batch_size = max(100, current_train_batch_size // 2)
                    num_train_batches = (len(train_idx) + current_train_batch_size - 1) // current_train_batch_size
                    print(f"OOM: train_batch_size {old_batch_size} -> {current_train_batch_size} (next epoch)", flush=True)
                    print(f"   New num_batches: {num_train_batches}", flush=True)
                    oom_count = 0
                
                if oom_count >= 5:
                    print("\nRepeated OOM; options that reduce peak VRAM:", flush=True)
                    print("   1. --keep_data_cpu", flush=True)
                    print("   2. --use_subgraph", flush=True)
                    print("   3. Smaller --batch_size_triangles (e.g. 1000-2000)", flush=True)
                    print("   4. --mixed_precision", flush=True)
                    print("   5. Smaller --hidden_dim (e.g. 32)", flush=True)
                    print("   Note: --train_batch_size only scales loss; it does not cut peak VRAM.", flush=True)
                print("Skipping this epoch after cache clear; smaller batch next time.", flush=True)
                continue
            else:
                raise e
        except Exception as e:
            print(f"\nEpoch {epoch+1}: error: {e}", flush=True)
            print("Skipping epoch, continuing...", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    total_time = time.time() - start_time
    print(f"\nTraining done. Total time: {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print(f"Best val accuracy: {best_val_acc:.4f}", flush=True)
    
    return {
        'best_model_state': best_model_state,
        'best_val_acc': best_val_acc,
        'train_history': train_history
    }


def evaluate_batched(model, h_0, edge_index, triangles_dict, h_triangles, labels, idx, 
                     device, eval_batch_size=None, features_on_cpu=False):
    """
    Evaluate accuracy; optional batching over idx after one full-graph forward.

    Returns float accuracy.
    """
    model.eval()
    
    if eval_batch_size is None or eval_batch_size >= len(idx):
        if features_on_cpu:
            torch.cuda.empty_cache()
            h_0_gpu = h_0.to(device, non_blocking=False)
            h_triangles_gpu = h_triangles
        else:
            h_0_gpu = h_0
            h_triangles_gpu = h_triangles
        
        with torch.no_grad():
            logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
            pred = logits[idx].argmax(dim=1)
            acc = (pred == labels[idx]).float().mean().item()
        
        if features_on_cpu:
            del h_0_gpu, logits, pred
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        return acc
    
    if features_on_cpu:
        torch.cuda.empty_cache()
        h_0_gpu = h_0.to(device, non_blocking=False)
        h_triangles_gpu = h_triangles
    else:
        h_0_gpu = h_0
        h_triangles_gpu = h_triangles
    
    with torch.no_grad():
        logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
        num_batches = (len(idx) + eval_batch_size - 1) // eval_batch_size
        all_pred = []
        all_labels = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * eval_batch_size
            end_idx = min((batch_idx + 1) * eval_batch_size, len(idx))
            batch_idx_tensor = idx[start_idx:end_idx]
            batch_pred = logits[batch_idx_tensor].argmax(dim=1)
            batch_labels = labels[batch_idx_tensor]
            all_pred.append(batch_pred.cpu())
            all_labels.append(batch_labels.cpu())
        all_pred = torch.cat(all_pred, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = (all_pred == all_labels).float().mean().item()
    
    if features_on_cpu:
        del h_0_gpu, logits
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return acc


def get_predictions_and_logits(model, h_0, edge_index, triangles_dict, h_triangles, labels, idx,
                               device, eval_batch_size=None, features_on_cpu=False):
    """
    Predictions and logits for idx (CPU tensors). logits shape [len(idx), num_classes].
    """
    model.eval()
    if features_on_cpu:
        torch.cuda.empty_cache()
        h_0_gpu = h_0.to(device, non_blocking=False)
        h_triangles_gpu = h_triangles
    else:
        h_0_gpu = h_0
        h_triangles_gpu = h_triangles

    with torch.no_grad():
        logits = model(h_0_gpu, edge_index, triangles_dict, h_triangles_gpu)
        if eval_batch_size is None or eval_batch_size >= len(idx):
            logits_idx = logits[idx].cpu()
        else:
            chunks = []
            for start in range(0, len(idx), eval_batch_size):
                end = min(start + eval_batch_size, len(idx))
                chunks.append(logits[idx[start:end]].cpu())
            logits_idx = torch.cat(chunks, dim=0)
    y_true = labels[idx].cpu()
    y_pred = logits_idx.argmax(dim=1)
    if features_on_cpu:
        del h_0_gpu, logits
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return y_true, y_pred, logits_idx


def evaluate_model(model, h_0, edge_index, triangles_dict, h_triangles, labels,
                   train_idx, val_idx, test_idx, num_classes, best_model_state=None, device=None, args=None):
    """
    Load best checkpoint if given, then report train/val/test metrics.
    """
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best checkpoint for evaluation")
    
    print("\nEvaluating...")
    
    features_on_cpu = h_0.device.type == 'cpu' or (hasattr(h_0, 'device') and h_0.device.type == 'cpu')
    
    if device is None:
        device = h_0.device if not features_on_cpu else torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    eval_batch_size = getattr(args, 'eval_batch_size', None) if args is not None else None
    
    train_acc = evaluate_batched(
        model, h_0, edge_index, triangles_dict, h_triangles, labels, 
        train_idx, device, eval_batch_size, features_on_cpu
    )
    print(f"Train accuracy: {train_acc:.4f}")
    
    val_acc = evaluate_batched(
        model, h_0, edge_index, triangles_dict, h_triangles, labels, 
        val_idx, device, eval_batch_size, features_on_cpu
    )
    print(f"Val accuracy: {val_acc:.4f}")
    
    y_true, y_pred, logits_test = get_predictions_and_logits(
        model, h_0, edge_index, triangles_dict, h_triangles, labels,
        test_idx, device, eval_batch_size, features_on_cpu
    )
    test_metrics = calculate_metrics(y_true, y_pred, logits_test, num_classes=num_classes)
    test_acc = test_metrics['accuracy']
    print(f"Test - Accuracy: {test_acc:.4f}, Precision: {test_metrics['precision']:.4f}, "
          f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    log_per_class_metrics(y_true, y_pred, num_classes, logits=logits_test)
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'test_metrics': test_metrics
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='grocery',
                        choices=['grocery', 'toys'],
                        help='Dataset: grocery or toys (MAGB format)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Data directory; overrides --dataset if set')
    parser.add_argument('--feat_path', type=str, default=None, help='Optional merged feature file')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Train ratio (MAGB split)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Val ratio (MAGB split)')
    
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim (default 64)')
    parser.add_argument('--num_layers', type=int, default=2, help='Message passing layers (default 2)')
    parser.add_argument('--lambda_high', type=float, default=1.0, help='Higher-order strength (default 1.0)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout (default 0.5)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 weight decay (default 5e-4)')
    parser.add_argument('--epochs', type=int, default=1000, help='Max epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--repeat_times', type=int, default=5, help='Repeat runs for mean±std on test')
    
    parser.add_argument('--batch_size_triangles', type=int, default=None, 
                        help='Node batch size for triangle aggregation (GPU mem); None disables. Try 1000-5000')
    parser.add_argument('--auto_batch_size', action='store_true', 
                        help='Auto-set triangle batch size from triangle count')
    parser.add_argument('--train_batch_size', type=int, default=None,
                        help='Micro-batch for loss only (full-graph forward). For VRAM use --batch_size_triangles, --keep_data_cpu, --use_subgraph')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Slice eval indices; None = full idx. Try 2000-10000')
    parser.add_argument('--keep_data_cpu', action='store_true',
                        help='Keep features on CPU; load to GPU on demand')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='AMP training (~50%% less GPU memory)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Gradient checkpointing (slower, saves memory)')
    
    parser.add_argument('--use_subgraph', action='store_true',
                        help='k-hop subgraph around centers (saves memory)')
    parser.add_argument('--subgraph_hops', type=int, default=2,
                        help='Subgraph hops (default 2 for 2-layer GNN)')
    parser.add_argument('--subgraph_include_val_test', action='store_true',
                        help='Also use val/test nodes as subgraph centers')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    load_device = torch.device("cpu") if args.keep_data_cpu else device
    raw_data = load_raw_dataset(
        dataset=args.dataset,
        device=load_device,
        data_path=args.data_path,
        feat_path=args.feat_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    if args.keep_data_cpu:
        print(f"Data loaded on CPU (--keep_data_cpu)")
    
    print(f"\nRaw data summary:")
    print(f"Edges: {len(raw_data['edges'])}")
    print(f"Nodes: {raw_data['num_nodes']}")
    print(f"Classes: {raw_data['num_classes']}")
    print(f"Train nodes: {len(raw_data['splits']['train_idx'])}")
    print(f"Val nodes: {len(raw_data['splits']['val_idx'])}")
    print(f"Test nodes: {len(raw_data['splits']['test_idx'])}")
    print(f"Vision feat shape: {raw_data['v_feat'].shape}")
    print(f"Text feat shape: {raw_data['t_feat'].shape}")


    if isinstance(raw_data['edges'], torch.Tensor):
        edge_index_full = raw_data['edges'].clone()
    elif isinstance(raw_data['edges'], np.ndarray):
        edge_index_full = torch.from_numpy(raw_data['edges'])
    else:
        edge_index_full = torch.tensor(raw_data['edges'], dtype=torch.long)
    
    train_idx_raw = raw_data['splits']['train_idx']
    val_idx_raw = raw_data['splits']['val_idx']
    test_idx_raw = raw_data['splits']['test_idx']
    
    subgraph_info = None
    original_num_nodes = raw_data['num_nodes']
    if args.use_subgraph:
        print(f"\n{'='*60}")
        print("Subgraph sampling: {}-hop around centers...".format(args.subgraph_hops))
        print(f"{'='*60}")
        
        if args.subgraph_include_val_test:
            center_nodes = torch.cat([
                torch.tensor(train_idx_raw, dtype=torch.long),
                torch.tensor(val_idx_raw, dtype=torch.long),
                torch.tensor(test_idx_raw, dtype=torch.long)
            ]).unique()
            print(f"Centers: train+val+test ({len(center_nodes):,} nodes)")
        else:
            center_nodes = torch.tensor(train_idx_raw, dtype=torch.long)
            print(f"Centers: train only ({len(center_nodes):,} nodes)")
            print("   Val/test nodes included if within {} hops of train.".format(args.subgraph_hops))
        
        subgraph_info = extract_k_hop_subgraph(
            center_nodes=center_nodes,
            edge_index=edge_index_full,
            num_nodes=raw_data['num_nodes'],
            k=args.subgraph_hops,
            return_node_mapping=True
        )
        
        print(f"Full graph nodes: {raw_data['num_nodes']:,}")
        print(f"Subgraph nodes: {subgraph_info['num_subgraph_nodes']:,}")
        print(f"Node reduction: {(1 - subgraph_info['num_subgraph_nodes'] / raw_data['num_nodes']) * 100:.1f}%")
        print(f"Full graph edges: {edge_index_full.shape[0]:,}")
        print(f"Subgraph edges: {subgraph_info['num_subgraph_edges']:,}")
        print(f"Edge reduction: {(1 - subgraph_info['num_subgraph_edges'] / edge_index_full.shape[0]) * 100:.1f}%")
        
        edge_index_for_simplicial = subgraph_info['subgraph_edge_index']
        num_nodes_subgraph = subgraph_info['num_subgraph_nodes']
        
        node_mapping = subgraph_info['node_mapping']
        train_idx_subgraph = []
        val_idx_subgraph = []
        test_idx_subgraph = []
        
        for idx in train_idx_raw:
            if idx in node_mapping:
                train_idx_subgraph.append(node_mapping[idx])
        for idx in val_idx_raw:
            if idx in node_mapping:
                val_idx_subgraph.append(node_mapping[idx])
        for idx in test_idx_raw:
            if idx in node_mapping:
                test_idx_subgraph.append(node_mapping[idx])
        
        train_idx_subgraph = torch.tensor(train_idx_subgraph, dtype=torch.long)
        val_idx_subgraph = torch.tensor(val_idx_subgraph, dtype=torch.long)
        test_idx_subgraph = torch.tensor(test_idx_subgraph, dtype=torch.long)
        
        print(f"Train in subgraph: {len(train_idx_subgraph):,} / {len(train_idx_raw):,}")
        print(f"Val in subgraph: {len(val_idx_subgraph):,} / {len(val_idx_raw):,}")
        print(f"Test in subgraph: {len(test_idx_subgraph):,} / {len(test_idx_raw):,}")
        
        if len(train_idx_subgraph) < len(train_idx_raw):
            missing_count = len(train_idx_raw) - len(train_idx_subgraph)
            print(f"ERROR: {missing_count} train nodes missing from subgraph (unexpected).")
            print(f"   Train nodes should be centers; check subgraph code.")
        else:
            print(f"OK: all {len(train_idx_subgraph):,} train nodes in subgraph")
        
        if len(val_idx_subgraph) < len(val_idx_raw):
            missing_count = len(val_idx_raw) - len(val_idx_subgraph)
            missing_pct = (missing_count / len(val_idx_raw)) * 100
            print(f"Warning: {missing_count} val nodes ({missing_pct:.1f}%) not in subgraph")
            if missing_pct > 10:
                print(f"   Low val acc may be due to missing val nodes.")
                print(f"   Try: 1) --subgraph_hops 3 or 4")
                print(f"         2) --subgraph_include_val_test")
        else:
            print(f"OK: all {len(val_idx_subgraph):,} val nodes in subgraph")
        
        if len(test_idx_subgraph) < len(test_idx_raw):
            missing_count = len(test_idx_raw) - len(test_idx_subgraph)
            missing_pct = (missing_count / len(test_idx_raw)) * 100
            print(f"Warning: {missing_count} test nodes ({missing_pct:.1f}%) not in subgraph")
            if missing_pct > 10:
                print(f"   Low test acc may be due to missing test nodes.")
                print(f"   Try: 1) --subgraph_hops 3 or 4")
                print(f"         2) --subgraph_include_val_test")
        else:
            print(f"OK: all {len(test_idx_subgraph):,} test nodes in subgraph")
        
        subgraph_nodes_orig = subgraph_info['subgraph_nodes']
        h_0_subgraph = raw_data['v_feat'][subgraph_nodes_orig.cpu()].clone()
        t_feat_subgraph = raw_data['t_feat'][subgraph_nodes_orig.cpu()].clone()
        labels_subgraph = raw_data['labels'][subgraph_nodes_orig.cpu()].clone()
        
        raw_data['num_nodes'] = num_nodes_subgraph
        raw_data['v_feat'] = h_0_subgraph
        raw_data['t_feat'] = t_feat_subgraph
        raw_data['labels'] = labels_subgraph
        raw_data['splits'] = {
            'train_idx': train_idx_subgraph,
            'val_idx': val_idx_subgraph,
            'test_idx': test_idx_subgraph
        }
        
        print(f"\nSubgraph sampling done; using induced subgraph.")
        print(f"{'='*60}\n")
    else:
        edge_index_for_simplicial = edge_index_full
        num_nodes_subgraph = raw_data['num_nodes']

    simplicial_data = build_triangles_and_B2(edge_index_for_simplicial, num_nodes_subgraph)

    print(f"Edges: {simplicial_data['num_edges']}")
    print(f"Triangles: {simplicial_data['num_triangles']}")

    v_feat_for_compute = raw_data['v_feat']
    t_feat_for_compute = raw_data['t_feat']
    edge_index_for_compute = simplicial_data['edge_index_tensor']
    B2_for_compute = simplicial_data['B2']
    
    if args.keep_data_cpu:
        if v_feat_for_compute.device.type != 'cpu':
            v_feat_for_compute = v_feat_for_compute.cpu()
        if t_feat_for_compute.device.type != 'cpu':
            t_feat_for_compute = t_feat_for_compute.cpu()
        if edge_index_for_compute.device.type != 'cpu':
            edge_index_for_compute = edge_index_for_compute.cpu()
        if B2_for_compute.device.type != 'cpu':
            B2_for_compute = B2_for_compute.cpu()
        print("Computing features on CPU (--keep_data_cpu)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    simplex_features = compute_all_simplex_features(
        v_feat=v_feat_for_compute,
        t_feat=t_feat_for_compute,
        edge_index_tensor=edge_index_for_compute,
        triangles=simplicial_data['triangles'],
        B2=B2_for_compute,
        is_binary=False,
        verbose=True
    )
    
    h_0 = simplex_features['h_0']
    h_2_continuous = simplex_features['h_2']
    del simplex_features['h_1']
    del v_feat_for_compute, t_feat_for_compute, edge_index_for_compute, B2_for_compute
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Feature computation done; GPU cache cleared")
    
    print("\nBuilding node-to-triangle map...")
    triangles_dict = build_node_to_triangles_dict(
        simplicial_data['triangles'], 
        num_nodes_subgraph
    )
    print(f"Node-to-triangle map built")
    
    print("\nPreparing training data...")
    
    train_idx = to_tensor(raw_data['splits']['train_idx'])
    val_idx = to_tensor(raw_data['splits']['val_idx'])
    test_idx = to_tensor(raw_data['splits']['test_idx'])

    labels_for_log = raw_data['labels']
    if hasattr(labels_for_log, 'cpu'):
        labels_for_log = labels_for_log.cpu()
    log_label_distribution(
        labels_for_log,
        raw_data['splits']['train_idx'],
        raw_data['splits']['val_idx'],
        raw_data['splits']['test_idx'],
        raw_data['num_classes'],
        name="label distribution (current split)"
    )
    
    if args.keep_data_cpu:
        print("\nKeeping features on CPU (staged GPU load)...")
        h_0 = h_0.cpu()
        h_2_continuous = h_2_continuous.cpu()
        edge_index = simplicial_data['edge_index_tensor'].to(device)
        labels = raw_data['labels'].to(device)
        print("Features stay on CPU; moved to GPU per train/eval step as needed")
        if args.use_subgraph:
            print("With subgraph: only subgraph features touch GPU")
            print(f"   Subgraph nodes: {h_0.shape[0]:,} ({(1 - h_0.shape[0] / original_num_nodes) * 100:.1f}% fewer than full graph)")
        else:
            print("Note: full-graph GNN still needs full h_0 on GPU during forward.")
            print("   Consider --use_subgraph to shrink working set.")
    else:
        print("\nLoading tensors to GPU...")
        if args.use_subgraph:
            print(f"Subgraph: loading {h_0.shape[0]:,} node feats (full graph had {original_num_nodes:,})")
        else:
            print("If OOM, try --keep_data_cpu or --use_subgraph")
        h_0 = h_0.to(device)
        h_2_continuous = h_2_continuous.to(device)
        edge_index = simplicial_data['edge_index_tensor'].to(device)
        labels = raw_data['labels'].to(device)
    
    print("\nInitializing model...")
    
    batch_size_triangles = args.batch_size_triangles
    num_triangles = h_2_continuous.shape[0]
    
    if args.keep_data_cpu and batch_size_triangles is None:
        if num_triangles > 50000:
            num_nodes_current = h_0.shape[0]
            if num_nodes_current > 90000:
                if num_triangles > 5000000:
                    batch_size_triangles = 500
                elif num_triangles > 2000000:
                    batch_size_triangles = 800
                elif num_triangles > 500000:
                    batch_size_triangles = 1000
                elif num_triangles > 100000:
                    batch_size_triangles = 1500
                else:
                    batch_size_triangles = 2000
            else:
                if num_triangles > 5000000:
                    batch_size_triangles = 1000
                elif num_triangles > 2000000:
                    batch_size_triangles = 1500
                elif num_triangles > 500000:
                    batch_size_triangles = 2000
                elif num_triangles > 100000:
                    batch_size_triangles = 2500
                else:
                    batch_size_triangles = 4000
            print(f"\nWarning: --keep_data_cpu without --batch_size_triangles")
            print(f"   Auto triangle batch size: {batch_size_triangles} (num_nodes={num_nodes_current:,})")
            print(f"   Avoids loading all triangle feats to GPU at once")
    
    if args.auto_batch_size and batch_size_triangles is None:
        if num_triangles > 50000:
            if num_triangles > 5000000:
                auto_batch_size = 1000
            elif num_triangles > 2000000:
                auto_batch_size = 1500
            elif num_triangles > 500000:
                auto_batch_size = 2000
            elif num_triangles > 100000:
                auto_batch_size = 2500
            else:
                auto_batch_size = 4000
            batch_size_triangles = auto_batch_size
            print(f"Many triangles ({num_triangles:,}); auto batch_size_triangles={batch_size_triangles}")
    elif batch_size_triangles is None:
        if num_triangles > 50000:
            if num_triangles > 500000:
                suggested_batch_size = 4000
            elif num_triangles > 100000:
                suggested_batch_size = 5000
            else:
                suggested_batch_size = 8000
            print(f"\nTip: {num_triangles:,} triangles; consider --batch_size_triangles {suggested_batch_size} or --auto_batch_size")
    
    if batch_size_triangles is not None:
        print(f"Batched triangles, node batch size: {batch_size_triangles}")
    
    num_classes = raw_data['num_classes']
    all_test_metrics = []

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoint dir: {save_dir}\n")

    for run in range(args.repeat_times):
        if args.repeat_times > 1:
            print(f"\n{'='*60}")
            print(f"Run {run + 1}/{args.repeat_times}")
            print(f"{'='*60}")
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + run)
        
        model = SimplicialGNN(
            input_dim=h_0.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers,
            lambda_high=args.lambda_high,
            dropout=args.dropout,
            batch_size_triangles=batch_size_triangles
        ).to(device)
        
        if run == 0:
            print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
            print(f"Model:")
            print(model)
        
        train_results = train_model(
            model=model,
            h_0=h_0,
            edge_index=edge_index,
            triangles_dict=triangles_dict,
            h_triangles=h_2_continuous,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            device=device,
            args=args
        )
        
        eval_results = evaluate_model(
            model=model,
            h_0=h_0,
            edge_index=edge_index,
            triangles_dict=triangles_dict,
            h_triangles=h_2_continuous,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_classes=num_classes,
            best_model_state=train_results['best_model_state'],
            device=device,
            args=args
        )

        best_model_state = train_results['best_model_state']
        if best_model_state is not None:
            save_path = os.path.join(save_dir, f"best_model_{args.dataset}_run{run+1}.pt")
            torch.save({
                'model_state_dict': best_model_state,
                'run': run + 1,
                'best_val_acc': train_results['best_val_acc'],
                'test_metrics': eval_results['test_metrics'],
            }, save_path)
            print(f"Saved best model run {run+1} to: {save_path}")

        all_test_metrics.append(eval_results['test_metrics'])
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    print(f"\nFinal test metrics (mean ± std over runs):")
    print("="*60)
    for name in metrics_names:
        values = [m[name] for m in all_test_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{name.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    print("===================================")
    
    
