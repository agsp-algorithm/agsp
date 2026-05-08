"""
MAGB-format data loader — loads MAGB multimodal graph data (Grocery / Toys and same-format datasets).
Follows MAGB-master/GNN/GraphData.py and converts DGL graphs to the required format.
Infers graph and feature file names from the last directory component of data_path
(e.g. Toys -> ToysGraph.pt, Grocery -> GroceryGraph.pt).
"""

import os
import glob
import numpy as np
import torch


def _dataset_name_from_path(data_path):
    """Infer dataset name from data directory path, e.g. /path/to/Data/Toys -> Toys."""
    return os.path.basename(os.path.normpath(os.path.abspath(data_path)))


def _split_graph(nodes_num, train_ratio, val_ratio, labels, fewshots=None):
    """Split train/val/test indices; matches MAGB GraphData.split_graph."""
    np.random.seed(45)
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    indices = np.random.permutation(nodes_num)
    if fewshots is not None:
        train_ids = []
        for label in np.unique(labels_np):
            label_indices = np.where(labels_np == label)[0]
            np.random.shuffle(label_indices)
            train_ids.extend(label_indices[:fewshots])
        remaining = np.setdiff1d(indices, train_ids)
        np.random.shuffle(remaining)
        val_size = int(len(remaining) * val_ratio)
        val_ids, test_ids = remaining[:val_size], remaining[val_size:]
    else:
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)
        train_ids = indices[:train_size]
        val_ids = indices[train_size : train_size + val_size]
        test_ids = indices[train_size + val_size :]
    return train_ids, val_ids, test_ids


def load_magb_grocery_data(
    data_path,
    graph_name=None,
    feat_path=None,
    text_feat_path=None,
    image_feat_path=None,
    train_ratio=0.6,
    val_ratio=0.2,
    fewshots=None,
    undirected=True,
):
    """
    Load MAGB data. Supports Grocery, Toys, and same-format datasets.
    If graph_name is omitted, it is inferred from the last directory name of data_path
    (e.g. Toys -> ToysGraph.pt).
    Returns: edges, labels, splits, v_feat, t_feat, num_classes
    """
    import dgl

    dataset_name = _dataset_name_from_path(data_path)
    if graph_name is None:
        graph_name = f"{dataset_name}Graph.pt"

    graph_path = os.path.join(data_path, graph_name)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    graph_list, _ = dgl.load_graphs(graph_path)
    graph = graph_list[0]
    labels = graph.ndata["label"].long()

    src, dst = graph.edges()
    src, dst = src.numpy(), dst.numpy()
    edges = np.stack([src, dst], axis=1)
    if undirected:
        edges = np.vstack([edges, np.stack([dst, src], axis=1)])
    edges = torch.LongTensor(edges)

    train_ids, val_ids, test_ids = _split_graph(
        graph.num_nodes(), train_ratio, val_ratio, labels, fewshots
    )
    splits = {
        "train_idx": torch.LongTensor(train_ids),
        "val_idx": torch.LongTensor(val_ids),
        "test_idx": torch.LongTensor(test_ids),
    }

    if feat_path is not None and os.path.exists(feat_path):
        clip_feat = np.load(feat_path).astype(np.float32)
        clip_feat = torch.from_numpy(clip_feat).float()
        mid = clip_feat.shape[1] // 2
        t_feat, v_feat = clip_feat[:, :mid], clip_feat[:, mid:]
    elif text_feat_path is not None and image_feat_path is not None:
        t_raw = np.load(text_feat_path).astype(np.float32)
        v_raw = np.load(image_feat_path).astype(np.float32)
        if t_raw.shape[0] != v_raw.shape[0]:
            raise ValueError("Text and image features have inconsistent numbers of nodes")
        t_feat = torch.from_numpy(t_raw).float()
        v_feat = torch.from_numpy(v_raw).float()
    else:
        default_mm = os.path.join(data_path, "MMFeature", f"{dataset_name}_LLAMA8B_CLIP.npy")
        default_t = os.path.join(data_path, "TextFeature", f"{dataset_name}_roberta_base_256_mean.npy")
        default_v = os.path.join(data_path, "ImageFeature", f"{dataset_name}_openai_clip-vit-large-patch14.npy")
        if not os.path.exists(default_t) and os.path.isdir(os.path.join(data_path, "TextFeature")):
            text_candidates = glob.glob(os.path.join(data_path, "TextFeature", f"{dataset_name}_roberta*.npy"))
            if text_candidates:
                default_t = text_candidates[0]
        if os.path.exists(default_mm):
            clip_feat = torch.from_numpy(np.load(default_mm).astype(np.float32)).float()
            mid = clip_feat.shape[1] // 2
            t_feat, v_feat = clip_feat[:, :mid], clip_feat[:, mid:]
        elif os.path.exists(default_t) and os.path.exists(default_v):
            t_feat = torch.from_numpy(np.load(default_t).astype(np.float32)).float()
            v_feat = torch.from_numpy(np.load(default_v).astype(np.float32)).float()
        else:
            raise FileNotFoundError(
                f"No feature files found; pass --feat_path or ensure {default_mm} exists, or both {default_t} and {default_v}"
            )

    num_classes = int(labels.max().item()) + 1
    return edges, labels, splits, v_feat, t_feat, num_classes
