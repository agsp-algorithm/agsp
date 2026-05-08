

import torch
import numpy as np


def build_triangles_and_B2(edges, num_nodes):
    """
    Build the set of triangles and the B2 boundary operator from edge data.

    Args:
        edges: Edge data in one of the following formats:
            - torch.Tensor: shape [num_edges, 2], each row is [source, target]
            - numpy.ndarray: shape [num_edges, 2]
            - list: list of (u, v) tuples
        num_nodes: Number of nodes (int).

    Returns:
        dict with keys:
            - 'triangles': set of triangles, each as (u, v, w) with u < v < w
            - 'B2': B2 boundary operator (sparse COO tensor), shape [num_edges, num_triangles]
            - 'edges_index': dict mapping edge (u, v) to its index in the edge list
            - 'num_edges': total number of edges
            - 'num_triangles': total number of triangles
            - 'edge_index_tensor': edge index tensor, shape [num_edges, 2]

    Example:
        >>> edges = torch.tensor([[0, 1], [1, 2], [0, 2]])  # three edges forming one triangle
        >>> num_nodes = 3
        >>> result = build_triangles_and_B2(edges, num_nodes)
        >>> print(result['num_triangles'])  # should print 1
        >>> print(result['B2'].shape)  # should be [3, 1]
    """

    # 1. Normalize edge data to torch.Tensor
    if isinstance(edges, list):
        if len(edges) > 0 and isinstance(edges[0], tuple):
            # Tuple list -> numpy -> tensor
            edges_array = np.array(edges)
            edge_index_tensor = torch.from_numpy(edges_array)
        else:
            edge_index_tensor = torch.tensor(edges)
    elif isinstance(edges, np.ndarray):
        edge_index_tensor = torch.from_numpy(edges)
    elif isinstance(edges, torch.Tensor):
        edge_index_tensor = edges.clone()
    else:
        raise TypeError(f"Unsupported edge data type: {type(edges)}")

    # Ensure shape [num_edges, 2]
    if edge_index_tensor.dim() != 2 or edge_index_tensor.shape[1] != 2:
        raise ValueError(f"Edges must have shape [num_edges, 2], got {edge_index_tensor.shape}")

    num_edges = edge_index_tensor.shape[0]

    # 2. Adjacency list: key = node id, value = set of neighbors
    graph = {}
    for i in range(num_edges):
        u = edge_index_tensor[i, 0].item()
        v = edge_index_tensor[i, 1].item()

        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            continue

        # Undirected: add both directions
        if u not in graph:
            graph[u] = set()
        graph[v] = graph.get(v, set())
        graph[u].add(v)
        graph[v].add(u)

    # 3. Edge index map: (u, v) -> position in edge list (for fast lookup)
    edges_index = {}
    for i in range(num_edges):
        u = edge_index_tensor[i, 0].item()
        v = edge_index_tensor[i, 1].item()

        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            continue

        # Store only u < v to avoid duplicates; reverse lookup still used later
        if (u, v) not in edges_index and (v, u) not in edges_index:
            edges_index[(u, v)] = i

    # 4. Enumerate triangles and build B2
    triangles = set()
    triangle_num = 0
    B2_row = []
    B2_column = []
    values = []

    # For each pair (u, v), common neighbors yield triangles
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

                        # Edge (u, v)
                        edge_index_uv = edges_index.get((u, v), None)
                        if edge_index_uv is None:
                            edge_index_uv = edges_index.get((v, u), None)
                        if edge_index_uv is None:
                            triangles.discard((u, v, w))
                            continue

                        # Edge (u, w)
                        edge_index_uw = edges_index.get((u, w), None)
                        if edge_index_uw is None:
                            edge_index_uw = edges_index.get((w, u), None)
                        if edge_index_uw is None:
                            triangles.discard((u, v, w))
                            continue

                        # Edge (v, w)
                        edge_index_vw = edges_index.get((v, w), None)
                        if edge_index_vw is None:
                            edge_index_vw = edges_index.get((w, v), None)
                        if edge_index_vw is None:
                            triangles.discard((u, v, w))
                            continue

                        # B2: each column is a triangle, each row an edge
                        B2_row.extend([edge_index_uv, edge_index_uw, edge_index_vw])
                        B2_column.extend([triangle_num, triangle_num, triangle_num])
                        # Boundary coefficients: (u,v)=1, (u,w)=-1, (v,w)=1
                        values.extend([1, -1, 1])
                        triangle_num += 1

    # 5. Sparse B2
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

    return {
        'triangles': triangles,
        'B2': B2,
        'edges_index': edges_index,
        'num_edges': num_edges,
        'num_triangles': num_triangles,
        'edge_index_tensor': edge_index_tensor
    }


if __name__ == "__main__":
    """Smoke tests for this module."""
    print("=" * 60)
    print("Simplicial complex builder — tests")
    print("=" * 60)

    print("\nTest 1: simple triangle graph")
    print("-" * 60)
    edges = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)
    num_nodes = 3
    result = build_triangles_and_B2(edges, num_nodes)
    print(f"Number of edges: {result['num_edges']}")
    print(f"Number of triangles: {result['num_triangles']}")
    print(f"Triangle set: {result['triangles']}")
    print(f"B2 shape: {result['B2'].shape}")
    print(f"B2 dense:\n{result['B2'].to_dense()}")

    print("\nTest 2: graph with multiple triangles")
    print("-" * 60)
    # K4 complete graph on 4 nodes has C(4,3)=4 triangles
    edges = torch.tensor([
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3],
        [2, 3]
    ], dtype=torch.long)
    num_nodes = 4
    result = build_triangles_and_B2(edges, num_nodes)
    print(f"Number of edges: {result['num_edges']}")
    print(f"Number of triangles: {result['num_triangles']}")
    print(f"Triangle set: {result['triangles']}")
    print(f"B2 shape: {result['B2'].shape}")

    print("\nTests finished.")
    print("=" * 60)
