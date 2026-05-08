"""
Example: how to use the build_simplicial module together with run.py.

Demonstrates building simplicial-complex structure from data loaded via run.py.
"""

import torch
from run import load_raw_dataset
from build_simplicial import build_triangles_and_B2


def example_build_simplicial_from_run():
    """
    Example: build a simplicial complex from data loaded by run.py.
    """
    # 1. Load raw data (load_raw_dataset from run.py)
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_data = load_raw_dataset(
        dataset='ele-fashion',  # or 'books-nc'
        device=device
    )

    # 2. Edges and node count
    edges = raw_data['edges']
    num_nodes = raw_data['num_nodes']

    print(f"\nDataset summary:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Edge type: {type(edges)}")

    # 3. Normalize edge layout if needed
    if isinstance(edges, torch.Tensor):
        if edges.dim() == 2 and edges.shape[1] == 2:
            edge_data = edges
        elif edges.dim() == 2 and edges.shape[0] == 2:
            edge_data = edges.T  # [2, num_edges] -> [num_edges, 2]
        else:
            raise ValueError(f"Unsupported edge tensor shape: {edges.shape}")
    elif isinstance(edges, list):
        edge_data = torch.tensor(edges, dtype=torch.long)
    else:
        raise TypeError(f"Unsupported edge data type: {type(edges)}")

    print(f"  Edge tensor shape: {edge_data.shape}")
    print(f"  Number of edges: {edge_data.shape[0]}")

    # 4. Triangles and B2 boundary operator
    print("\n" + "=" * 60)
    print("Building simplicial complex (triangles and B2)...")
    print("=" * 60)

    result = build_triangles_and_B2(edge_data, num_nodes)

    # 5. Report
    print(f"\nBuild summary:")
    print(f"  Number of triangles: {result['num_triangles']}")
    print(f"  B2 shape: {result['B2'].shape}")
    print(f"  B2 is sparse: {result['B2'].is_sparse}")

    triangles_list = sorted(list(result['triangles']))
    print(f"\nFirst 10 triangles (if any):")
    for i, triangle in enumerate(triangles_list[:10]):
        print(f"    {i+1}. {triangle}")
    if len(triangles_list) > 10:
        print(f"    ... {len(triangles_list) - 10} more triangle(s)")

    return {
        'raw_data': raw_data,
        'simplicial_result': result
    }


if __name__ == '__main__':
    try:
        result = example_build_simplicial_from_run()
        print("\n" + "=" * 60)
        print("Example finished successfully.")
        print("=" * 60)
        print("\nReturn value contains:")
        print("  - raw_data: original dataset (nodes, edges, features, etc.)")
        print("  - simplicial_result: simplicial build output (triangles, B2, etc.)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
