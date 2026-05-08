"""
Example: how to use the build_simplicial module from run.py

This file demonstrates how to build simplicial complex structures from data loaded by run.py.
"""

import torch
from run import load_raw_dataset
from build_simplicial import build_triangles_and_B2


def example_build_simplicial_from_run():
    """
    Example: build a simplicial complex from data loaded by run.py
    """
    # 1. Load raw data (via load_raw_dataset in run.py)
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_data = load_raw_dataset(
        dataset='ele-fashion',  # or 'books-nc'
        device=device
    )
    
    # 2. Get edges and node info
    edges = raw_data['edges']
    num_nodes = raw_data['num_nodes']
    
    print(f"\nDataset info:")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Edge type: {type(edges)}")
    
    # 3. Check edge format and convert if needed
    # edges may be in different formats; normalize here
    if isinstance(edges, torch.Tensor):
        # Already a tensor: check shape
        if edges.dim() == 2 and edges.shape[1] == 2:
            # Expected shape: [num_edges, 2]
            edge_data = edges
        elif edges.dim() == 2 and edges.shape[0] == 2:
            # Transpose: [2, num_edges] -> [num_edges, 2]
            edge_data = edges.T
        else:
            raise ValueError(f"Unsupported edge tensor shape: {edges.shape}")
    elif isinstance(edges, list):
        # List: convert to tensor
        edge_data = torch.tensor(edges, dtype=torch.long)
    else:
        raise TypeError(f"Unsupported edge data type: {type(edges)}")
    
    print(f"  Edge tensor shape: {edge_data.shape}")
    print(f"  Number of edges: {edge_data.shape[0]}")
    
    # 4. Build triangles and B2 boundary operator
    print("\n" + "=" * 60)
    print("Building simplicial complex (triangles and B2 matrix)...")
    print("=" * 60)
    
    result = build_triangles_and_B2(edge_data, num_nodes)
    
    # 5. Print results
    print(f"\nBuild result:")
    print(f"  Number of triangles: {result['num_triangles']}")
    print(f"  B2 matrix shape: {result['B2'].shape}")
    print(f"  B2 is sparse: {result['B2'].is_sparse}")
    
    # Show first few triangles if many exist
    triangles_list = sorted(list(result['triangles']))
    print(f"\nFirst 10 triangles (if any):")
    for i, triangle in enumerate(triangles_list[:10]):
        print(f"    {i+1}. {triangle}")
    if len(triangles_list) > 10:
        print(f"    ... {len(triangles_list) - 10} more triangles")
    
    # 6. Return for downstream use
    return {
        'raw_data': raw_data,
        'simplicial_result': result
    }


if __name__ == '__main__':
    try:
        result = example_build_simplicial_from_run()
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        print("\nReturned dict contains:")
        print("  - raw_data: raw dataset (nodes, edges, features, etc.)")
        print("  - simplicial_result: simplicial build output (triangles, B2 matrix, etc.)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
