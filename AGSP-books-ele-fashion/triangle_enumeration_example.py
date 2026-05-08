"""
Triangle enumeration example — how triangles are found by enumeration.

This example shows enumerating all candidate triangles with nested loops.
"""

# Example graph:
#   0 --- 1
#   | \   |
#   |  \  |
#   |   \ |
#   2 --- 3
# This graph has two triangles: (0,1,3) and (0,2,3)

# Edge list
edges = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
num_nodes = 4

# Build adjacency lists
graph = {}
for u, v in edges:
    if u not in graph:
        graph[u] = set()
    if v not in graph:
        graph[v] = set()
    graph[u].add(v)
    graph[v].add(u)

print("Adjacency lists:")
for node, neighbors in graph.items():
    print(f"  Node {node} neighbors: {neighbors}")

print("\nEnumeration trace:")

# Enumerate all triangles
triangles_found = []
enumeration_steps = []

for u in range(num_nodes):
    if u not in graph:
        continue
    
    neighbors_u = graph[u]
    
    for v in neighbors_u:
        if u < v:  # avoid duplicates: only u < v
            neighbors_v = graph[v]
            common_neighbors = neighbors_u.intersection(neighbors_v)
            
            for w in common_neighbors:
                if u < v < w:  # avoid duplicates: only u < v < w
                    # found a triangle
                    triangle = (u, v, w)
                    triangles_found.append(triangle)
                    enumeration_steps.append(f"enumerate: u={u}, v={v}, common neighbor w={w} -> triangle {triangle}")

print("\nStep-by-step:")
for i, step in enumerate(enumeration_steps, 1):
    print(f"  Step {i}: {step}")

print(f"\nTotal triangles found by enumeration: {len(triangles_found)}")
print(f"Triangle list: {triangles_found}")

print("\nNotes:")
print("- Triangles come from nested-loop enumeration (explicit search).")
print("- set() stores results and deduplicates; it does not generate triangles.")
print("- Hashing only speeds lookups (e.g. common neighbors); enumeration logic is unchanged.")
