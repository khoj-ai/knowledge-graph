import networkx as nx
import html
from typing import Any, Dict, List, Tuple, cast
from graspologic.partition import hierarchical_leiden, HierarchicalClusters

def run_leiden(
    G: nx.Graph, 
    max_cluster_size = 10,
    use_lcc = True
) -> Tuple[Dict[int, Dict[int, List[str]]], Dict[int, int]]:
    node_id_to_community_map, community_hierarchy_map = compute_leiden_communities(
        G,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc
    )
    
    levels = sorted(node_id_to_community_map.keys())
    
    results_by_level: Dict[int, Dict[int, List[str]]] = {}
    for level in levels:
        results: Dict[int, List[str]] = {}
        results_by_level[level] = results
        for node_id, community_id in node_id_to_community_map[level].items():
            results[community_id] = results.get(community_id, [])
            results[community_id].append(node_id)
            
    return results_by_level, community_hierarchy_map
    

def compute_leiden_communities(
    graph: nx.Graph,
    max_cluster_size: int = 10,
    use_lcc: bool = True,
    seed=0xDEADBEEF,
) -> Tuple[Dict[int, Dict[str, int]], Dict[int, int]]:
    
    if use_lcc:
        graph = stable_largest_connected_component(graph)
        
    community_mapping: HierarchicalClusters = hierarchical_leiden(graph, max_cluster_size=max_cluster_size, random_seed=seed)
    
    results: Dict[int, Dict[str, int]] = {}
    hierarchy: Dict[int, int] = {}
    
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster
        
        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )
        
    return results, hierarchy
        
        

def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
    # NOTE: The import is done here to reduce the initial import time of the module
    from graspologic.utils import largest_connected_component

    graph = graph.copy()
    graph = cast("nx.Graph", largest_connected_component(graph))
    graph = normalize_node_names(graph)
    return _stabilize_graph(graph)


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure an undirected graph with the same relationships will always be read the same way."""
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    sorted_nodes = graph.nodes(data=True)
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))

    # If the graph is undirected, we create the edges in a stable way, so we get the same results
    # for example:
    # A -> B
    # in graph theory is the same as
    # B -> A
    # in an undirected graph
    # however, this can lead to downstream issues because sometimes
    # consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
    # but they base some of their logic on the order of the nodes, so the order ends up being important
    # so we sort the nodes in the edge in a stable way, so that we always get the same order
    if not graph.is_directed():

        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                temp = source
                source = target
                target = temp
            return source, target, edge_data

        edges = [_sort_source_target(edge) for edge in edges]

    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

    fixed_graph.add_edges_from(edges)
    return fixed_graph


def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """Normalize node names."""
    node_mapping = {node: node.strip() for node in graph.nodes()}  # type: ignore
    return nx.relabel_nodes(graph, node_mapping)