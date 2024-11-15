import numpy as np
import networkx as nx
from typing import List, Tuple
from InfluenceDiffusion.Graph import Graph


def adjacency_matrix_2_dict(adjacency_matrix, vertex_names=None):
    vertex_names = np.arange(len(adjacency_matrix)) if vertex_names is None else vertex_names
    adjacency_dict = {}
    for v, adj_row in enumerate(adjacency_matrix):
        adjacency_dict[v] = vertex_names[adj_row != 0]
    return adjacency_dict


def adjacency_matrix_2_edge_list(adjacency_matrix, vertex_names=None, directed=False):
    vertex_names = np.arange(len(adjacency_matrix)) if vertex_names is None else vertex_names
    edge_set = set()
    for v, adj_row in zip(vertex_names, adjacency_matrix):
        v_neighbors = vertex_names[adj_row != 0]
        for v_adj in v_neighbors:
            if not directed and (v_adj, v) in edge_set:
                continue
            else:
                edge_set.add((v, v_adj))
    return list(edge_set)


def make_prob_adj_matrix(adj_matrix, probs):
    prob_adj_matrix = np.zeros((*adj_matrix.shape[:2], len(probs)))
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 0:
                prob_adj_matrix[i, j] = [1] + [0] * (len(probs) - 1)
            else:
                prob_adj_matrix[i, j] = probs
    return prob_adj_matrix


def plot_graph(graph: Graph, plot_weights=True):
    adj_matrix = graph.get_adj_matrix()
    nx_graph = nx.from_numpy_matrix(adj_matrix.values, create_using=nx.DiGraph)
    pos = nx.spring_layout(nx_graph)

    labels = {i: vertex_name for i, vertex_name in enumerate(adj_matrix.columns)}
    vertex_2_index = {vertex_name: i for i, vertex_name in labels.items()}

    nx.draw(nx_graph, pos, with_labels=True, labels=labels, arrows=True)
    if plot_weights:
        edge_labels = {(vertex_2_index[v1], vertex_2_index[v2]): np.round(weight, 2)
                       for (v1, v2), weight in zip(graph.edge_array, graph.weights)}
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')

    else:
        edge_labels = {(vertex_2_index[v1], vertex_2_index[v2]): edge_index
                       for edge_index, (v1, v2) in enumerate(graph.edge_array)}
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')


def construct_edge_groups(edge_list: list,
                          vertex_communities: list,
                          directed=True) -> List[Tuple]:
    # assumes vertex names are 0..|V|-1
    edge_groups = []
    for (v, v_adj) in edge_list:
        v_com, v_adj_com = vertex_communities[v], vertex_communities[v_adj]
        group = (v_com, v_adj_com) if directed else tuple(sorted((v_com, v_adj_com)))
        edge_groups.append(group)
    return edge_groups


def make_erdos_renyi_graph(n_nodes, p, directed=True, loops_allowed=False, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    adj_matrix = np.random.rand(n_nodes, n_nodes) <= p
    if not loops_allowed:
        np.fill_diagonal(adj_matrix, False)
    edge_list = adjacency_matrix_2_edge_list(adj_matrix, directed=directed)
    return Graph(edge_list, directed=directed)


def make_sbm_graph(com_sizes, p_matrix, directed=True, loops_allowed=False, random_seed=None):
        
    p_matrix = np.array(p_matrix)
    assert len(com_sizes) == len(p_matrix), "Number of communities should be equal to the size of the matrix"
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    adj_matrix_blocks = []
    for com1, com1_size in enumerate(com_sizes):
        adj_matrix_row = []
        for com2, com2_size in enumerate(com_sizes):
            if not directed and com1 > com2:
                block = adj_matrix_blocks[com2][com1].T
            else:
                block = np.random.rand(com1_size, com2_size) <= p_matrix[com1, com2]
            adj_matrix_row.append(block)
        adj_matrix_blocks.append(adj_matrix_row)
    
    adj_matrix = np.block(adj_matrix_blocks).astype(int)
    
    if not loops_allowed:
        np.fill_diagonal(adj_matrix, 0)
    
    return Graph(adjacency_matrix_2_edge_list(adj_matrix), directed=directed)
