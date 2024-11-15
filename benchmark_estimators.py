import numpy as np
from typing import List, Union
from InfluenceDiffusion.Trace import Traces, PseudoTraces
from InfluenceDiffusion.Graph import Graph
from utils.utils import invert_non_zeros
from utils.pseudo_trace_utils import compute_edge_stats_from_pseudo_traces


def compute_num_traces_sink_activates_after_source(g: Graph, traces: Traces):
    weights = np.array(
        [np.sum([trace.get_vertex_activation_time(u) < trace.get_vertex_activation_time(v)
                 for trace in traces if {u, v}.issubset(trace.get_all_activated_vertices())])
         for u, v in g.get_edges()]
    )
    return weights


def propagated_trace_number_weight_estimator(g: Graph, traces: Traces):
    weights = compute_num_traces_sink_activates_after_source(g, traces)
    indegrees = np.array([np.sum(weights[g.get_parents_mask(v)]) for _, v in g.get_edges()])
    return weights * invert_non_zeros(indegrees)


def propagated_trace_proportion_weight_estimator(g: Graph, traces: Union[Traces, PseudoTraces]):
    if isinstance(traces, Traces):
        weights = compute_num_traces_sink_activates_after_source(g, traces)
        vertex_2_num_activations = {v: np.sum([v in trace.get_all_activated_vertices() for trace in traces])
                                    for v in g.get_vertices()}
        weights = weights * invert_non_zeros(np.array([vertex_2_num_activations[u] for u, _ in g.get_edges()]))
       
    elif isinstance(traces, PseudoTraces):
        edge_2_pos_neg_appearences = compute_edge_stats_from_pseudo_traces(traces, g.edge_array)
        weights = []
        for edge in g.edge_array:
            edge = tuple(edge)
            if edge_2_pos_neg_appearences[edge] != [0, 0]:
                pos, neg = edge_2_pos_neg_appearences[edge]
                weights.append(pos / (pos + neg))
            else:
                weights.append(0)
        weights = np.array(weights)
    else:
        raise NotImplementedError("Only Traces and PseudoTraces types are supported")
        
    indegrees = np.array([weights[g.get_parents_mask(v)].sum() for _, v in g.get_edges()])
    return weights * invert_non_zeros(indegrees)
