import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Union, List

from InfluenceDiffusion.Trace import Traces, Trace


def make_report_traces(traces: Traces, min_edge_participation=5):
    num_activated_nodes = np.array([len(trace.get_all_activated_vertices()) for trace in traces])
    mean_num_active = np.round(np.mean(num_activated_nodes), 2)
    std_num_active = np.round(np.std(num_activated_nodes), 2)
    print(f"Avg number of activated nodes: {mean_num_active}. Std: {std_num_active}")
    edge_activations = np.vstack([trace.get_attempted_edges_mask() for trace in tqdm(traces)]).sum(0)
    prop_edges_below_min_participation = np.mean(edge_activations <= min_edge_participation)
    print(
        f"Proportion of edges participated in <= {min_edge_participation} traces: {prop_edges_below_min_participation}")
    _, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].hist(edge_activations)
    axs[0].set_xlabel("Num attempts through edge")
    axs[0].set_ylabel("Num edges")
    axs[1].hist(num_activated_nodes, density=True)
    axs[1].set_xlabel("Number of nodes activated in trace")
    axs[1].set_ylabel("Proportion of traces")
    plt.show()
    return edge_activations


def trace_train_test_split(traces: Traces, test_ratio)-> Tuple[Traces, Traces]:
    assert 1 >= test_ratio >= 0
    length_sorted_traces = np.array(sorted(traces, key=lambda trace: len(trace.get_all_activated_vertices())),
                                    dtype=object)
    indices = np.arange(len(traces))
    test_indices = indices[::int(1 / test_ratio)]
    train_indices = list(set(indices) - set(test_indices))
    train_traces = list(length_sorted_traces[train_indices])
    test_traces = list(length_sorted_traces[test_indices])
    return Traces(traces.graph, train_traces), \
           Traces(graph=traces.graph, traces=test_traces)


def extract_vertex_preactivation_masks(traces: Union[Traces, List[Trace]]):
    g = traces[0].graph
    vertex_2_t_tm1_masks = {vertex: ([], []) for vertex in g.get_sinks()}
    vertex_2_parent_mask = {vertex: g.get_parents_mask(vertex) for vertex in g.get_sinks()}
    for trace in traces:
        for vertex in trace.get_all_failed_and_activated_vertices_no_seed():
            if vertex in trace.get_all_activated_vertices_no_seed():
                t_v = trace.get_vertex_activation_time(vertex) - 1
            else:
                t_v = trace.length
            parent_mask = vertex_2_parent_mask[vertex]
            mask_t = trace.get_active_parents_mask_at_time(vertex, time=t_v)[parent_mask]
            mask_tm1 = trace.get_active_parents_mask_at_time(vertex, time=t_v - 1)[parent_mask]
            vertex_2_t_tm1_masks[vertex][0].append(mask_t)
            vertex_2_t_tm1_masks[vertex][1].append(mask_tm1)
    return {vertex: (np.vstack(masks_t), np.vstack(masks_tm1))
            for vertex, (masks_t, masks_tm1) in vertex_2_t_tm1_masks.items() if len(masks_t) > 0}


def compute_new_parent_appearence_times_in_trace(trace: Trace):
    informative_vs = trace.get_all_failed_and_activated_vertices_no_seed()
    activ_time_dict = trace.get_activation_time_dict()
    v_2_new_parent_times = {}
    for v in informative_vs:
        v_time = activ_time_dict[v] if v in activ_time_dict else np.inf
        active_parents = set(trace.graph.get_parents(v)).intersection(activ_time_dict)
        new_parent_appearence_times = set([activ_time_dict[u] for u in active_parents if activ_time_dict[u] < v_time])
        v_2_new_parent_times[v] = new_parent_appearence_times
    return v_2_new_parent_times


def compute_vertex_sample_size(traces: Traces):
    v_2_total_appearences = {v: 0 for v in traces[0].graph.get_sinks()}
    v_2_num_informative_traces = {v: 0 for v in traces[0].graph.get_sinks()}
    for trace in traces:
        v_2_new_parent_times = compute_new_parent_appearence_times_in_trace(trace)
        for v, new_parent_times in v_2_new_parent_times.items():
            v_2_total_appearences[v] += len(new_parent_times)
            v_2_num_informative_traces[v] += 1
    return v_2_total_appearences, v_2_num_informative_traces
