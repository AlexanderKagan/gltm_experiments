import numpy as np
import pandas as pd
from InfluenceDiffusion.Graph import Graph
from tqdm import tqdm
from utils.utils import multiple_union
from collections import defaultdict
from sklearn.model_selection import train_test_split


def construct_pseudo_traces(action_df: pd.DataFrame, graph: Graph):
    assert {"action", "user", "time"}.issubset(action_df.columns)
    pseudo_traces = defaultdict(list)
    all_users = action_df["user"].unique()
    user_2_parents = {user: graph.get_parents(user) for user in all_users}
    user_2_children = {user: graph.get_children(user) for user in all_users}
    
    for action in tqdm(action_df["action"].unique()):
        action_mask = action_df["action"] == action
        action_subdf = action_df[action_mask]
        all_action_users = set(action_subdf["user"])

        for _, (user, user_time) in action_subdf[["user", "time"]].iterrows():
            prev_action_users = set(action_subdf["user"][action_subdf["time"] < user_time])
            action_parents = prev_action_users & user_2_parents[user]
            if not action_parents:
                continue
            pseudo_traces[user].append((set(), action_parents))
            
        failed_users = multiple_union([user_2_children[v] for v in all_action_users]) - all_action_users
        for user in failed_users:
            action_parents = user_2_parents[user] & all_action_users
            pseudo_traces[user].append((action_parents, set()))
            
    return pseudo_traces 


def pseudo_trace_train_test_split(pseudo_traces: dict[int, list[tuple[set, set]]], 
                                  test_prop: float = 0.2, random_state=None):
    train_pseudo_traces = {}
    test_pseudo_traces = {}
    for vertex, vertex_traces in pseudo_traces.items():
        y = [len(s2) > 0 for s1, s2 in vertex_traces]
        train_v_traces, test_v_traces = train_test_split(vertex_traces, 
                                                         stratify=y if np.sum(y) >= 2 else None, 
                                                         test_size=test_prop, 
                                                         random_state=random_state)
  
        train_pseudo_traces[vertex] = train_v_traces
        test_pseudo_traces[vertex] = test_v_traces
    return train_pseudo_traces, test_pseudo_traces


def compute_edge_stats_from_pseudo_traces(pseudo_traces, edges):
    edge_2_participation = {tuple(edge): [0, 0] for edge in edges}
    for v, v_pseudo_traces in pseudo_traces.items():
        for A_tm1, D_t in v_pseudo_traces:
            if D_t:
                for u in A_tm1 | D_t:
                    edge_2_participation[(u, v)][0] += 1
            else:
                for u in A_tm1:
                    edge_2_participation[(u, v)][1] += 1
    return edge_2_participation
