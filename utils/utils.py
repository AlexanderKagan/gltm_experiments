import numpy as np
from typing import Iterable, Set, List
import matplotlib.pyplot as plt


def make_set_intersection_table(seeds, names):
    seeds = [set(seed) for seed in seeds]
    intersection_table = np.zeros((len(seeds), len(seeds)), dtype=int)
    for i in range(len(seeds)):
        for j in range(len(seeds)):

            intersection_table[i, j] = len(seeds[i].intersection(seeds[j]))
    return pd.DataFrame(intersection_table, columns=names, index=names)


def assign_to_bins(array, bins=10):
    bin_size = int(np.ceil(len(array) / bins))
    sorted_indices = np.argsort(array)
    bin_assignment = np.zeros_like(array)
    for bin_idx, start_idx in enumerate(range(0, len(array), bin_size)):
        end_idx = start_idx + bin_size
        bin_assignment[sorted_indices[start_idx: end_idx]] = bin_idx
    return bin_assignment


def get_appearance_indices_dict(list_of_sets: List[Set]):
    seen_elems = set()
    elem_2_appearance_indices = {}
    for idx, cur_set in enumerate(list_of_sets):
        for elem in cur_set:
            if elem in seen_elems:
                elem_2_appearance_indices[elem].append(idx)
            else:
                elem_2_appearance_indices[elem] = [idx]
        seen_elems.update(cur_set)
    return elem_2_appearance_indices


def make_distrib_dict(com_sizes: list, distribs: list):
    assert len(com_sizes) == len(distribs)
    distrib_dict = {}
    cum_sizes = np.cumsum([0] + com_sizes)
    for distrib, start_idx, end_idx in zip(distribs, cum_sizes[:-1], cum_sizes[1:]):
        for vertex in range(start_idx, end_idx):
            distrib_dict[vertex] = distrib
    return distrib_dict


def plot_distribution(distrib, n_points=100, ub=5, lb=-5):
    sup_min, sup_max = distrib.support()
    xs = np.linspace(max(sup_min, lb), min(sup_max, ub), n_points)
    plt.plot(xs, distrib.pdf(xs))
    plt.xlabel("x", fontsize=10)
    plt.ylabel("PDF",  fontsize=10)


def multiple_union(set_list: Iterable[Set]):
    final_set = set()
    for cur_set in set_list:
        final_set = final_set.union(cur_set)
    return final_set


def relative_mean_absolute_error(true, pred):
    return np.sum(np.abs(true - pred)) / np.sum(np.abs(true))


def bernoulli(p):
    return np.random.choice([1, 0], p=[p, 1-p])


def make_name_from_dict(dic):
    name = ''
    for var, val in dic.items():
        name += (var + '_' + str(val) + '_')
    return name[:-1]


def root_mean_squared_error(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def invert_non_zeros(array):
    out = np.array(array, dtype=float)
    non_zero_mask = array != 0
    out[non_zero_mask] = 1. / out[non_zero_mask]
    return out


def random_vector_inside_simplex(dim, ub=1):
    U = np.random.uniform(low=0, high=ub, size=dim)
    U_sorted = np.sort(U)
    U_sorted = np.concatenate(([0], U_sorted))
    x = np.diff(U_sorted)
    return x


def random_vector_on_simplex(dim, ub=1):
    X = random_vector_inside_simplex(dim=dim, ub=1)
    return X / np.sum(X) * ub


def RMAE(true, pred):
    return np.linalg.norm(true - pred, ord=1) / np.linalg.norm(true, ord=1)
