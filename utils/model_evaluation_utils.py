import numpy as np
import matplotlib.pyplot as plt

from typing import List
from time import time
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, roc_curve, auc
from .utils import RMAE, assign_to_bins
import pandas as pd
import seaborn as sns


def compare_metric_over_bins(true_spread: np.array, pred_spreads,
                             metric, pred_names, bins=10, markersize=15, 
                             colors=[ "b", "r", "g", "orange", "violet", "brown"],
                             markers = ["X",  "d", "p","s", "2", "+"]):
    

    pred_spread_array = np.array(pred_spreads)

    bin_assignment = assign_to_bins(true_spread, bins=bins)
    pred_bin_means = np.empty((len(pred_spreads), 0))
    pred_bin_metrics = np.empty((len(pred_spreads), 0))
    
    true_bin_mean = []
    
    for bin_idx in np.unique(bin_assignment):
        bin_mask = bin_assignment == bin_idx
        bin_means = pred_spread_array[:, bin_mask].mean(1).reshape(-1, 1)
        bin_metrics = np.array([metric(true_spread[bin_mask], pred_spread[bin_mask])
                                for pred_spread in pred_spreads]).reshape(-1, 1)

        true_bin_mean.append(true_spread[bin_mask].mean())
        pred_bin_means = np.hstack([pred_bin_means, bin_means])
        pred_bin_metrics = np.hstack([pred_bin_metrics, bin_metrics])

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    for marker, name, pred_spread, color in zip(markers, pred_names, pred_spreads, colors):
        axs[0].scatter(true_spread, pred_spread, s=markersize, alpha=0.5, label=name, c=color, marker=marker)

    axs[0].set_xlabel("Actual spread", fontsize=15)
    axs[0].set_ylabel("Estimated spread", fontsize=15)
    axs[0].plot([np.min(true_bin_mean), np.max(true_bin_mean)],
                [np.min(true_bin_mean), np.max(true_bin_mean)], c="black", linestyle="--")
    axs[0].legend(fontsize=17)

    for name, pred_metric, marker, color in zip(pred_names, pred_bin_metrics, markers, colors):
        axs[1].scatter(true_bin_mean, pred_metric, s=markersize, label=name, marker=marker, c=color)
        axs[1].plot(true_bin_mean, pred_metric, c=color)

    axs[1].set_xlabel("Actual spread", fontsize=15)
    axs[1].set_ylabel("MAE", fontsize=15)
    
    axs[1].legend(fontsize=17)
    plt.title(save_name)
    plt.show()
    

def make_heatmap(values, xy_pairs, x_name="a", y_name="b", title=None):
    assert len(values) == len(xy_pairs)
    xy_pairs_array = np.array(list(xy_pairs))
    xs = sorted(np.unique(xy_pairs_array[:, 0]))
    ys = sorted(np.unique(xy_pairs_array[:, 1]))
    
    df = pd.DataFrame(columns=xs, index=ys, dtype=float)
    for i, (x, y) in enumerate(xy_pairs):
        df.at[y, x] = values[i]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df)
    plt.xlabel(x_name, fontsize=15)
    plt.ylabel(y_name, fontsize=15)
    if title is not None:
        plt.title(title,  fontsize=15)
    plt.show()


def run_experiment(g, influence_model, estimating_models_dict, error_metric=RMAE,
                   n_simulations=5, n_traces=20, seed_size_range=range(1, 6),
                   fit_kwargs={}):
    
    estimation_results = {model: {"times": [], "weights": [], "errors": []} for model in estimating_models_dict}
    
    for _ in tqdm(range(n_simulations)):
        traces = influence_model.sample_traces(n_traces=n_traces, seed_size_range=seed_size_range)
        
        for model_name, estimating_model in estimating_models_dict.items():
            model_kwargs = fit_kwargs[model_name] if model_name in fit_kwargs else {}
            time_start = time()
            estimating_model.fit(traces, **model_kwargs)
            time_end = time()
                          
            estim_weights = estimating_model.weights_
                          
            estimation_results[model_name]["times"].append(time_end - time_start)
            estimation_results[model_name]["weights"].append(estim_weights)
            estimation_results[model_name]["errors"].append(error_metric(g.weights, estim_weights))               

    return estimation_results


def plot_true_vs_pred_weights(true, pred):
    true_linsapce = np.linspace(0, max(true), 100)
    plt.figure(figsize=(10, 10))
    plt.scatter(true, pred)
    plt.plot(true_linsapce, true_linsapce, c="black", linestyle="dashed")
    plt.xlabel("True", fontsize=14)
    plt.ylabel("Predict", fontsize=14)


def plot_weight_distribution(weight_predictions, true_weight, title=None):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    hist_vals, _, _ = plt.hist(weight_predictions, bins=5)
    plt.vlines(true_weight, 0, max(hist_vals), linestyle="dashed", color='black')
    plt.title("Distribution of weight estimates" if title is None else title)
    plt.xlabel("weight estimate")
    plt.ylabel("Count")
    plt.show()    


def plot_error_statistics(error_statistics,
                          traces_range,
                          edge_names_list,
                          figsize=(8, 8),
                          colors=("g", "r", "purple", "b", "orange", "grey", "pink", "yellow"),
                          random_state=1):

    assert len(error_statistics) == 3, "error_statistics should have the form [MSE, Std, Bias]"

    titles = ["MSE vs Trace Number", "Std vs Trace Number", 'Bias (Pred - True) vs Trace Number', ]
    y_labels = ["MSE", "Std", "Bias"]

    n_edges_to_plot = min(len(colors), len(edge_names_list))
    np.random.seed(random_state)
    edge_indices_to_plot = np.random.choice(np.arange(len(edge_names_list)), n_edges_to_plot, replace=False)

    for error_stat, title, y_label, in zip(error_statistics, titles, y_labels):
        plt.figure(figsize=figsize)
        for idx, color in zip(edge_indices_to_plot, colors):
            label = "b" + str(tuple(edge_names_list[idx]))
            plt.plot(traces_range, error_stat[:, idx], label=label, color=color)
        plt.title(title, fontsize=14)
        plt.xlabel("Trace num", fontsize=14)
        plt.xticks(traces_range)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.show()


def plot_estimator_output_vs_variable(variable_2_estimators_outputs, 
                                      output_key, 
                                      markers=None,
                                      colors=None,
                                      linestyles=None,
                                      ylabel=None,
                                      xlabel=None,
                                      n_std=2,
                                      fontsize=14):
    
    if markers is None:
        markers = [None] * len(variable_2_estimators_outputs)
    if colors is None:
        colors = [None] * len(variable_2_estimators_outputs)
    if linestyles is None:
        linestyles = [None] * len(variable_2_estimators_outputs)
        
    variable_range = list(variable_2_estimators_outputs.keys())
    model_names = variable_2_estimators_outputs[variable_range[0]].keys()
    
    for model_name, marker, color, linestyle in zip(model_names, markers, colors, linestyles):

        output_means = np.array([np.mean(variable_2_estimators_outputs[val][model_name][output_key]) 
                                 for val in variable_range])
        output_stds = np.array([np.std(variable_2_estimators_outputs[val][model_name][output_key]) 
                                for val in variable_range])
        
        if np.allclose(output_stds * n_std, 0):
            plt.plot(variable_range, output_means, 
                     label=model_name, c=color, marker=marker, markersize=5, linestyle=linestyle)
        else:
            plt.errorbar(variable_range, output_means, yerr= n_std * output_stds, capsize=5, 
                     label=model_name, c=color, marker=marker, markersize=5, linestyle=linestyle)

    # plt.title(f"Estimator work time vs {variable_name}", fontsize=14)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(output_key if ylabel is None else ylabel, fontsize=fontsize)
    plt.legend(fontsize=fontsize, )
    plt.xticks(variable_range, rotation=45, fontsize=fontsize)
    

def plot_roc_auc(y_true, y_score, label="ROC curve", color="darkorange", sample_weight=None):
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=label + ' (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC')
    plt.legend(loc="lower right")
    

def plot_auprc(y_true, y_score, label="ROC curve", color="darkorange", sample_weight=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, sample_weight=sample_weight)
    auc_score = auc(recall, precision)
    plt.plot(recall, precision, color=color, lw=2, label=label + ' (area = %0.3f)' % auc_score)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("AUPRC")
    plt.legend(loc="upper right")
