import os
import torch
import gc
import json
import argparse
import pandas as pd
import numpy as np

from data_loader import *
from  AL_strategies import *
from models import *
from metrics import *
from Stream_Bootstrap_helpers import *
from Novel_Metrics_and_Plotting_utils import *

import warnings
warnings.filterwarnings("ignore")





# Change the working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def evaluate_and_plot_all_strategies_with_uncertainty(X, y, A, num_days, num_nodes, num_features, L, k, model_class,
                                                      metrics_list, n_bootstraps, dynamic_A, num_classes, train_mask,
                                                      test_mask, seed=None, save_folder="results", model_params=None):
    all_metrics_df = []  # To collect metrics for all strategies

    # Set up a folder for consolidated metrics and user logs
    metrics_folder = os.path.join(save_folder, "metrics")
    os.makedirs(metrics_folder, exist_ok=True)

    # Define strategies including each uncertainty variant and the 'no_AL' strategy
    strategies = ['graphpart', 'age', 'coreset', 'featProp', "density", 'graphpartfar',
                  "uncertainty_entropy", "uncertainty_least_confidence", "uncertainty_margin",
                  "pagerank", "degree", "no_AL","random_sample","upper_bound_AL",]
    
    # Map uncertainty strategy names to the actual metric used in uncertainty_strategy
    uncertainty_strategies = {
        "uncertainty_entropy": "entropy",
        "uncertainty_least_confidence": "least_confidence",
        "uncertainty_margin": "margin"
    }

    for strategy in strategies:
        print(f"Running bootstrap and evaluation for strategy: {strategy}")
        
        # Adjust strategy and k for 'no_AL'
        if strategy == 'no_AL':
            current_strategy =  "random_sample"
            current_k = 0
        else:
            current_strategy = strategy
            current_k = k

        # Determine if the current strategy is an uncertainty-based one
        if current_strategy.startswith("uncertainty"):
            # Set the uncertainty metric from the mapping
            uncertainty_metric = uncertainty_strategies[current_strategy]
            metrics_df = bootstrap_and_evaluate(
                X=X, y=y, A=A, num_days=num_days, num_nodes=num_nodes, num_features=num_features,
                L=L, k=current_k, model_class=model_class, strategy="uncertainty", uncertainity_metric=uncertainty_metric,
                metrics_list=metrics_list, n_bootstraps=n_bootstraps, dynamic_A=dynamic_A,
                num_classes=num_classes, train_mask=train_mask, test_mask=test_mask, seed=seed,
                save_folder=save_folder, model_params=model_params  # Pass model_params here
            )
        else:
            # For non-uncertainty strategies including 'no_AL' which uses 'random'
            metrics_df = bootstrap_and_evaluate(
                X=X, y=y, A=A, num_days=num_days, num_nodes=num_nodes, num_features=num_features,
                L=L, k=current_k, model_class=model_class, strategy=current_strategy, uncertainity_metric=None,
                metrics_list=metrics_list, n_bootstraps=n_bootstraps, dynamic_A=dynamic_A,
                num_classes=num_classes, train_mask=train_mask, test_mask=test_mask, seed=seed,
                save_folder=save_folder, model_params=model_params  # Pass model_params here
            )

        # Add strategy as a column in the metrics DataFrame
        metrics_df["strategy"] = strategy  # Keep original strategy name for clarity
        all_metrics_df.append(metrics_df)

    # Combine all DataFrames
    consolidated_df = pd.concat(all_metrics_df, ignore_index=True)

    # Save the consolidated metrics DataFrame for all strategies in the metrics folder
    consolidated_metrics_path = os.path.join(metrics_folder, "consolidated_metrics_all_strategies_with_baselines.csv")
    consolidated_df.to_csv(consolidated_metrics_path, index=False)



def process_cohorts_and_evaluate(data_dict, selected_cohorts, L_values, k_values, num_classes, metrics_list, 
                                 n_bootstraps, dynamic_A, train_ratio, model_class, model_params, seed, base_folder):
    """
    Process specified cohorts and evaluate using evaluate_and_plot_all_strategies_with_uncertainty.
    """
    os.makedirs(base_folder, exist_ok=True)

    for cohort_id, cohort_data in data_dict.items():
        if cohort_id not in selected_cohorts:
            continue  # Skip if the cohort is not in the selected list

        # Extract data for the cohort
        X = cohort_data['X']  # Feature tensor: days x users x features
        y = cohort_data['y']  # Label tensor: days x users x 1
        A = cohort_data['A']  # Adjacency matrix: users x users
        print(f"Processing Cohort: {cohort_id}, Data Shape: {X.shape}")

        num_days, num_nodes, num_features = X.shape

        # Create folder for the cohort
        cohort_folder = os.path.join(base_folder, f"Cohort_{cohort_id}")
        os.makedirs(cohort_folder, exist_ok=True)

        for L in L_values:
            if L > num_days:
                print(f"Skipping L={L} for Cohort {cohort_id} as it exceeds the number of available days ({num_days}).")
                continue

            for k in k_values:
                if k > num_nodes:
                    print(f"Skipping k={k} for Cohort {cohort_id} as it exceeds the number of nodes ({num_nodes}).")
                    continue

                # Create folder for L and k
                experiment_folder = os.path.join(cohort_folder, f"L_{L}_k_{k}_nodes_{num_nodes}")
                os.makedirs(experiment_folder, exist_ok=True)

                print(f"Processing Cohort: {cohort_id}, L: {L}, k: {k} (Cohort size: {num_nodes})")

                # Generate train and test masks
                train_mask, test_mask = generate_train_test_masks(num_nodes, train_ratio=train_ratio)
                train_mask = torch.tensor(train_mask, dtype=torch.bool) if isinstance(train_mask, np.ndarray) else train_mask
                test_mask = torch.tensor(test_mask, dtype=torch.bool) if isinstance(test_mask, np.ndarray) else test_mask

                # Run evaluation
                evaluate_and_plot_all_strategies_with_uncertainty(
                    X, y, A, num_days, num_nodes, num_features, L, k, model_class,
                    metrics_list, n_bootstraps, dynamic_A, num_classes, train_mask,
                    test_mask, seed=seed, save_folder=experiment_folder, model_params=model_params
                )

                # Save cohort metadata for proportion calculation
                metadata_path = os.path.join(experiment_folder, "metadata.json")
                metadata = {
                    "cohort_id": cohort_id,
                    "L": L,
                    "k": k,
                    "num_nodes": num_nodes,
                    "num_days": num_days
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

                # Free up memory after each iteration
                del train_mask, test_mask
                gc.collect()
                torch.cuda.empty_cache()  # If using GPU

    print("All experiments completed and results saved.")


def run_func(data_path, base_folder, selected_cohorts, L_values, k_values):
    """
    Function to process cohorts and run experiments.

    Args:
        data_path (str): Path to cohort data file.
        base_folder (str): Base folder for saving results.
        selected_cohorts (list): List of cohort IDs to process.
        L_values (list): List of L values (initial training days).
        k_values (list): List of k values (queried nodes).
    """
    # Load data
    data_dict = load_data(dataset='SNAPSHOT',output_folder=None, file_path='Dataset/SNAPSHOT/SNAPSHOT_Processed_all_Cohort.pkl')

    # Define experiment parameters
    num_classes = 2
    metrics_list = ['accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc', 'f1_micro', 'f1_macro']
    n_bootstraps = 25
    dynamic_A = False
    train_ratio = 0.8
    seed = 42

    # Model parameters
    model_class = GAT_custom_hidden
    model_params = {
        'hidden_channels': [48, 32, 16],
        'num_layers': 4,
        'num_heads': 1,
        'dropout': 0.4,
        'activation': "relu",
        'batchnorm': True
    }

    # Process cohorts and evaluate
    process_cohorts_and_evaluate(
        data_dict=data_dict,
        selected_cohorts=selected_cohorts,
        L_values=L_values,
        k_values=k_values,
        num_classes=num_classes,
        metrics_list=metrics_list,
        n_bootstraps=n_bootstraps,
        dynamic_A=dynamic_A,
        train_ratio=train_ratio,
        model_class=model_class,
        model_params=model_params,
        seed=seed,
        base_folder=base_folder
    )


if __name__ == "__main__":
    # Define default parameters
    data_path = "Dataset/gnn_data_all_cohorts.pkl"
    base_folder = "SNAPSHOT_Sleep_results"
    selected_cohorts = [2, 3, 4, 5, 6, 7]  # Cohort IDs to process
    L_values = [5, 6, 7, 8, 9, 10, 3, 4]  # Initial training days
    k_values = [3, 4, 5, 6, 7, 8, 9, 10]  # Queried nodes

    # Run the function
    run_func(data_path, base_folder, selected_cohorts, L_values, k_values)


