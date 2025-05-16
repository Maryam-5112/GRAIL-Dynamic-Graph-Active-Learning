import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import *
from  AL_strategies import *
from models import *
from metrics import *
from Stream_Bootstrap_helpers import *
from Novel_Metrics import *

# Change the working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def evaluate_all_strategies_with_uncertainty(X, y, A, num_days, num_nodes, num_features, L, k, model_class,
                                             metrics_list, n_bootstraps, dynamic_A, num_classes, train_mask,
                                             test_mask, seed=None):
    all_metrics_df = []  # To collect metrics for all strategies

    # Define strategies including each uncertainty variant
    strategies = ['graphpart', 'graphpartfar', 'age', 'coreset', 'featProp', "density",
                  "uncertainty_entropy", "uncertainty_least_confidence", "uncertainty_margin",
                  "pagerank", "degree", "random_sample"]

    # Map uncertainty strategy names to the actual metric used in uncertainty_strategy
    uncertainty_strategies = {
        "uncertainty_entropy": "entropy",
        "uncertainty_least_confidence": "least_confidence",
        "uncertainty_margin": "margin"
    }

    for strategy in strategies:
        print(f"Running bootstrap and evaluation for strategy: {strategy}")
        
        # Determine if the current strategy is an uncertainty-based one
        if strategy.startswith("uncertainty"):
            # Set the uncertainty metric from the mapping
            uncertainty_metric = uncertainty_strategies[strategy]
            metrics_df = bootstrap_and_evaluate(
                X=X, y=y, A=A, num_days=num_days, num_nodes=num_nodes, num_features=num_features,
                L=L, k=k, model_class=model_class, strategy="uncertainty", uncertainity_metric=uncertainty_metric,
                metrics_list=metrics_list, n_bootstraps=n_bootstraps, dynamic_A=dynamic_A,
                num_classes=num_classes, train_mask=train_mask, test_mask=test_mask, seed=seed
            )
        else:
            # For non-uncertainty strategies
            metrics_df = bootstrap_and_evaluate(
                X=X, y=y, A=A, num_days=num_days, num_nodes=num_nodes, num_features=num_features,
                L=L, k=k, model_class=model_class, strategy=strategy, uncertainity_metric=None,
                metrics_list=metrics_list, n_bootstraps=n_bootstraps, dynamic_A=dynamic_A,
                num_classes=num_classes, train_mask=train_mask, test_mask=test_mask, seed=seed
            )

        # Add strategy as a column in the metrics DataFrame
        metrics_df["strategy"] = strategy
        all_metrics_df.append(metrics_df)

    # Combine all strategy DataFrames for this L and k into a single DataFrame
    consolidated_df = pd.concat(all_metrics_df, ignore_index=True)

    return consolidated_df  # Return the combined DataFrame for this specific L and k


def main():

     # Load the data for each label
    data_dict  = load_data('FnF',output_folder='Dataset/FnF')
    # Retrieving data for "sleep_class"
    X = data_dict['sleep_class']['X']
    y_raw = data_dict['sleep_class']['y']
    A = data_dict['sleep_class']['L_friend']

    # Create y_sleep_binary
    y = np.where(y_raw == 2, 1, y_raw)


    num_days, num_nodes, num_features = X.shape
    num_classes = 2

    # Define the metrics list you want to compute
    metrics_list = ['accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc', 'f1_micro', 'f1_macro']

                     # Fixed number of days to train on
    n_bootstraps = 100      # Number of bootstrap iterations
    dynamic_A = False      # Whether the adjacency matrix is dynamic

    train_ratio = 0.8
    # Generate train and test masks (replace num_nodes with the actual number of nodes)
    train_mask, test_mask = generate_train_test_masks(num_nodes, train_ratio=train_ratio)

    # Convert train and test masks to torch.Tensor if they're numpy arrays
    train_mask = torch.tensor(train_mask, dtype=torch.bool) if isinstance(train_mask, np.ndarray) else train_mask
    test_mask = torch.tensor(test_mask, dtype=torch.bool) if isinstance(test_mask, np.ndarray) else test_mask

    # Define the model class to pass in
    model_class = GATModel  

    # Set a base seed for reproducibility
    seed = 42
    # Loop over L values from 1 to 10
    for L_value in range(1, 11):
        print(f"Evaluating for L = {L_value}")
        
        # Initialize an empty DataFrame to collect all results for this L
        all_results_df = pd.DataFrame()
        
        # Loop over each k value for the current L
        for k_value in range(1,16):
            print(f"  Evaluating with k = {k_value}")
            
            # Run the evaluation function and get the DataFrame with results for this L and k
            results_df = evaluate_all_strategies_with_uncertainty(
                X=X, y=y, A=A, num_days=num_days, num_nodes=num_nodes, num_features=num_features,
                L=L_value, k=k_value, model_class=GATModel, metrics_list=metrics_list, n_bootstraps=n_bootstraps,
                dynamic_A=dynamic_A, num_classes=num_classes, train_mask=train_mask, test_mask=test_mask, seed=42
            )
            
            # Add columns for L and k to each result for identification
            results_df["L"] = L_value
            results_df["k"] = k_value
            
            # Append this result to the all_results_df for the current L value
            all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
        
        # After looping through all k values, save the result for this L value in a unique file
        os.makedirs("Sleep_FnF_Results", exist_ok=True)
        filename = f"Sleep_FnF_Results/consolidated_metrics_L_{L_value}.csv"
        all_results_df.to_csv(filename, index=False)
        print(f"Results saved for L = {L_value} in file: {filename}")


if __name__ == "__main__":
    main()