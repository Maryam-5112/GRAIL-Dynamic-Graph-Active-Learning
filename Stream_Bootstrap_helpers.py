
import torch
import numpy as np
import os
import pandas as pd
from collections import deque
from metrics import *
from AL_strategies import *
from models import *


def train_and_evaluate_gnn_with_al(X, y, A, num_days, num_nodes, num_features, L, k, burden_gap, model, strategy,uncertainity_metric=None, metrics_list=None, train_mask=None, test_mask=None, num_classes=2):
    """
    Train and evaluate a GNN model with Active Learning (AL) strategies, tracking queried users over the last burden_gap days.

    Parameters:
    - X: Feature matrix.
    - y: Labels.
    - A: Adjacency matrix.
    - num_days: Total number of days.
    - num_nodes: Total number of nodes (users).
    - num_features: Number of features in X.
    - L: Initial number of days to train on.
    - k: Number of users to sample each day.
    - burden_gap: Number of days to avoid re-querying a user. If 0, users are not excluded.
    - model: GNN model to use for training and evaluation.
    - strategy: Active learning strategy.
    - metrics_list: List of metrics to compute.
    - train_mask: Mask for training nodes.
    - test_mask: Mask for test nodes.
    - num_classes: Number of classes (1 for binary classification, > 1 for multi-class).

    Returns:
    - Metrics and queried user log.
    """

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Loss function selection based on number of classes
    criterion = torch.nn.BCELoss() if num_classes == 2 else torch.nn.CrossEntropyLoss()

    # Ensure A is a torch tensor for compatibility
    A = torch.tensor(A) if not isinstance(A, torch.Tensor) else A

    # Convert adjacency matrix to edge index
    edge_index = torch.nonzero(A, as_tuple=False).t().contiguous()

    # # Edge index conversion for PyTorch Geometric
    # edge_index = torch.from_numpy(np.vstack(np.nonzero(A)))

    # Convert X and y to torch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    

    # Initialize train and test masks if not provided
    if train_mask is None:
        train_mask = torch.ones(num_nodes, dtype=bool)  # All nodes for training
    if test_mask is None:
        test_mask = ~train_mask  # Remaining nodes for testing

    queried_users_log = []  # Track all queried users
    train_mask_history = deque(maxlen=max(burden_gap, 1))  # Track queried users daily, even if burden_gap=0

    # Initial training on the first L days
    for day in range(L):
        X_day = X[day]
        y_day = y[day].squeeze()

        model.train()
        optimizer.zero_grad()
        output = model(X_day, edge_index).squeeze()
        
        # Apply train mask for loss calculation
        loss = criterion(output[train_mask], y_day[train_mask]) if num_classes == 2 else criterion(output[train_mask], y_day[train_mask].long())

        loss.backward()
        optimizer.step()

        print(f"Initial training - Day {day + 1}/{L}, Loss: {loss.item():.4f}")

    # Stream processing after L days
    all_metrics = {'train_nodes_next_day': [], 'unqueried_nodes_same_day': [], 'unqueried_nodes_next_day': [], 'test_set_same_day': []}

    for day in range(L, num_days - 1):
        X_day = X[day]
        y_day = y[day].squeeze()

        # Create a temporary train mask if burden_gap > 0
        if burden_gap > 0:
            recent_queried = set(user for day_users in train_mask_history for user in day_users)
            available_train_mask = train_mask.clone()
            available_train_mask[list(recent_queried)] = False
        else:
            available_train_mask = train_mask  # No temporary exclusion if burden_gap = 0

        # Sample users based on the chosen strategy, respecting the available train mask
        
        selected_users =  AL_query(X_day, A, num_classes,  strategy=strategy,uncertainity_metric= uncertainity_metric, num_queries=k, model=model, train_mask=available_train_mask)

        # Log queried users and update train_mask_history
        queried_users_log.append(selected_users.tolist())
        train_mask_history.append(selected_users.tolist())  # Log selected users even if burden_gap=0

        # Train on selected users
        model.train()
        optimizer.zero_grad()
        output = model(X_day, edge_index).squeeze()

        selected_users_mask = torch.zeros(num_nodes, dtype=bool)
        selected_users_mask[selected_users] = True

        # Compute loss only for selected users
        loss = criterion(output[selected_users_mask], y_day[selected_users_mask]) if num_classes == 2 else criterion(output[selected_users_mask], y_day[selected_users_mask].long())
        loss.backward()
        optimizer.step()

        # Evaluation on next day training nodes
        model.eval()
        X_next_day = X[day + 1]
        y_next_day = y[day + 1].squeeze()
        y_pred_train_next_day = model(X_next_day, edge_index).squeeze()

        # Apply mask for next day's selected users
        selected_users_next_day_mask = torch.zeros(num_nodes, dtype=bool)
        selected_users_next_day_mask[selected_users] = True

        y_pred_train_next_day_filtered = y_pred_train_next_day[selected_users_next_day_mask]
        y_true_train_next_day_filtered = y_next_day[selected_users_next_day_mask]

        # Metrics evaluation for different node sets
        if num_classes == 2:
            train_nodes_next_day_metrics = custom_metrics_binary(y_true_train_next_day_filtered.detach().cpu().numpy(), y_pred_train_next_day_filtered.detach().cpu().numpy(), metrics_list)
        else:
            train_nodes_next_day_metrics = custom_metrics_multiclass(y_true_train_next_day_filtered.detach().cpu().numpy(), y_pred_train_next_day_filtered.detach().cpu().numpy(), metrics_list)

        all_metrics['train_nodes_next_day'].append(train_nodes_next_day_metrics)

        # Unqueried nodes performance same day
        y_pred_unqueried_same_day = output[~selected_users_mask & train_mask]
        y_true_unqueried_same_day = y_day[~selected_users_mask & train_mask]

        if num_classes == 2:
            unqueried_same_day_metrics = custom_metrics_binary(y_true_unqueried_same_day.detach().cpu().numpy(), y_pred_unqueried_same_day.detach().cpu().numpy(), metrics_list)
        else:
            unqueried_same_day_metrics = custom_metrics_multiclass(y_true_unqueried_same_day.detach().cpu().numpy(), y_pred_unqueried_same_day.detach().cpu().numpy(), metrics_list)

        all_metrics['unqueried_nodes_same_day'].append(unqueried_same_day_metrics)

        # Unqueried nodes performance next day
        y_pred_unqueried_next_day = y_pred_train_next_day[~selected_users_mask & train_mask]
        y_true_unqueried_next_day = y_next_day[~selected_users_mask & train_mask]

        if num_classes == 2:
            unqueried_next_day_metrics = custom_metrics_binary(y_true_unqueried_next_day.detach().cpu().numpy(), y_pred_unqueried_next_day.detach().cpu().numpy(), metrics_list)
        else:
            unqueried_next_day_metrics = custom_metrics_multiclass(y_true_unqueried_next_day.detach().cpu().numpy(), y_pred_unqueried_next_day.detach().cpu().numpy(), metrics_list)

        all_metrics['unqueried_nodes_next_day'].append(unqueried_next_day_metrics)

        # Test set (holdout set) evaluation same day
        y_pred_test_same_day = output[test_mask]
        y_true_test_same_day = y_day[test_mask]

        if num_classes == 2:
            test_set_same_day_metrics = custom_metrics_binary(y_true_test_same_day.detach().cpu().numpy(), y_pred_test_same_day.detach().cpu().numpy(), metrics_list)
        else:
            test_set_same_day_metrics = custom_metrics_multiclass(y_true_test_same_day.detach().cpu().numpy(), y_pred_test_same_day.detach().cpu().numpy(), metrics_list)

        all_metrics['test_set_same_day'].append(test_set_same_day_metrics)

    return all_metrics, queried_users_log



def generate_train_test_masks(num_nodes, train_ratio=0.8):
    """
    Generates random train and test masks for the nodes of a graph.
    
    Args:
        num_nodes (int): Total number of nodes in the graph.
        train_ratio (float): The ratio of nodes used for training.
        
    Returns:
        train_mask (np.array): Boolean mask for the training nodes.
        test_mask (np.array): Boolean mask for the test nodes.
    """
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * num_nodes)
    
    train_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    return train_mask, test_mask




def bootstrap_and_evaluate(X, y, A, num_days, num_nodes, num_features, L, k, model_class, strategy='random',
                           uncertainity_metric=None, metrics_list=None, n_bootstraps=10, dynamic_A=False,
                           num_classes=2, train_mask=None, test_mask=None, seed=None, save_folder="results"):
    
    """
    Perform bootstrapping and evaluate Active Learning strategies over multiple days.

    Parameters:
    - X: The feature matrix.
    - y: The labels.
    - A: The adjacency matrix (should be 2D).
    - num_days: Total number of days.
    - num_nodes: Total number of nodes (users).
    - num_features: Number of features in X.
    - L: Initial number of days to train on.
    - k: Number of users to sample each day.
    - model_class: The GNN model class (uninitialized) to use for training and evaluation.
    - strategy: Active learning strategy.
    - metrics_list: List of metrics to compute.
    - n_bootstraps: Number of bootstrap iterations.
    - dynamic_A: Boolean flag to indicate whether the adjacency matrix is static or changes over time (default is False).
    - num_classes: Number of classes (1 for binary classification, > 1 for multi-class).
    - train_mask: Mask for training nodes.
    - test_mask: Mask for test nodes.
    - seed: Base seed to ensure reproducibility. Each bootstrap will use `seed + i` for iteration `i`.
    - save_folder (str): The base folder to save user logs.

    Returns:
    - DataFrame containing bootstrap metrics.
    """
    

    # Set up the user log folder for each strategy
    user_log_folder = os.path.join(save_folder, "user_log", strategy)
    os.makedirs(user_log_folder, exist_ok=True)

    # Initialize lists to collect metrics and logs for each bootstrap iteration
    data = []  # Metrics collection for DataFrame
    user_sampling_log = []  # Track user sampling information

    for i in range(n_bootstraps):
        print(f"Bootstrap iteration {i + 1}/{n_bootstraps}")
        
        # Set unique random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed + i)
            np.random.seed(seed + i)

        # Initialize a new model instance for each bootstrap iteration
        out_channels = 1 if num_classes == 2 else num_classes
        model = model_class(num_features, hidden_channels=32, out_channels=out_channels, num_heads=2,
                            num_layers=3, dropout=0.5, activation="relu", batchnorm=True)

        # Call the train_and_evaluate_gnn_with_al function
        metrics, queried_users_log = train_and_evaluate_gnn_with_al(
            X, y, A, num_days, num_nodes, num_features, L, k, burden_gap=0, model=model,
            strategy=strategy, uncertainity_metric=uncertainity_metric, metrics_list=metrics_list,
            train_mask=train_mask, test_mask=test_mask, num_classes=num_classes
        )

        # Add each day's metrics to data with L and k for tracking
        for node_type, days in metrics.items():
            for day_index, day_metrics in enumerate(days, start=1):
                row = {
                    'bootstrap_iteration': i + 1, 'node_type': node_type, 'day': day_index,
                    'L': L, 'k': k  # Include L and k in each row for easy tracking
                }
                row.update(day_metrics)
                data.append(row)

        # Track user sampling for each day with the correct graph structure
        daily_log = []
        for day_index, sampled_users in enumerate(queried_users_log):
            adj_matrix = A[day_index] if dynamic_A else A
            daily_log.append({
                'day': day_index,
                'sampled_users': sampled_users,
                'adjacency_matrix': adj_matrix
            })
        user_sampling_log.append(daily_log)

    # Create a single DataFrame with all metrics data (without saving individual files)
    metrics_df = pd.DataFrame(data)

    # Save user logs in the user_log folder
    for i, daily_log in enumerate(user_sampling_log, start=1):
        user_log_file_path = os.path.join(user_log_folder, f"bootstrap_{i}_user_log.csv")
        pd.DataFrame(daily_log).to_csv(user_log_file_path, index=False)

    return metrics_df

