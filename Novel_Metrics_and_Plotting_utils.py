import pandas as pd
import numpy as np
from sklearn.metrics import auc
import os
from scipy.stats import entropy
import matplotlib.pyplot as plt
import pickle 
import seaborn as sns
from collections import defaultdict


def compute_cpi_summary(df, label_name):
    """
    Computes the CPI (normalized AUC) for each strategy and node type,
    and returns a summary DataFrame with mean and standard deviation of CPI.

    Parameters:
    - df: DataFrame containing accuracy over days for each strategy and bootstrap.
    - label_name: Label name for identification in output DataFrame.

    Returns:
    - summary_df: DataFrame with columns ['label', 'strategy', 'node_type', 'mean_CPI', 'std_CPI'].
    """
    # List to store CPI results
    cpi_results = []

    # Loop over all unique node types in the DataFrame
    for node_type in df['node_type'].unique():
        node_df = df[df['node_type'] == node_type]

        for strategy in node_df['strategy'].unique():
            strategy_df = node_df[node_df['strategy'] == strategy]
            cpi_values = []

            # Calculate CPI for each bootstrap
            for bootstrap in strategy_df['bootstrap_iteration'].unique():
                bootstrap_df = strategy_df[strategy_df['bootstrap_iteration'] == bootstrap]

                # Ensure days are sorted for CPI calculation
                bootstrap_df = bootstrap_df.sort_values(by='day')
                days = bootstrap_df['day'].values
                accuracy = bootstrap_df['accuracy'].values

                # Calculate AUC for this bootstrap
                auc_value = auc(days, accuracy)

                # Normalize AUC by dividing by the maximum possible AUC (number of days) to get CPI
                max_auc = days[-1] - days[0] + 1  # Assuming day sequence is continuous
                cpi = auc_value / max_auc
                cpi_values.append(cpi)

            # Calculate mean and standard deviation of CPI for this strategy and node type
            mean_cpi = np.mean(cpi_values)
            std_cpi = np.std(cpi_values)

            # Append results to the list
            cpi_results.append({
                'label': label_name,
                'strategy': strategy,
                'node_type': node_type,
                'mean_CPI': mean_cpi,
                'std_CPI': std_cpi
            })

    # Convert results to DataFrame
    summary_df = pd.DataFrame(cpi_results)
    return summary_df


def compute_and_plot_diversity_metrics(base_folder, label_name, strategies):
    """
    Computes and visualizes diversity metrics (Sampling Entropy, Coverage Ratio, Average Time Gap)
    for each Active Learning strategy across multiple bootstraps.

    Parameters:
        base_folder (str): Path to the base directory containing user log folders.
        label_name (str): Label name used for the folder naming convention.
        strategies (list): List of AL strategies to evaluate.

    Returns:
        pd.DataFrame: DataFrame containing the computed diversity metrics for each strategy.
    """
    
    # List to store diversity metrics for each strategy and bootstrap
    all_metrics = []

    for strategy in strategies:
        # Define the path for the strategy's user log folder
        strategy_folder = os.path.join(base_folder, f"{label_name}_L3_k3", "user_log", strategy)
        
        if not os.path.exists(strategy_folder):
            print(f"Strategy folder not found: {strategy_folder}")
            continue

        # Loop through each bootstrap log file in the strategy folder
        for filename in os.listdir(strategy_folder):
            if filename.startswith("bootstrap_") and filename.endswith("_user_log.csv"):
                # Load the user log for the current bootstrap
                file_path = os.path.join(strategy_folder, filename)
                user_log_df = pd.read_csv(file_path)

                print(f"Processing file: {file_path}")

                # Determine the number of nodes from adjacency matrix shape
                num_nodes = user_log_df['adjacency_matrix'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ").shape[0]).iloc[0]

                # Flatten the list of sampled users
                all_sampled_users = []
                for sampled_users in user_log_df['sampled_users']:
                    all_sampled_users.extend(eval(sampled_users))

                # Create a frequency series, including all nodes with zero frequency
                user_freq = pd.Series(all_sampled_users).value_counts(normalize=True).reindex(range(num_nodes), fill_value=0)

                # Calculate entropy of the user sampling distribution
                sampling_entropy = entropy(user_freq)
                
                # Calculate the maximum possible entropy for comparison (uniform distribution)
                max_entropy = np.log(num_nodes)

                # Calculate coverage ratio (unique users sampled divided by total number of nodes)
                total_users_sampled = (user_freq > 0).sum()  # Unique users sampled
                coverage_ratio = total_users_sampled / num_nodes

                # Calculate time gaps for each user
                time_gaps = {}
                for user in user_freq.index:
                    if user_freq[user] > 0:  # Only if user was sampled
                        user_days = user_log_df[user_log_df['sampled_users'].apply(lambda x: user in eval(x))]['day'].tolist()
                        if len(user_days) > 1:
                            time_gaps[user] = np.mean(np.diff(user_days))
                        else:
                            time_gaps[user] = np.nan  # Not enough data points for gap

                # Average time gap for the current bootstrap
                avg_time_gap = np.nanmean(list(time_gaps.values()))

                # Store metrics for the current bootstrap and strategy
                all_metrics.append({
                    'strategy': strategy,
                    'bootstrap': filename.split('_')[1],
                    'entropy': sampling_entropy,
                    'max_entropy': max_entropy,
                    'coverage_ratio': coverage_ratio,
                    'avg_time_gap': avg_time_gap
                })

    # Convert the list of dictionaries into a DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Save diversity metrics DataFrame
    diversity_metrics_path = os.path.join(base_folder, f"{label_name}_L3_k3", 'diversity_metrics_per_bootstrap.pkl')
    with open(diversity_metrics_path, 'wb') as f:
        pickle.dump(metrics_df, f)

    print(f"Diversity metrics computed and saved in {diversity_metrics_path}")

    # Plot boxplots for diversity metrics across bootstraps
    plt.figure(figsize=(15, 5))
    
    # Entropy plot with max entropy in the title
    plt.subplot(1, 3, 1)
    sns.boxplot(x='strategy', y='entropy', data=metrics_df)
    plt.title(f'Sampling Entropy (Max = {metrics_df["max_entropy"].iloc[0]:.2f})')
    plt.xticks(rotation=90)

    # Coverage Ratio plot
    plt.subplot(1, 3, 2)
    sns.boxplot(x='strategy', y='coverage_ratio', data=metrics_df)
    plt.title('Coverage Ratio')
    plt.xticks(rotation=90)
    # Average Time Gap plot
    plt.subplot(1, 3, 3)
    sns.boxplot(x='strategy', y='avg_time_gap', data=metrics_df)
    plt.title('Average Time Gap Between Sampling')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return metrics_df



def plot_time_avg_performance(df, label_name):
    """
    Plots the average performance (accuracy) of different Active Learning (AL) strategies 
    over time for each node type in the given dataset.

    Parameters:
        df (pd.DataFrame): DataFrame containing performance data with columns ['node_type', 'day', 'accuracy', 'strategy'].
        label_name (str): Label name used for the plot title (e.g., dataset name).

    Returns:
        None
    """
    
    # Get unique node types from the dataframe
    node_types = df['node_type'].unique()
    
    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Average Performance (Accuracy) Over Days for {label_name}', fontsize=16)
    
    # Loop through each node type and create a plot in the 2x2 grid
    for i, node_type in enumerate(node_types):
        ax = axes[i // 2, i % 2]  # Calculate grid position
        sns.lineplot(
            data=df[df['node_type'] == node_type],
            x='day',
            y='accuracy',
            hue='strategy',
            estimator='mean',
            ax=ax
        )
        ax.set_title(f'Node Type: {node_type}')
        ax.set_xlabel('Day')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Strategy')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the main title
    plt.show()





def compute_cpi_df(results_df, metric='f1_macro'):
    """
    Function to compute CPI for each unique combination of L and k.

    Args:
    - results_df (pd.DataFrame): Input DataFrame with results.
    - metric (str): Metric to compute CPI, default is 'f1_macro'.

    Returns:
    - cpi_df (pd.DataFrame): Consolidated CPI DataFrame.
    """
    # List to store CPI results
    cpi_results = []

    # Loop through unique combinations of L and k
    for L_value in results_df['L'].unique():
        for k_value in results_df['k'].unique():
            # Filter subset of the DataFrame
            subset_df = results_df[(results_df['L'] == L_value) & (results_df['k'] == k_value)]

            # Compute CPI summary using the provided metric
            cpi_summary = compute_cpi_summary(subset_df, f"L={L_value}, k={k_value}", metric=metric)

            # Add metadata for L and k
            cpi_summary['L'] = L_value
            cpi_summary['k'] = k_value

            # Append to results list
            cpi_results.append(cpi_summary)

    # Combine all CPI results into a single DataFrame
    cpi_df = pd.concat(cpi_results, ignore_index=True)
    return cpi_df





# Function to analyze time metrics based on log files
def analyze_time_metrics_for_strategies(base_folder, label_name, graph_type, num_nodes,L=3, k=5, gap_threshold=2):
    folder_path = os.path.join(
        base_folder,
        label_name,
        graph_type,
        f"L_{L}_k_{k}",
        "user_log"
    )

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None

    # List all strategies within the user_log directory
    strategies = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    all_metrics = []

    for strategy in strategies:
        strategy_folder = os.path.join(folder_path, strategy)

        for filename in os.listdir(strategy_folder):
            if filename.startswith("bootstrap_") and filename.endswith("_user_log.csv"):
                file_path = os.path.join(strategy_folder, filename)
                user_log_df = pd.read_csv(file_path)
                
                # Flatten sampled users
                all_sampled_users = []
                for sampled_users in user_log_df['sampled_users']:
                    all_sampled_users.extend(eval(sampled_users))

                # Frequency series with zero frequency included
                user_freq = pd.Series(all_sampled_users).value_counts(normalize=True).reindex(range(num_nodes), fill_value=0)

                # Entropy and coverage ratio
                sampling_entropy = entropy(user_freq)
                max_entropy = np.log(num_nodes)
                total_users_sampled = (user_freq > 0).sum()
                coverage_ratio = total_users_sampled / num_nodes

                # Time gaps per user
                time_gaps = defaultdict(list)
                for user in range(num_nodes):
                    user_days = user_log_df[user_log_df['sampled_users'].apply(lambda x: user in eval(x))]['day'].tolist()
                    if len(user_days) > 1:
                        time_gaps[user] = np.mean(np.diff(user_days))

                avg_time_gap = np.nanmean(list(time_gaps.values()))

                # Calculate back-to-back and within-gap sampling
                back_to_back_sampling = np.sum(np.array(list(time_gaps.values())) == 1) / num_nodes * 100
                within_gap_sampling = np.sum(np.array(list(time_gaps.values())) <= gap_threshold) / num_nodes * 100
                over_exertion_score = within_gap_sampling

                # Extract the bootstrap iteration number
                bootstrap_iter = filename.split('_')[1]

                all_metrics.append({
                    'strategy': strategy,
                    'bootstrap_iter': bootstrap_iter,  # Add bootstrap iteration
                    'entropy': sampling_entropy,
                    'max_entropy': max_entropy,
                    'coverage_ratio': coverage_ratio,
                    'avg_time_gap': avg_time_gap,
                    'back_to_back_sampling': back_to_back_sampling,
                    'within_gap_sampling': within_gap_sampling,
                    'over_exertion_score': over_exertion_score
                })

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df


import os
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import entropy

def analyze_time_metrics_for_strategiesSNAP(base_folder, label_name, graph_type, num_nodes, L=3, k=5, gap_threshold=2):
    """
    Analyze time metrics based on log files for various strategies.

    Args:
    - base_folder (str): Base folder containing results.
    - label_name (str): Label type for filtering results.
    - graph_type (str): Graph type for specific analysis.
    - num_nodes (int): Total number of nodes in the graph.
    - L (int): Specific L value (default: 3).
    - k (int): Specific k value (default: 5).
    - gap_threshold (int): Threshold for within-gap sampling (default: 2).

    Returns:
    - metrics_df (pd.DataFrame): Consolidated DataFrame with calculated metrics.
    """
    folder_path = os.path.join(
        base_folder,
        label_name,
        graph_type,
        f"L_{L}_k_{k}_nodes_{num_nodes}",  # Updated folder naming convention
        "user_log"
    )

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None

    # List all strategies within the user_log directory
    strategies = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    all_metrics = []

    for strategy in strategies:
        strategy_folder = os.path.join(folder_path, strategy)

        for filename in os.listdir(strategy_folder):
            if filename.startswith("bootstrap_") and filename.endswith("_user_log.csv"):
                file_path = os.path.join(strategy_folder, filename)
                user_log_df = pd.read_csv(file_path)
                
                # Flatten sampled users
                all_sampled_users = []
                for sampled_users in user_log_df['sampled_users']:
                    try:
                        all_sampled_users.extend(eval(sampled_users))
                    except Exception as e:
                        print(f"Error parsing sampled_users in file {file_path}: {e}")
                        continue

                # Frequency series with zero frequency included
                user_freq = pd.Series(all_sampled_users).value_counts(normalize=True).reindex(range(num_nodes), fill_value=0)

                # Entropy and coverage ratio
                sampling_entropy = entropy(user_freq, base=np.e)
                max_entropy = np.log(num_nodes)
                total_users_sampled = (user_freq > 0).sum()
                coverage_ratio = total_users_sampled / num_nodes

                # Time gaps per user
                time_gaps = defaultdict(list)
                for user in range(num_nodes):
                    user_days = user_log_df[user_log_df['sampled_users'].apply(lambda x: user in eval(x))]['day'].tolist()
                    if len(user_days) > 1:
                        time_gaps[user] = np.mean(np.diff(user_days))

                avg_time_gap = np.nanmean(list(time_gaps.values()))

                # Calculate back-to-back and within-gap sampling
                back_to_back_sampling = np.sum(np.array(list(time_gaps.values())) == 1) / num_nodes * 100
                within_gap_sampling = np.sum(np.array(list(time_gaps.values())) <= gap_threshold) / num_nodes * 100
                over_exertion_score = within_gap_sampling

                # Extract the bootstrap iteration number
                try:
                    bootstrap_iter = filename.split('_')[1]
                except IndexError:
                    print(f"Error extracting bootstrap iteration from filename {filename}.")
                    bootstrap_iter = "Unknown"

                all_metrics.append({
                    'strategy': strategy,
                    'bootstrap_iter': bootstrap_iter,
                    'entropy': sampling_entropy,
                    'max_entropy': max_entropy,
                    'coverage_ratio': coverage_ratio,
                    'avg_time_gap': avg_time_gap,
                    'back_to_back_sampling': back_to_back_sampling,
                    'within_gap_sampling': within_gap_sampling,
                    'over_exertion_score': over_exertion_score
                })

    metrics_df = pd.DataFrame(all_metrics)
    if metrics_df.empty:
        print("No metrics were calculated. Please check your folder structure and log files.")
        return None

    return metrics_df



def compute_cpi_full(df, label_name, metric='accuracy'):
    """
    Computes the CPI (normalized AUC) for each strategy and node type,
    and returns a summary DataFrame with mean and standard deviation of CPI,
    as well as a DataFrame with individual CPI values for each bootstrap.

    Parameters:
    - df: DataFrame containing performance metrics over days for each strategy and bootstrap.
    - label_name: Label name for identification in output DataFrame.
    - metric: The metric for which to compute CPI (e.g., 'accuracy', 'f1_macro').

    Returns:
    - summary_df: DataFrame with columns ['label', 'strategy', 'node_type', 'mean_CPI', 'std_CPI'].
    - cpi_values_df: DataFrame with individual CPI values for each bootstrap iteration.
    """
    # List to store CPI summary and individual CPI values
    cpi_results = []
    cpi_values_list = []

    # Loop over all unique node types in the DataFrame
    for node_type in df['node_type'].unique():
        node_df = df[df['node_type'] == node_type]

        for strategy in node_df['strategy'].unique():
            strategy_df = node_df[node_df['strategy'] == strategy]
            cpi_values = []

            # Calculate CPI for each bootstrap
            for bootstrap in strategy_df['bootstrap_iteration'].unique():
                bootstrap_df = strategy_df[strategy_df['bootstrap_iteration'] == bootstrap]

                # Ensure days are sorted for CPI calculation
                bootstrap_df = bootstrap_df.sort_values(by='day')
                days = bootstrap_df['day'].values
                metric_values = bootstrap_df[metric].values

                # Calculate AUC for this bootstrap
                if len(days) > 1:  # Ensure there are multiple days for AUC calculation
                    auc_value = auc(days, metric_values)

                    # Normalize AUC by dividing by the maximum possible AUC (number of days) to get CPI
                    max_auc = days[-1] - days[0] + 1  # Assuming day sequence is continuous
                    cpi = auc_value / max_auc
                    cpi_values.append(cpi)

                    # Store individual CPI values for analysis
                    cpi_values_list.append({
                        'label': label_name,
                        'strategy': strategy,
                        'node_type': node_type,
                        'bootstrap_iteration': bootstrap,
                        'CPI': cpi
                    })

            # Calculate mean and standard deviation of CPI for this strategy and node type
            mean_cpi = np.mean(cpi_values) if cpi_values else np.nan
            std_cpi = np.std(cpi_values) if cpi_values else np.nan

            # Append summary results to the list
            cpi_results.append({
                'label': label_name,
                'strategy': strategy,
                'node_type': node_type,
                'mean_CPI': mean_cpi,
                'std_CPI': std_cpi
            })

    # Convert results to DataFrames
    summary_df = pd.DataFrame(cpi_results)
    cpi_values_df = pd.DataFrame(cpi_values_list)

    return summary_df, cpi_values_df




def load_perf_results(base_folder, label_type):
    """
    Function to load and consolidate results from a folder structure.

    Args:
    - base_folder (str): The base folder containing the results.
    - label_type (str): Specific label type to filter.

    Returns:
    - results_df (pd.DataFrame): Consolidated DataFrame of all results.
    """
    results_list = []

    # Loop through the folder structure to read results
    label_path = os.path.join(base_folder, label_type)
    if not os.path.isdir(label_path):
        raise ValueError(f"The specified label_type '{label_type}' does not exist or is not a directory.")

    for graph_type in os.listdir(label_path):
        graph_path = os.path.join(label_path, graph_type)
        if not os.path.isdir(graph_path):
            continue

        for experiment in os.listdir(graph_path):
            experiment_path = os.path.join(graph_path, experiment)
            if not os.path.isdir(experiment_path):
                continue

            # Path to the metrics CSV
            metrics_folder = os.path.join(experiment_path, 'metrics')
            result_file = os.path.join(metrics_folder, 'consolidated_metrics_all_strategies_with_baselines.csv')

            if os.path.exists(result_file) and result_file.endswith('.csv'):
                # Extract L and k from the folder name
                try:
                    L_value = int(experiment.split('_')[1])
                    k_value = int(experiment.split('_')[3])
                except (IndexError, ValueError):
                    continue

                # Read the CSV file
                df = pd.read_csv(result_file)

                # Add metadata columns to each result row
                df['label_type'] = label_type
                df['graph_type'] = graph_type
                df['L'] = L_value
                df['k'] = k_value

                # Ensure the node_type column is included if it's required
                if 'node_type' not in df.columns:
                    df['node_type'] = 'Unknown'  # Assign a default value if not present

                results_list.append(df)

    # Concatenate all the results into a single DataFrame
    if results_list:
        results_df = pd.concat(results_list, ignore_index=True)
        return results_df
    else:
        raise FileNotFoundError("No results found. Please check the folder structure and paths.")

def analyze_exertion_for_multiple_gaps(base_folder, L, k, num_nodes, gap_thresholds=[1, 2, 3, 4, 5]):
    folder_path = os.path.join(
        base_folder,
        f"L_{L}_k_{k}_nodes_{num_nodes}",
        "user_log"
    )

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None

    # List all strategies within the user_log directory
    strategies = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    all_metrics = []

    # Iterate over each gap threshold
    for gap_threshold in gap_thresholds:
        for strategy in strategies:
            strategy_folder = os.path.join(folder_path, strategy)

            for filename in os.listdir(strategy_folder):
                if filename.startswith("bootstrap_") and filename.endswith("_user_log.csv"):
                    file_path = os.path.join(strategy_folder, filename)
                    user_log_df = pd.read_csv(file_path)

                    # Flatten sampled users
                    all_sampled_users = []
                    for sampled_users in user_log_df['sampled_users']:
                        all_sampled_users.extend(eval(sampled_users))

                    # Frequency series with zero frequency included
                    user_freq = pd.Series(all_sampled_users).value_counts(normalize=True).reindex(range(num_nodes), fill_value=0)

                    # Calculate within-gap sampling for the given threshold
                    time_gaps = defaultdict(list)
                    for user in range(num_nodes):
                        user_days = user_log_df[user_log_df['sampled_users'].apply(lambda x: user in eval(x))]['day'].tolist()
                        if len(user_days) > 1:
                            time_gaps[user] = np.mean(np.diff(user_days))

                    within_gap_sampling = np.sum(np.array(list(time_gaps.values())) <= gap_threshold) / num_nodes * 100
                    over_exertion_score = within_gap_sampling

                    # Store the metrics with current gap threshold and strategy
                    all_metrics.append({
                        'strategy': strategy,
                        'gap_threshold': gap_threshold,
                        'over_exertion_score': over_exertion_score
                    })

    # Convert metrics to DataFrame
    exertion_df = pd.DataFrame(all_metrics)
    return exertion_df