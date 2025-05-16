import os
import numpy as np
import pickle

def load_preprocessed_data(dataset, output_folder=None, file_path=None):
    """
    Load GNN data for either 'FnF' or 'SNAPSHOT' datasets based on user input.

    Args:
        dataset (str): The name of the dataset to load ('FnF' or 'SNAPSHOT').
        output_folder (str, optional): Path to the folder containing 'FnF' data files.
                                       Required for 'FnF' dataset.'Dataset/FnF'
        file_path (str, optional): Path to the saved 'SNAPSHOT' GNN data file.
                                   Required for 'SNAPSHOT' dataset. 'Dataset/SNAPSHOT/SNAPSHOT_Processed_all_Cohort.pkl

    Returns:
        dict: A dictionary containing the loaded data, structured as:
              - For 'FnF': A dictionary with labels as keys and sub-dictionaries containing:
                - 'X': Feature matrix (samples x features)
                - 'y': Label array (samples)
                - 'L_*': Adjacency matrices for different relationship types including friends, call_duration, number of calls, whether they are couples
              - For 'SNAPSHOT': A dictionary with cohort IDs as keys and sub-dictionaries containing:
                - 'X': Feature tensor (days x users x features)
                - 'y': Label tensor (days x users x 1)
                - 'A': Adjacency matrix (users x users)

    """
    if dataset == 'FnF':
        if output_folder is None:
            raise ValueError("For 'FnF' dataset, 'output_folder' must be specified.")

        label_list = ['sleep_class']  # Add any other labels you want to load here
        data_dict = {}

        for label in label_list:
            label_key = label.replace(" ", "_").lower()

            try:
                # Load X and y
                X = np.load(os.path.join(output_folder, f'X_{label_key}.npy'))
                y = np.load(os.path.join(output_folder, f'y_{label_key}.npy'))

                # Convert sleep_class label to binary (0: <=6 hours, 1: >=7 hours)
                if label_key == 'sleep_class':
                    y[y == 2] = 1

                # Load adjacency matrices
                L_friend = np.load(os.path.join(output_folder, f'L_friend_{label_key}.npy'))
                L_duration = np.load(os.path.join(output_folder, f'L_duration_{label_key}.npy'))
                L_calls = np.load(os.path.join(output_folder, f'L_calls_{label_key}.npy'))
                L_non_zero = np.load(os.path.join(output_folder, f'L_non_zero_{label_key}.npy'))
                L_couple = np.load(os.path.join(output_folder, f'L_couple_{label_key}.npy'))

                # Store loaded data
                data_dict[label] = {
                    'X': X,
                    'y': y,
                    'L_friend': L_friend,
                    'L_duration': L_duration,
                    'L_calls': L_calls,
                    'L_non_zero': L_non_zero,
                    'L_couple': L_couple
                }

                print(f"Successfully loaded 'FnF' data for label: {label}")

            except FileNotFoundError as fnf_error:
                print(f"File not found: {fnf_error}. Please check file paths in '{output_folder}'.")

            except Exception as e:
                print(f"An error occurred while loading 'FnF' data: {e}")

        return data_dict

    elif dataset == 'SNAPSHOT':
        if file_path is None:
            raise ValueError("For 'SNAPSHOT' dataset, 'file_path' must be specified.")

        try:
            with open(file_path, 'rb') as file:
                gnn_data = pickle.load(file)

            # Prepare the output dictionary
            output_data = {}
            for cohort_id, data in gnn_data.items():
                output_data[cohort_id] = {
                    'X': data.get('features'),
                    'y': data.get('labels'),
                    'A': data.get('adj_matrix')
                }

            print(f"Successfully loaded 'SNAPSHOT' data from {file_path}")
            return output_data

        except FileNotFoundError:
            print(f"File not found at {file_path}. Please check the path and try again.")
            return None
        except Exception as e:
            print(f"An error occurred while loading 'SNAPSHOT' data: {e}")
            return None

    else:
        raise ValueError("Invalid dataset specified. Choose either 'FnF' or 'SNAPSHOT'.")

# Example Usage:
# For FnF dataset:
# fnf_data = load_gnn_data('FnF', output_folder='path/to/FnF/data')

# For SNAPSHOT dataset:
# snapshot_data = load_gnn_data('SNAPSHOT', file_path='path/to/SNAPSHOT/data.pkl')
