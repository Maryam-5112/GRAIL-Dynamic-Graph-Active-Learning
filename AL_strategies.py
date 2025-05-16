

import torch
import numpy as np
import random
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from torch_geometric.nn import GCNConv, SAGEConv
from networkx.algorithms.community import greedy_modularity_communities



def get_embedded_space_rep(model, x, adj_t, rep_type='model_embedding', encoder='gcn', device='cpu'):
    """
    Get node representations in graph embedded space.
    
    Args:
        model: The GNN model with an `embed` method.
        x: Node feature matrix.
        adj_t: Adjacency matrix in sparse format.
        rep_type: Type of representation to retrieve:
                  'graph_rep' - Aggregation using adjacency normalization.
                  'model_embedding' - Full embedding using model's `embed` function.
        encoder: Type of encoder for aggregation ('gcn' or 'sage').
        device: Device to run the computation on (e.g., 'cpu' or 'cuda').
    
    Returns:
        Tensor of node embeddings based on the selected rep_type.
    """
    model.to(device)
    x = x.to(device)
    adj_t = adj_t.to(device)
    model.eval()  # Set model to evaluation mode

    if rep_type == 'graph_rep':
        feat_dim = x.size(1)

        # Initialize a layer based on the encoder type
        if encoder == 'sage':
            conv = SAGEConv(feat_dim, feat_dim, bias=False)
            conv.lin_l.weight = torch.nn.Parameter(torch.eye(feat_dim))
            conv.lin_r.weight = torch.nn.Parameter(torch.eye(feat_dim))
        elif encoder == 'gcn':
            conv = GCNConv(feat_dim, feat_dim, cached=True, bias=False)
            conv.lin.weight = torch.nn.Parameter(torch.eye(feat_dim))
        else:
            raise ValueError(f"Unsupported encoder type: {encoder}")

        conv.to(device)
        
        # Apply aggregation (double pass to simulate multi-hop aggregation)
        with torch.no_grad():
            aggregated_rep = conv(x, adj_t)
            aggregated_rep = conv(aggregated_rep, adj_t)  # Second pass for multi-hop

        return aggregated_rep  # Return the graph representation

    elif rep_type == 'model_embedding':
        # Use the model's embed method for final embedding representation
        with torch.no_grad():
            embedding = model.embed(x, adj_t)
        return embedding  # Return the final model embedding

    else:
        raise ValueError(f"Unsupported representation type: {rep_type}")



def AL_query(X, A, num_classes, strategy, uncertainity_metric=None, num_queries=10, model=None, train_mask=None, num_clusters=None,device='cpu',gamma=0.9):
    """
    Samples users based on the active learning strategy and the provided train_mask.

    Parameters:
    - X: Feature matrix for the current day.
    - A: Adjacency matrix, used for strategies requiring graph structure.
    - strategy: Active learning strategy to use.
    - num_queries: Number of users to sample.
    - model: GNN model, required for certain AL strategies.
    - y_pred: Model predictions, used to calculate uncertainty in certain AL strategies.
    - train_mask: Boolean mask indicating which nodes are available for querying.

    Returns:
    - Indices of selected users.
    """
    # Create a detached copy of A and convert to numpy for NetworkX
    A_copy = A.clone().cpu().numpy()
    G = nx.from_numpy_array(A_copy)

    # Filter available indices based on train_mask to ensure only eligible nodes are considered
    available_indices = torch.nonzero(train_mask).squeeze()
    

    if strategy == "random_sample":
        indices = random_strategy(num_queries, available_indices)
    elif strategy == "degree":
        indices = degree_strategy(G, num_queries, available_indices)
    elif strategy == "pagerank":
        indices = pagerank_strategy(G, num_queries, available_indices)
    # Generate predictions from the model if using an uncertainty-based strategy
    elif strategy == "uncertainty":
        if model is None:
            raise ValueError("A model is required for uncertainty-based strategies.")
         
        # Ensure model is in evaluation mode
        model.eval()  # Set model to evaluation mode for uncertainty sampling
        # Convert adjacency matrix to edge_index format for GNN models
        edge_index = torch.nonzero(A.clone().detach(), as_tuple=False).t().contiguous()
        # Generate predictions
        with torch.no_grad():  # Disable gradient tracking for prediction
            y_pred = model(X, edge_index).squeeze()
        # Run the uncertainty-based sampling
        indices = uncertainty_strategy(y_pred, num_queries, available_indices, uncertainity_metric, num_classes)

    elif strategy == 'density':
        # Use num_clusters if provided; otherwise, default to num_classes
        selected_num_clusters = num_clusters if num_clusters is not None else num_queries
        indices = density_strategy(A, X, model, num_queries, available_indices, selected_num_clusters, device)
    elif strategy == "age":
        # Use num_clusters if provided; otherwise, default to num_samples
        selected_num_clusters = num_clusters if num_clusters is not None else num_queries
        indices = age_strategy(X, A, model, num_queries,num_classes, available_indices, gamma=gamma, num_clusters=selected_num_clusters, device=device)
    elif strategy == "coreset":
        indices = Coreset_strategy(X, num_queries, available_indices, model, A, device)
    elif strategy == 'featProp':
        indices = feat_prop(X, A, model, num_queries, representation='graph_rep', encoder='gcn', initialization='k-means++', device=device)
    elif strategy =='graphpart':
        indices = partition_based_query(X, A, model, num_samples=num_queries, 
                                         representation='graph_rep', encoder='gcn', 
                                         initialization='k-means++', compensation=0.0, device=device)
    elif strategy =='graphpartfar':
        indices = partition_based_query(X, A, model, num_samples=num_queries, 
                                         representation='graph_rep', encoder='gcn', 
                                         initialization='k-means++', compensation=1.0, device=device)
    
    else:
        raise ValueError(f"Strategy '{strategy}' not recognized.")

    return indices


# Define individual strategy functions

def random_strategy(num_queries, available_indices):
    """Randomly sample from available indices."""
    return available_indices[torch.randperm(len(available_indices))[:num_queries]]


def degree_strategy(G, num_queries, available_indices):
    """Select users based on node degree centrality."""
    degrees = dict(G.degree)  
    degree_scores = torch.tensor([degrees[int(i)] for i in available_indices])
    sorted_indices = available_indices[degree_scores.argsort(descending=True)]
    return sorted_indices[:num_queries]


def pagerank_strategy(G, num_queries, available_indices):
    """Select users based on PageRank centrality."""
    pagerank_scores = nx.pagerank(G)
    pagerank_tensor = torch.tensor([pagerank_scores[int(i)] for i in available_indices])
    sorted_indices = available_indices[pagerank_tensor.argsort(descending=True)]
    return sorted_indices[:num_queries]



def uncertainty_strategy(y_pred, num_queries, available_indices, strategy="entropy", num_classes=2):
    """
    Select users with the highest uncertainty based on the specified uncertainty strategy.
    
    Parameters:
    - y_pred: Tensor of model predictions (probabilities or logits, depending on num_classes).
    - num_queries: Number of users to select.
    - available_indices: Indices of users eligible for selection.
    - strategy: Uncertainty strategy to use ("least_confidence", "entropy", "margin").
    - num_classes: Number of classes (2 for binary classification, >2 for multi-class).
    
    Returns:
    - Top num_queries indices of users with highest uncertainty.
    """
    if y_pred is None:
        raise ValueError("y_pred is required for uncertainty-based strategy.")
    
    if num_classes == 2:
        # For binary classification, convert sigmoid output to probabilities
        probs = torch.cat([1 - y_pred.unsqueeze(1), y_pred.unsqueeze(1)], dim=1)
    else:
        # For multi-class classification, apply softmax to get probabilities across classes
        probs = y_pred.softmax(dim=1)
    
    if strategy == "least_confidence":
        # Least confidence: 1 - max probability
        uncertainty_scores = 1 - probs.max(dim=1)[0]
        
    elif strategy == "entropy":
        # Entropy-based uncertainty
        uncertainty_scores = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
    elif strategy == "margin":
        # Margin-based uncertainty: difference between the two highest probabilities
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        uncertainty_scores = sorted_probs[:, 0] - sorted_probs[:, 1]
    
    else:
        raise ValueError(f"Unknown uncertainty strategy '{strategy}'. Choose from 'least_confidence', 'entropy', or 'margin'.")
    
    # Select top num_queries uncertain indices from available indices
    sorted_indices = available_indices[uncertainty_scores[available_indices].argsort(descending=True)]
    return sorted_indices[:num_queries]



def density_strategy(A, X, model, num_samples, available_indices, num_clusters, device='cpu'):
    """
    Density-based Active Learning Strategy.
    Clusters node embeddings and samples nodes with the highest density score.

    Parameters:
    - A: Adjacency matrix in dense format.
    - X: Feature matrix for the current day.
    - model: GNN model used to embed nodes.
    - num_samples: Number of users to sample based on density.
    - available_indices: Boolean mask indicating which nodes are available for querying.
    - num_clusters: Number of clusters to use in K-Means.
    - device: Device to run the computation on (e.g., 'cpu' or 'cuda').

    Returns:
    - Indices of selected users based on density score.
    """
    # Convert adjacency matrix to edge_index format for GNN models
    edge_index = torch.nonzero(A.clone().detach(), as_tuple=False).t().contiguous()

    # Get node embeddings using the edge_index format of A
    x_embed = get_embedded_space_rep(model, X, edge_index, rep_type='model_embedding', device=device)
    x_embed = x_embed.cpu()


    # Perform K-Means clustering on embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_embed)
    centers = torch.tensor(kmeans.cluster_centers_)

    # Calculate the distance of each point to its cluster center
    labels = torch.tensor(kmeans.labels_)
    cluster_centers = centers[labels]  # Assign each node to its cluster center
    dist_map = torch.linalg.norm(x_embed - cluster_centers, dim=1)
    
    # Calculate density (inverse of distance to the cluster center)
    density = 1 / (1 + dist_map)

    # Mask out non-available nodes
    density[~available_indices] = 0

    # Select top `num_samples` nodes with the highest density scores
    _, indices = torch.topk(density, k=num_samples)
    
    return indices




def age_strategy(X, A, model, num_samples, num_classes, available_indices, gamma=0.4, num_clusters=10, device='cpu'):
    """
    AGE-based Active Learning Strategy.
    Combines entropy, density, and centrality (PageRank) metrics for node selection.

    Parameters:
    - X: Feature matrix for the current day.
    - A: Adjacency matrix in dense format.
    - model: GNN model used for predictions.
    - num_samples: Number of users to sample based on the AGE score.
    - num_classes: Number of classes for classification (used in entropy calculation).
    - available_indices: Boolean mask indicating which nodes are available for querying.
    - gamma: Weight for PageRank centrality in the AGE score.
    - num_clusters: Number of clusters to use in K-Means for density calculation.
    - device: Device to run the computation on (e.g., 'cpu' or 'cuda').

    Returns:
    - Indices of selected users based on AGE score.
    """
    # Calculate alpha and beta based on gamma
    alpha = beta = (1 - gamma) / 2

    # Ensure model is in evaluation mode and move to device
    model.to(device)
    model.eval()

    # Convert adjacency matrix to edge_index format for GNN models
    edge_index = torch.nonzero(A.clone().detach(), as_tuple=False).t().contiguous()

    # Generate predictions and calculate entropy
    with torch.no_grad():
        y_pred = model(X, edge_index).squeeze()
    
    if num_classes == 2:
        # For binary classification, convert sigmoid output to probabilities
        probs = torch.cat([1 - y_pred.unsqueeze(1), y_pred.unsqueeze(1)], dim=1)
    else:
        # For multi-class classification, apply softmax to get probabilities across classes
        probs = y_pred.softmax(dim=1)

    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

    # Calculate PageRank centrality
    G = nx.from_numpy_array(A.cpu().numpy())
    page_rank = torch.tensor(list(nx.pagerank(G).values()), dtype=entropy.dtype, device=device)

    # Calculate density via K-Means clustering with specified number of clusters
    x_embed = get_embedded_space_rep(model, X, edge_index, rep_type='model_embedding', device=device).cpu()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x_embed)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=entropy.dtype, device=device)
    labels = torch.tensor(kmeans.labels_, dtype=torch.long, device=device)
    cluster_centers = centers[labels]
    dist_map = torch.linalg.norm(x_embed.to(device) - cluster_centers, dim=1)
    density = (1 / (1 + dist_map)).to(entropy.dtype)

    # Normalize metrics to percentile ranks
    N = len(X)
    percentile = torch.arange(N, dtype=entropy.dtype, device=device) / N
    id_sorted = density.argsort(descending=False)
    density[id_sorted] = percentile
    id_sorted = entropy.argsort(descending=False)
    entropy[id_sorted] = percentile
    id_sorted = page_rank.argsort(descending=False)
    page_rank[id_sorted] = percentile

    # Calculate AGE score as a weighted combination of metrics
    age_score = alpha * entropy + beta * density + gamma * page_rank

    # Mask out nodes that are not available for querying
    age_score[~available_indices] = 0

    # Select top `num_samples` nodes with the highest AGE scores
    _, indices = torch.topk(age_score, k=num_samples)

    return indices



def Coreset_strategy(X, num_queries, available_indices, model=None, A=None, device='cpu'):
    """
    Implements the CoreSet strategy using K-Center Greedy selection in the embedded space.
    Selects num_queries points that maximize coverage of the feature space.

    Parameters:
    - X: Feature matrix for the current day.
    - num_queries: Number of users to sample.
    - available_indices: Indices of nodes available for selection.
    - model: GNN model to get node embeddings.
    - A: Adjacency matrix in dense format.
    - device: Device to run computations on (e.g., 'cpu' or 'cuda').

    Returns:
    - Indices of selected users.
    """
    
    # Get the node embeddings using the GNN model in the embedded space
    edge_index = torch.nonzero(A.clone().detach(), as_tuple=False).t().contiguous()
    embeddings = get_embedded_space_rep(model, X, edge_index, rep_type='model_embedding', device=device)
    embeddings = embeddings.cpu().numpy()  # Convert embeddings to numpy for pairwise distance calculation

    # Mask embeddings to consider only available indices
    available_embeddings = embeddings[available_indices]
    selected_indices = []

    # Initialize distances for k-center greedy selection
    min_distances = np.full(len(available_embeddings), np.inf)

    for _ in range(num_queries):
        if len(selected_indices) == 0:
            # Randomly select the first center if no points are selected yet
            new_index = np.random.choice(range(len(available_embeddings)))
        else:
            # Compute distances from available points to the closest selected center
            dist_to_selected = pairwise_distances(available_embeddings, available_embeddings[selected_indices], metric='euclidean')
            min_distances = np.minimum(min_distances, dist_to_selected.min(axis=1))
            new_index = np.argmax(min_distances)

        # Add the selected point to the batch of selected indices
        selected_indices.append(new_index)

    # Map selected indices back to the available indices in the original dataset
    final_indices = available_indices[selected_indices]

    return final_indices


def feat_prop(X, A, model, num_samples, representation='graph_rep', encoder='gcn', initialization='k-means++', device='cpu'):
    """
    Feature Propagation-based Active Learning Query Function.
    Performs clustering on aggregated node features and selects nodes closest to cluster centers.

    Parameters:
    - X: Feature matrix for the current day.
    - A: Adjacency matrix in dense format.
    - model: GNN model used for node representation.
    - num_samples: Number of nodes to sample based on cluster centers.
    - representation: Type of representation ('feature', 'model_embedding', or 'graph_rep').
    - encoder: Encoder type for feature aggregation (e.g., 'gcn', 'sage').
    - initialization: Initialization method for K-Means ('k-means++' or 'random').
    - device: Device to run the computation on (e.g., 'cpu' or 'cuda').

    Returns:
    - Indices of selected nodes based on clustering.
    """
    # Ensure model is in evaluation mode and move to device
    model.to(device)
    model.eval()

    # Convert adjacency matrix to edge_index format for GNN models
    edge_index = torch.nonzero(A.clone().detach(), as_tuple=False).t().contiguous()

    # Get node representations
    x_embed = get_embedded_space_rep(model, X, edge_index, rep_type=representation, encoder=encoder, device=device).cpu()

    # Perform K-Means clustering on representations
    kmeans = KMeans(n_clusters=num_samples, init=initialization, random_state=0).fit(x_embed.numpy())
    centers = torch.tensor(kmeans.cluster_centers_, dtype=x_embed.dtype, device=x_embed.device)

    # Obtain the nodes closest to each cluster center
    selected_indices = []
    for center in centers:
        center = center.to(dtype=x_embed.dtype, device=x_embed.device)
        dist_map = torch.linalg.norm(x_embed - center, dim=1)
        
        # Set already selected nodes to infinity to avoid reselection
        for idx in selected_indices:
            dist_map[idx] = float('inf')
        
        # Find the closest node to the center
        idx = int(torch.argmin(dist_map))
        selected_indices.append(idx)

    return torch.tensor(selected_indices)



def partition_based_query(X, A, model, num_samples, representation='graph_rep', encoder='gcn', 
                          initialization='k-means++', compensation=1.0, device='cpu'):
    """
    Partition-based Active Learning Query Function.
    Partitions the graph into communities, clusters each community, and selects nodes closest to cluster centers.

    Parameters:
    - X: Feature matrix for the current day.
    - A: Adjacency matrix in dense format.
    - model: GNN model used for node representation.
    - num_samples: Number of nodes to sample.
    - representation: Type of node representation ('none', 'model_embedding', or 'graph_rep').
    - encoder: Encoder type for feature aggregation (e.g., 'gcn', 'sage').
    - initialization: Initialization method for K-Means ('k-means++' or 'random').
    - compensation: Float [0-1] to control interference compensation between partitions.
    - device: Device to run the computation on (e.g., 'cpu' or 'cuda').

    Returns:
    - Indices of selected nodes based on partitioned clustering.
    """
    # Ensure model is in evaluation mode and move to device
    model.to(device)
    model.eval()

    # Convert adjacency matrix to NetworkX graph for partitioning
    G = nx.from_numpy_array(A.cpu().numpy())

    # Perform graph partition using Clauset-Newman-Moore
    communities = list(greedy_modularity_communities(G))
    num_parts = len(communities)
    

    # Get node representations
    edge_index = torch.nonzero(A.clone().detach(), as_tuple=False).t().contiguous()
    x_embed = get_embedded_space_rep(model, X, edge_index, rep_type=representation, encoder=encoder, device=device).cpu()

    # Determine the number of samples per partition
    part_size = [len(comm) * num_samples // len(G.nodes) for comm in communities]
   
    selected_indices = []

    # Cluster and select nodes within each partition
    for i, comm in enumerate(communities):
        comm_nodes = list(comm)
        x_partition = x_embed[comm_nodes]

        # Set number of clusters within this partition
        n_clusters = part_size[i]
        if n_clusters <= 0:
            continue

        # Perform K-Means clustering within the partition
        kmeans = KMeans(n_clusters=n_clusters, init=initialization, random_state=0).fit(x_partition)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=x_partition.dtype, device=x_partition.device)

        # Compensate for interference if enabled
        dist_to_center = None
        if compensation > 0:
            dist_to_center = torch.ones(x_embed.size(0), dtype=x_embed.dtype, device=x_embed.device) * float('inf')
            for idx in selected_indices:
                if idx < x_embed.size(0):  # Ensure idx is within bounds
                    dist_to_center = torch.minimum(dist_to_center, torch.linalg.norm(x_embed - x_embed[idx], dim=1))
            dist_partition = dist_to_center[comm_nodes]

        # Select the nodes closest to each center
        for center in centers:
            center = center.to(dtype=x_partition.dtype, device=x_partition.device)
            dist_map = torch.linalg.norm(x_partition - center, dim=1)

            if compensation > 0:
                dist_map -= dist_partition * compensation
            
            # Prevent reselection of already selected nodes within bounds
            valid_selected_indices = [idx for idx in selected_indices if idx < len(dist_map)]
            dist_map[torch.tensor(valid_selected_indices, dtype=torch.long)] = float('inf')
            
            # Find the closest node to the center and add to selected nodes
            idx = comm_nodes[int(torch.argmin(dist_map))]
            selected_indices.append(idx)

    return torch.tensor(selected_indices, dtype=torch.long)

