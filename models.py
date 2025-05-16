
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, BatchNorm
import numpy as np

def norm_A(A, norm_type="max"):
    """
    Normalize adjacency matrix A with specified norm_type.
    
    Parameters:
    - A (np.ndarray or torch.Tensor): The adjacency matrix.
    - norm_type (str): The type of normalization to apply. Options are "max" or "symmetric".
    
    Returns:
    - np.ndarray or torch.Tensor: The normalized adjacency matrix, matching the input type.
    """
    assert norm_type in ["max", "symmetric"], "norm_type must be 'max' or 'symmetric'."
    
    # Convert A to a PyTorch tensor if it's a NumPy array
    is_numpy = isinstance(A, np.ndarray)
    if is_numpy:
        A = torch.tensor(A, dtype=torch.float32)

    if norm_type == "max":
        # Max normalization: Divide by the maximum value in A
        max_val = A.max()
        A_norm = A / max_val if max_val != 0 else A
    
    elif norm_type == "symmetric":
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        degree = A.sum(dim=1)  # Degree matrix
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree == 0] = 0  # Handle isolated nodes with degree 0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt  # Symmetric normalization
    
    # Convert A_norm back to NumPy array if input was NumPy
    return A_norm.numpy() if is_numpy else A_norm



class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1,
                 num_layers=2, dropout=0.5, activation="relu", batchnorm=True):
        super(GATModel, self).__init__()
        
        assert activation in ["relu", "elu"], "Unsupported activation. Choose 'relu' or 'elu'."
        
        # Activation function
        self.activation_fn = F.relu if activation == "relu" else F.elu
        self.dropout = dropout

        # Initial input layer
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=True))
        
        # Add optional batch normalization for the input layer
        self.batchnorms = torch.nn.ModuleList()
        if batchnorm:
            self.batchnorms.append(BatchNorm1d(hidden_channels * num_heads))

        # Hidden layers (flexible number based on num_layers)
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True))
            if batchnorm:
                self.batchnorms.append(BatchNorm1d(hidden_channels * num_heads))

        # Output layer
        self.layers.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index):
        # Forward pass through each GAT layer except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)            # GAT layer
            if len(self.batchnorms) > i:         # Batch normalization if enabled
                x = self.batchnorms[i](x)
            x = self.activation_fn(x)            # Activation function
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout layer

        # Last layer (output layer) without activation or batchnorm
        x = self.layers[-1](x, edge_index)
        return torch.sigmoid(x) if self.layers[-1].out_channels == 1 else x

    def embed(self, x, edge_index):
        """
        Embedding function to get intermediate node representations.
        This function passes the input through all layers except the final output layer.
        
        Args:
            x: Node feature matrix.
            edge_index: Edge index for the graph.
        Returns:
            Tensor of node embeddings after the last hidden layer.
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            if len(self.batchnorms) > i:
                x = self.batchnorms[i](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # Return the intermediate representation



class GAT_custom_hidden(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1,
                 num_layers=None, dropout=0.5, activation="relu", batchnorm=True):
        """
        GATModel with customizable hidden layers.

        Args:
            in_channels (int): Size of input features.
            hidden_channels (list or int): List of hidden layer sizes (e.g., [48, 32, 16]) or scalar.
            out_channels (int): Size of the output features.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of layers. If None, inferred from hidden_channels.
            dropout (float): Dropout rate.
            activation (str): Activation function ('relu' or 'elu').
            batchnorm (bool): Whether to use BatchNorm.
        """
        super(GAT_custom_hidden, self).__init__()

        assert activation in ["relu", "elu"], "Unsupported activation. Choose 'relu' or 'elu'."
        assert isinstance(hidden_channels, list), "hidden_channels must be a list for GAT_custom_hidden."

        # Infer the number of layers from hidden_channels if not provided
        if num_layers is None:
            num_layers = len(hidden_channels) + 1

        # Activation function
        self.activation_fn = F.relu if activation == "relu" else F.elu
        self.dropout = dropout

        # Layers and BatchNorm lists
        self.layers = torch.nn.ModuleList()
        self.batchnorms = torch.nn.ModuleList() if batchnorm else None

        # Input layer
        self.layers.append(GATConv(in_channels, hidden_channels[0], heads=num_heads, concat=True))
        if batchnorm:
            self.batchnorms.append(BatchNorm1d(hidden_channels[0] * num_heads))

        # Hidden layers
        for i in range(1, num_layers - 1):
            in_dim = hidden_channels[i - 1] * num_heads
            out_dim = hidden_channels[i]
            self.layers.append(GATConv(in_dim, out_dim, heads=num_heads, concat=True))
            if batchnorm:
                self.batchnorms.append(BatchNorm1d(out_dim * num_heads))

        # Output layer
        self.layers.append(
            GATConv(hidden_channels[-1] * num_heads, out_channels, heads=1, concat=False)
        )

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
           
            x = layer(x, edge_index)  # GAT layer
            if self.batchnorms and len(self.batchnorms) > i:
                x = self.batchnorms[i](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
       
        x = self.layers[-1](x, edge_index)
        return torch.sigmoid(x) if self.layers[-1].out_channels == 1 else x


    def embed(self, x, edge_index):
        """
        Embedding function to get intermediate node representations.
        Args:
            x: Node feature matrix.
            edge_index: Edge index for the graph.
        Returns:
            torch.Tensor: Intermediate node embeddings.
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            if self.batchnorms and len(self.batchnorms) > i:
                x = self.batchnorms[i](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

