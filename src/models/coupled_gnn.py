"""
Coupled Heterogeneous Graph Neural Network for 1D-2D Urban Flood Networks.
Implements message passing across node types with edge-type-specific transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple


class HeteroGraphConv(nn.Module):
    """
    Single heterogeneous graph convolution layer.
    Applies edge-type-specific message passing and aggregates across types.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        edge_types: List[Tuple[str, str, str]],
        aggr: str = 'mean',
        use_attention: bool = False,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels_dict: Dict mapping node type to input dimension
            out_channels: Output dimension for all node types
            edge_types: List of (src_type, edge_type, dst_type) tuples
            aggr: Aggregation method ('mean', 'sum', 'max')
            use_attention: Whether to use GAT-style attention
            heads: Number of attention heads (if use_attention)
            dropout: Dropout rate
        """
        super().__init__()
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Build convolutions for each edge type
        conv_dict = {}
        for src_type, edge_name, dst_type in edge_types:
            in_channels = (in_channels_dict[src_type], in_channels_dict[dst_type])
            if use_attention:
                # GAT: out_channels must be divisible by heads
                conv = GATConv(
                    in_channels,
                    out_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
            else:
                conv = SAGEConv(in_channels, out_channels, aggr=aggr)
            conv_dict[(src_type, edge_name, dst_type)] = conv

        self.conv = HeteroConv(conv_dict, aggr='sum')  # Aggregate across edge types

        # Layer norm for each node type
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(out_channels)
            for node_type in in_channels_dict.keys()
        })

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_dict: Dict mapping node type to features [num_nodes, in_channels]
            edge_index_dict: Dict mapping edge type to edge index [2, num_edges]

        Returns:
            Dict mapping node type to updated features [num_nodes, out_channels]
        """
        # Apply heterogeneous convolution
        out_dict = self.conv(x_dict, edge_index_dict)

        # Apply normalization and activation
        for node_type, x in out_dict.items():
            x = self.norms[node_type](x)
            x = F.gelu(x)
            x = self.dropout(x)
            out_dict[node_type] = x

        return out_dict


class CoupledHeteroGNN(nn.Module):
    """
    Multi-layer Coupled Heterogeneous GNN for 1D-2D flood networks.
    Processes static graph structure with dynamic node features.
    """

    def __init__(
        self,
        in_channels_1d: int,
        in_channels_2d: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        use_attention: bool = True,
        heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        """
        Args:
            in_channels_1d: Input feature dimension for 1D nodes
            in_channels_2d: Input feature dimension for 2D nodes
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of message passing layers
            use_attention: Whether to use attention mechanism
            heads: Number of attention heads
            dropout: Dropout rate
            residual: Whether to use residual connections
        """
        super().__init__()
        self.num_layers = num_layers
        self.residual = residual

        # Define edge types
        self.edge_types = [
            ('1d', 'pipe', '1d'),
            ('2d', 'surface', '2d'),
            ('1d', 'couples_to', '2d'),
            ('2d', 'couples_from', '1d'),
        ]

        # Input projections
        self.input_proj = nn.ModuleDict({
            '1d': nn.Linear(in_channels_1d, hidden_channels),
            '2d': nn.Linear(in_channels_2d, hidden_channels),
        })

        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels

            self.convs.append(
                HeteroGraphConv(
                    in_channels_dict={'1d': in_ch, '2d': in_ch},
                    out_channels=out_ch,
                    edge_types=self.edge_types,
                    use_attention=use_attention,
                    heads=heads,
                    dropout=dropout,
                )
            )

        # Residual projection if dimensions don't match
        if residual and hidden_channels != out_channels:
            self.res_proj = nn.ModuleDict({
                '1d': nn.Linear(hidden_channels, out_channels),
                '2d': nn.Linear(hidden_channels, out_channels),
            })
        else:
            self.res_proj = None

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GNN.

        Args:
            x_dict: Dict with '1d' and '2d' node features
            edge_index_dict: Dict with edge indices for all edge types

        Returns:
            Dict with updated '1d' and '2d' node embeddings
        """
        # Input projection
        h_dict = {
            node_type: self.input_proj[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Message passing layers
        for i, conv in enumerate(self.convs):
            h_prev = h_dict

            # Apply convolution
            h_dict = conv(h_dict, edge_index_dict)

            # Residual connection (except last layer or if explicitly disabled)
            if self.residual and i < self.num_layers - 1:
                for node_type in h_dict:
                    h_dict[node_type] = h_dict[node_type] + h_prev[node_type]

        # Final residual if needed
        if self.residual and self.res_proj is not None:
            for node_type in h_dict:
                h_dict[node_type] = h_dict[node_type] + self.res_proj[node_type](h_prev[node_type])

        return h_dict

    def forward_from_data(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Convenience method to forward pass from HeteroData.

        Args:
            data: PyG HeteroData object

        Returns:
            Dict with updated node embeddings
        """
        x_dict = {
            '1d': data['1d'].x,
            '2d': data['2d'].x,
        }
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in self.edge_types
            if edge_type in data.edge_types
        }
        return self(x_dict, edge_index_dict)


class SpatialEncoder(nn.Module):
    """
    Encodes spatial/static features using the coupled GNN.
    Used to create position-aware embeddings that can be combined with temporal features.
    """

    def __init__(
        self,
        static_1d_dim: int,
        static_2d_dim: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        **kwargs
    ):
        super().__init__()
        self.gnn = CoupledHeteroGNN(
            in_channels_1d=static_1d_dim,
            in_channels_2d=static_2d_dim,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            **kwargs
        )

    def forward(
        self,
        data: HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode static graph features.

        Args:
            data: HeteroData with static node features

        Returns:
            spatial_emb_1d: [num_1d_nodes, hidden_channels]
            spatial_emb_2d: [num_2d_nodes, hidden_channels]
        """
        out = self.gnn.forward_from_data(data)
        return out['1d'], out['2d']
