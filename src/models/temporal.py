"""
Temporal processing modules for sequential flood prediction.
Includes GRU-based and TCN-based temporal encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalBlock(nn.Module):
    """
    GRU-based temporal processor for node-level time series.
    Processes sequences of node features and outputs hidden states.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Args:
            input_dim: Dimension of input features per timestep
            hidden_dim: Hidden state dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate between layers
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output projection (concatenates directions if bidirectional)
        output_dim = hidden_dim * self.num_directions
        self.output_proj = nn.Linear(output_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process temporal sequence.

        Args:
            x: Input tensor [batch, seq_len, num_nodes, input_dim]
               or [batch * num_nodes, seq_len, input_dim]
            h0: Optional initial hidden state

        Returns:
            output: Sequence outputs [batch, seq_len, num_nodes, hidden_dim]
            hidden: Final hidden state [num_layers * num_directions, batch * num_nodes, hidden_dim]
        """
        # Handle both 4D and 3D inputs
        if x.dim() == 4:
            batch, seq_len, num_nodes, input_dim = x.shape
            # Reshape to [batch * num_nodes, seq_len, input_dim]
            x = x.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, input_dim)
            reshape_back = True
        else:
            batch_nodes, seq_len, input_dim = x.shape
            reshape_back = False
            batch = None
            num_nodes = None

        # Forward through GRU
        output, hidden = self.gru(x, h0)

        # Project output
        output = self.output_proj(output)

        # Reshape back if needed
        if reshape_back:
            output = output.reshape(batch, num_nodes, seq_len, self.hidden_dim)
            output = output.permute(0, 2, 1, 3)  # [batch, seq_len, num_nodes, hidden_dim]

        return output, hidden

    def get_initial_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get zero initial hidden state."""
        return torch.zeros(
            self.num_layers * self.num_directions,
            batch_size * num_nodes,
            self.hidden_dim,
            device=device,
        )


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with dilated causal convolutions.
    Alternative to GRU for capturing long-range temporal dependencies.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, seq_len]
        Returns:
            [batch, out_channels, seq_len]
        """
        residual = self.residual(x)

        # First conv
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal: remove future
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # Second conv
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out + residual


class TemporalConvNet(nn.Module):
    """
    Multi-layer TCN for temporal processing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, num_nodes, input_dim]
        Returns:
            [batch, seq_len, num_nodes, hidden_dim]
        """
        batch, seq_len, num_nodes, input_dim = x.shape

        # Reshape to [batch * num_nodes, input_dim, seq_len] for Conv1d
        x = x.permute(0, 2, 3, 1).reshape(batch * num_nodes, input_dim, seq_len)

        # Apply TCN
        out = self.network(x)

        # Reshape back
        out = out.reshape(batch, num_nodes, self.hidden_dim, seq_len)
        out = out.permute(0, 3, 1, 2)  # [batch, seq_len, num_nodes, hidden_dim]

        return out


class SpatioTemporalEncoder(nn.Module):
    """
    Combined spatial (GNN) and temporal (GRU/TCN) encoder.
    First processes spatial structure, then temporal dynamics.
    """

    def __init__(
        self,
        spatial_dim: int,
        dynamic_dim: int,
        hidden_dim: int,
        temporal_type: str = 'gru',
        num_temporal_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            spatial_dim: Dimension of spatial embeddings from GNN
            dynamic_dim: Dimension of dynamic input features
            hidden_dim: Hidden dimension
            temporal_type: 'gru' or 'tcn'
            num_temporal_layers: Number of temporal layers
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Combine spatial and dynamic features
        self.input_proj = nn.Linear(spatial_dim + dynamic_dim, hidden_dim)

        # Temporal encoder
        if temporal_type == 'gru':
            self.temporal = TemporalBlock(
                hidden_dim, hidden_dim, num_temporal_layers, dropout
            )
        else:
            self.temporal = TemporalConvNet(
                hidden_dim, hidden_dim, num_temporal_layers, dropout=dropout
            )

        self.temporal_type = temporal_type

    def forward(
        self,
        spatial_emb: torch.Tensor,
        dynamic_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode spatiotemporal features.

        Args:
            spatial_emb: Static spatial embeddings [num_nodes, spatial_dim]
            dynamic_seq: Dynamic features [batch, seq_len, num_nodes, dynamic_dim]
            h0: Initial hidden state (for GRU)

        Returns:
            output: Encoded sequence [batch, seq_len, num_nodes, hidden_dim]
            hidden: Final hidden state (for GRU) or None (for TCN)
        """
        batch, seq_len, num_nodes, dynamic_dim = dynamic_seq.shape

        # Expand spatial embeddings to match sequence
        # [num_nodes, spatial_dim] -> [batch, seq_len, num_nodes, spatial_dim]
        spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1)

        # Concatenate spatial and dynamic features
        combined = torch.cat([spatial_expanded, dynamic_seq], dim=-1)

        # Project to hidden dim
        combined = self.input_proj(combined)

        # Apply temporal encoding
        if self.temporal_type == 'gru':
            output, hidden = self.temporal(combined, h0)
        else:
            output = self.temporal(combined)
            hidden = None

        return output, hidden
