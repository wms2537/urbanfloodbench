"""
Temporal Fusion Transformer (TFT) components for flood prediction.

TFT is designed for multi-horizon forecasting with:
- Variable selection for static/observed/known-future inputs
- Gated Residual Networks (GRN) for nonlinear processing
- Multi-head attention for temporal patterns
- Direct multi-horizon output (reduces autoregressive error accumulation)

Reference: https://arxiv.org/abs/1912.09363
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit with learnable gating."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        gate = torch.sigmoid(out[..., :out.shape[-1]//2])
        return out[..., out.shape[-1]//2:] * gate


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - core building block of TFT.
    Provides nonlinear processing with gating and skip connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Main processing layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_proj = None

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Gating
        self.gate = GatedLinearUnit(hidden_dim, output_dim)

        # Layer norm and skip connection
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if dimensions differ
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., input_dim]
            context: Optional context tensor [..., context_dim]
        Returns:
            Output tensor [..., output_dim]
        """
        # ELU activation
        hidden = F.elu(self.fc1(x))

        # Add context if provided
        if context is not None and self.context_proj is not None:
            hidden = hidden + self.context_proj(context)

        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)

        # Gating
        gated = self.gate(hidden)

        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x

        return self.layer_norm(gated + skip)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network - learns to weight different input variables.
    Key for interpretability and handling mixed input types.
    """

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of each input variable
            num_inputs: Number of input variables
            hidden_dim: Hidden dimension
            context_dim: Optional context dimension for conditioning
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim

        # GRN for each input variable
        self.grn_per_var = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_inputs)
        ])

        # Softmax weights GRN
        self.grn_weights = GatedResidualNetwork(
            num_inputs * hidden_dim,
            hidden_dim,
            num_inputs,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Input tensor [..., num_inputs, input_dim]
            context: Optional context [..., context_dim]
        Returns:
            selected: Weighted sum [..., hidden_dim]
            weights: Variable weights [..., num_inputs]
        """
        batch_shape = inputs.shape[:-2]

        # Process each variable
        processed = []
        for i in range(self.num_inputs):
            var_input = inputs[..., i, :]
            processed.append(self.grn_per_var[i](var_input))

        # Stack and flatten for weight computation
        processed_stack = torch.stack(processed, dim=-2)  # [..., num_inputs, hidden_dim]
        flattened = processed_stack.flatten(start_dim=-2)  # [..., num_inputs * hidden_dim]

        # Compute variable weights
        weights = self.grn_weights(flattened, context)
        weights = F.softmax(weights, dim=-1)  # [..., num_inputs]

        # Weighted combination
        selected = (processed_stack * weights.unsqueeze(-1)).sum(dim=-2)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable attention weights.
    Modified from standard attention to use additive attention per head.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, seq_q, embed_dim]
            key: [batch, seq_k, embed_dim]
            value: [batch, seq_k, embed_dim]
            mask: Optional attention mask
        Returns:
            output: [batch, seq_q, embed_dim]
            attn_weights: [batch, num_heads, seq_q, seq_k]
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Project and reshape for multi-head
        Q = self.q_proj(query).view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_q, self.embed_dim)
        output = self.out_proj(context)

        return output, attn_weights


class TemporalFusionEncoder(nn.Module):
    """
    Core TFT encoder that processes temporal sequences.

    Architecture:
    1. Variable selection for observed inputs
    2. LSTM encoder for local temporal patterns
    3. Self-attention for long-range dependencies
    4. GRN for final processing
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        static_dim: Optional[int] = None,
        event_latent_dim: int = 0,
    ):
        """
        Args:
            input_dim: Dimension of time-varying input features
            hidden_dim: Hidden dimension throughout
            num_heads: Number of attention heads
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            static_dim: Dimension of static features (spatial embeddings)
            event_latent_dim: Dimension of event latent (c_e)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim
        self.event_latent_dim = event_latent_dim

        # Context dimension for conditioning
        context_dim = 0
        if static_dim is not None:
            context_dim += static_dim
        if event_latent_dim > 0:
            context_dim += event_latent_dim
        self.context_dim = context_dim if context_dim > 0 else None

        # Static context processing (FiLM-style conditioning)
        if self.context_dim is not None:
            self.static_enrichment = GatedResidualNetwork(
                hidden_dim, hidden_dim, hidden_dim,
                context_dim=self.context_dim, dropout=dropout
            )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Temporal encoder (LSTM for local patterns)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False,  # Causal for autoregressive
        )

        # Gate for LSTM output
        self.lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Self-attention for long-range dependencies
        self.self_attention = InterpretableMultiHeadAttention(
            hidden_dim, num_heads, dropout
        )
        self.attn_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Position encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=500)

        # Final GRN
        self.final_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
        event_latent: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            static_context: Static features [batch, static_dim]
            event_latent: Event latent [batch, event_latent_dim]
            hidden: Optional LSTM hidden state
            mask: Optional attention mask

        Returns:
            output: Encoded sequence [batch, seq_len, hidden_dim]
            hidden: LSTM hidden state
            attn_weights: Attention weights [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Build context vector
        context_parts = []
        if static_context is not None:
            context_parts.append(static_context)
        if event_latent is not None:
            context_parts.append(event_latent)

        if context_parts:
            context = torch.cat(context_parts, dim=-1)
            # Expand to sequence length
            context_seq = context.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            context = None
            context_seq = None

        # Input projection
        h = self.input_proj(x)

        # Add positional encoding
        h = self.pos_encoder(h)

        # Static enrichment (condition on context)
        if self.context_dim is not None and context is not None:
            h = self.static_enrichment(h, context_seq)

        # LSTM encoder
        lstm_out, hidden = self.lstm(h, hidden)

        # Gated skip connection for LSTM
        lstm_gated = self.lstm_gate(lstm_out)
        h = self.lstm_norm(lstm_gated + h)

        # Self-attention with causal mask
        if mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            causal_mask = ~causal_mask  # True = attend, False = mask
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        else:
            causal_mask = mask

        attn_out, attn_weights = self.self_attention(h, h, h, causal_mask)

        # Gated skip connection for attention
        attn_gated = self.attn_gate(attn_out)
        h = self.attn_norm(attn_gated + h)

        # Final GRN
        output = self.final_grn(h)

        return output, hidden, attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHorizonHead(nn.Module):
    """
    Multi-horizon prediction head that outputs all future timesteps at once.
    Reduces autoregressive error accumulation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        horizon: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Number of output features per timestep
            horizon: Number of future timesteps to predict
        """
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim

        # Per-horizon GRNs for temporal weighting
        self.horizon_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(horizon)
        ])

        # Shared output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        encoder_output: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: [batch, seq_len, input_dim]
            context: Optional context [batch, context_dim]

        Returns:
            predictions: [batch, horizon, output_dim]
        """
        # Use last encoder state as base
        # Could also use attention over all states here
        base = encoder_output[:, -1]  # [batch, input_dim]

        predictions = []
        for h in range(self.horizon):
            # Process with horizon-specific GRN
            h_out = self.horizon_grns[h](base)
            pred = self.output_proj(h_out)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)  # [batch, horizon, output_dim]


class TemporalFusionDecoder(nn.Module):
    """
    TFT Decoder for multi-horizon prediction.
    Combines encoder outputs with known future inputs (rainfall).
    """

    def __init__(
        self,
        encoder_dim: int,
        known_future_dim: int,  # e.g., rainfall
        hidden_dim: int,
        output_dim: int,
        horizon: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim

        # Project known future inputs
        if known_future_dim > 0:
            self.future_proj = nn.Linear(known_future_dim, hidden_dim)
        else:
            self.future_proj = None

        # Combine encoder output with future context
        combine_dim = hidden_dim if known_future_dim > 0 else 0
        self.combine_grn = GatedResidualNetwork(
            encoder_dim + combine_dim, hidden_dim, hidden_dim, dropout=dropout
        )

        # Multi-head attention over encoder states
        self.cross_attention = InterpretableMultiHeadAttention(
            hidden_dim, num_heads, dropout
        )

        # Per-horizon output heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(horizon)
        ])

    def forward(
        self,
        encoder_output: torch.Tensor,
        known_future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: [batch, seq_len, encoder_dim]
            known_future: [batch, horizon, known_future_dim] (e.g., future rainfall)

        Returns:
            predictions: [batch, horizon, output_dim]
        """
        batch_size = encoder_output.shape[0]

        predictions = []
        for h in range(self.horizon):
            # Get context from last encoder state
            base = encoder_output[:, -1]  # [batch, encoder_dim]

            # Add known future if available
            if known_future is not None and self.future_proj is not None:
                future_h = self.future_proj(known_future[:, h])
                combined = torch.cat([base, future_h], dim=-1)
            else:
                combined = base

            # Process through GRN
            h_out = self.combine_grn(combined)
            h_out = h_out.unsqueeze(1)  # [batch, 1, hidden_dim]

            # Cross-attention over encoder states
            attn_out, _ = self.cross_attention(h_out, encoder_output, encoder_output)
            attn_out = attn_out.squeeze(1)  # [batch, hidden_dim]

            # Horizon-specific output
            pred = self.horizon_heads[h](attn_out)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)  # [batch, horizon, output_dim]


class SpatioTemporalTFT(nn.Module):
    """
    Spatiotemporal TFT encoder that processes nodes with shared temporal patterns.
    Combines spatial embeddings with TFT temporal processing.

    This replaces SpatioTemporalEncoder from the original CLDTS.
    """

    def __init__(
        self,
        spatial_dim: int,
        dynamic_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        event_latent_dim: int = 0,
    ):
        """
        Args:
            spatial_dim: Dimension of spatial embeddings from GNN
            dynamic_dim: Dimension of dynamic input features
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            event_latent_dim: Dimension of event latent
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim

        # Input projection (combine spatial + dynamic)
        input_dim = spatial_dim + dynamic_dim

        # TFT encoder
        self.tft_encoder = TemporalFusionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            static_dim=None,  # Static already concatenated
            event_latent_dim=event_latent_dim,
        )

    def forward(
        self,
        spatial_emb: torch.Tensor,
        dynamic_seq: torch.Tensor,
        event_latent: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Encode spatiotemporal features using TFT.

        Args:
            spatial_emb: Static spatial embeddings [num_nodes, spatial_dim]
            dynamic_seq: Dynamic features [batch, seq_len, num_nodes, dynamic_dim]
            event_latent: Event latent [batch, event_latent_dim]
            hidden: Optional LSTM hidden state

        Returns:
            output: Encoded sequence [batch, seq_len, num_nodes, hidden_dim]
            hidden: Updated LSTM hidden state (or None for stateless operation)
        """
        batch, seq_len, num_nodes, dynamic_dim = dynamic_seq.shape
        device = dynamic_seq.device

        # Expand spatial embeddings to match sequence
        spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1)

        # Concatenate spatial and dynamic features
        combined = torch.cat([spatial_expanded, dynamic_seq], dim=-1)

        # Process each node through shared TFT encoder
        # Reshape: [batch, seq_len, num_nodes, input_dim] -> [batch*num_nodes, seq_len, input_dim]
        combined = combined.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, -1)

        # Expand event latent for all nodes
        if event_latent is not None:
            event_latent_expanded = event_latent.unsqueeze(1).expand(-1, num_nodes, -1)
            event_latent_expanded = event_latent_expanded.reshape(batch * num_nodes, -1)
        else:
            event_latent_expanded = None

        # Forward through TFT encoder
        output, new_hidden, attn_weights = self.tft_encoder(
            combined,
            event_latent=event_latent_expanded,
            hidden=hidden,
        )

        # Reshape back: [batch*num_nodes, seq_len, hidden_dim] -> [batch, seq_len, num_nodes, hidden_dim]
        output = output.reshape(batch, num_nodes, seq_len, self.hidden_dim)
        output = output.permute(0, 2, 1, 3)  # [batch, seq_len, num_nodes, hidden_dim]

        return output, new_hidden

    def get_initial_state(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get zero initial hidden state for LSTM."""
        num_layers = self.tft_encoder.lstm.num_layers
        hidden_dim = self.hidden_dim

        h0 = torch.zeros(num_layers, batch_size * num_nodes, hidden_dim, device=device)
        c0 = torch.zeros(num_layers, batch_size * num_nodes, hidden_dim, device=device)

        return (h0, c0)


class MultiHorizonDecoder(nn.Module):
    """
    Decoder that predicts multiple horizons at once for a specific node type.
    Supports conditioning on known future inputs (rainfall).
    """

    def __init__(
        self,
        latent_dim: int,
        spatial_dim: int,
        event_latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        horizon: int,
        known_future_dim: int = 0,
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of encoder output
            spatial_dim: Dimension of spatial embeddings
            event_latent_dim: Dimension of event latent
            hidden_dim: Hidden dimension
            output_dim: Number of output features per timestep
            horizon: Number of future timesteps to predict
            known_future_dim: Dimension of known future inputs (e.g., rainfall)
        """
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.known_future_dim = known_future_dim

        input_dim = latent_dim + spatial_dim + event_latent_dim
        if known_future_dim > 0:
            input_dim += known_future_dim

        # Per-horizon MLPs
        self.horizon_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(horizon)
        ])

    def forward(
        self,
        z: torch.Tensor,
        spatial_emb: torch.Tensor,
        c_e: torch.Tensor,
        known_future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode multiple horizons at once.

        Args:
            z: Encoder output [batch, num_nodes, latent_dim]
            spatial_emb: Spatial embeddings [num_nodes, spatial_dim]
            c_e: Event latent [batch, event_latent_dim]
            known_future: Optional known future inputs [batch, horizon, num_nodes, known_future_dim]

        Returns:
            predictions: [batch, horizon, num_nodes, output_dim]
        """
        batch, num_nodes, latent_dim = z.shape

        # Expand spatial embeddings
        spatial_expanded = spatial_emb.unsqueeze(0).expand(batch, -1, -1)  # [batch, num_nodes, spatial_dim]

        # Expand event latent
        c_e_expanded = c_e.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch, num_nodes, event_latent_dim]

        predictions = []
        for h in range(self.horizon):
            # Base input
            inputs = [z, spatial_expanded, c_e_expanded]

            # Add known future if available
            if known_future is not None and self.known_future_dim > 0:
                inputs.append(known_future[:, h])  # [batch, num_nodes, known_future_dim]

            combined = torch.cat(inputs, dim=-1)

            # Predict for this horizon
            pred = self.horizon_mlps[h](combined)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)  # [batch, horizon, num_nodes, output_dim]
