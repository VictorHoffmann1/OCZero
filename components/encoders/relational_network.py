import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, top_k=64):
        super().__init__()
        # Self-Interaction MLP
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pairwise interaction MLP (excluding self-interaction)
        self.xi = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Top-K attention mechanism to select important interactions
        self.top_k_attention = TopKAttention(
            input_dim=input_dim,  # Input dimension for attention
            proj_dim=hidden_dim,  # Projection dimension for attention
            top_k=top_k,  # Number of top interactions to select
        )

        # Global MLP after pooling
        self.rho = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,  # Output dimension of the global MLP
                hidden_dim,
            ),
        )

        self.hidden_dim = hidden_dim

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x, obj_padding_mask = self.trim(x)  # Remove zero-padded objects
        B = x.shape[0]

        # Get top-k interactions
        ij_idxs, ij_weights = self.top_k_attention(x, padding_mask=obj_padding_mask)

        top_k = ij_idxs.shape[1]  # Number of top-k interactions

        # Efficient self-mask computation
        self_mask = ij_idxs[..., 0] == ij_idxs[..., 1]  # (B, top_k)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, top_k)

        # Vectorized indexing
        i_idx, j_idx = ij_idxs[..., 0], ij_idxs[..., 1]
        x_i = x[batch_indices, i_idx]  # (B, top_k, 6)
        x_j = x[batch_indices, j_idx]  # (B, top_k, 6)

        # Parallel feature computation
        feat_self = self.phi(x_i)  # (B, top_k, H)
        feat_pair = self.xi(torch.cat([x_i, x_j], dim=-1))  # (B, top_k, H)

        # Efficient conditional selection
        interaction_feat = torch.where(
            self_mask.unsqueeze(-1), feat_self, feat_pair
        )  # (B, top_k, H)

        # Optimized weighted pooling
        pooled = torch.sum(interaction_feat * ij_weights.unsqueeze(-1), dim=1)  # (B, H)

        return pooled
        # return self.rho(pooled)  # (B, H)

    @staticmethod
    def trim(x):
        """
        Remove trailing zero-padded objects from the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D) with zero-padded objects.
        Returns:
            torch.Tensor: Tensor with zero-padded objects trimmed, shape (B, max_valid_N, D).
        """

        obj_padding_mask = x.abs().sum(dim=-1) != 0  # (B, N)

        max_valid = obj_padding_mask.sum(dim=1).max()

        return x[:, :max_valid, :], obj_padding_mask[
            :, :max_valid
        ]  # Return trimmed tensor and mask

    @staticmethod
    def is_collision(x, y, eps=1.0):
        """
        Check if two objects collide based on their positions, shapes and speeds.
        Args:
            x (torch.Tensor): Position tensor of shape (B, top_k, D).
            y (torch.Tensor): Position tensor of shape (B, top_k, D).
        Returns:
            torch.Tensor: Boolean tensor indicating collisions, shape (B, top_k).
        """
        # Get positions, shapes and speeds
        x_pos, x_speed, x_shape = x[..., :2], x[..., 2:4], x[..., 4:6]
        y_pos, y_speed, y_shape = y[..., :2], y[..., 2:4], y[..., 4:6]

        # Check if the objects collide / overlap
        cond1 = torch.abs(x_pos - y_pos) < (x_shape + y_shape) / 2

        # Check if there will be a collision in the next step, since there can be object detection errors
        # when two objects touch, they can be detected as one single object
        cond2 = (
            torch.abs(x_pos + x_speed - y_pos - y_speed) < (x_shape + y_shape) / 2 + eps
        )

        return (cond1[..., 0] & cond1[..., 1]) | (cond2[..., 0] & cond2[..., 1])


class TopKAttention(nn.Module):
    """
    Top-K attention mechanism to select important interactions.
    This module computes pairwise attention scores and selects the top-k pairs.
    """

    def __init__(self, input_dim, proj_dim, top_k, verbose=False):
        """
        Initialize the Top-K attention mechanism.
        """
        super().__init__()
        self.top_k = top_k
        self.scale = proj_dim**-0.5

        self.query_proj = nn.Linear(input_dim, proj_dim)  # Query projection
        self.key_proj = nn.Linear(input_dim, proj_dim)  # Key projection

        self.verbose = verbose
        if verbose:
            self.count = 0

    def forward(self, x, padding_mask=None):
        B, L, _ = x.shape

        Q = self.query_proj(x)  # (B, L, proj_dim)
        K = self.key_proj(x)  # (B, L, proj_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Efficient masking
        if padding_mask is not None:
            # Create 2D mask efficiently
            mask_2d = padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
            attn_scores.masked_fill_(~mask_2d, float("-inf"))

        # Flatten and get top-k
        attn_scores_flat = attn_scores.view(B, -1)
        topk_vals, topk_idx = torch.topk(
            attn_scores_flat, min(self.top_k, L**2), dim=-1, largest=True
        )
        topk_weights = F.softmax(topk_vals, dim=-1)

        # Convert indices efficiently
        row_idx = topk_idx // L
        col_idx = topk_idx % L
        topk_indices = torch.stack([row_idx, col_idx], dim=-1)

        if self.verbose:
            if self.count % 1000 == 0:  # Print every 1000 calls
                print(f"Top-{min(L**2, 10)} Pairs:")
                for i in range(min(L**2, 10)):
                    print(
                        f"{topk_indices[0, i, 0].item()} -> {topk_indices[0, i, 1].item()} with weight {topk_weights[0][i].item():.2f}"
                    )
                print(f"Number of objects: {L}")
            self.count += 1

        return topk_indices, topk_weights
