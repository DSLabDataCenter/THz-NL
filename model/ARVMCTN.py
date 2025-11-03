import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    """A simplified Transformer encoder layer with multi-head self-attention and feed-forward network."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # QKV projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.GELU()

    def forward(self, src):
        B, L, D = src.size()

        # === Multi-head Self-Attention ===
        Q = self.q_proj(src).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, L, head_dim]
        K = self.k_proj(src).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(src).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)  # [B, nhead, L, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)
        src2 = self.dropout(attn_output)
        src = self.norm1(src + src2)

        # === Feed-Forward ===
        ff_output = self.fc2(self.dropout(self.activation(self.fc1(src))))
        src2 = self.dropout(ff_output)
        src = self.norm2(src + src2)

        return src


class MultiScaleConvEmbedding(nn.Module):
    """Multi-scale 1D convolutional embedding layer."""
    def __init__(self, in_dim, embed_dim, kernel_sizes, stride=1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=embed_dim,
                kernel_size=k,
                stride=stride,
                padding=k // 2  # keep temporal length unchanged
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        # x: [B, L, C_in] → permute to [B, C_in, L]
        x = x.permute(0, 2, 1)
        conv_outputs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outputs, dim=1)  # concatenate along channel dimension
        out = out.permute(0, 2, 1)  # [B, L, C_out]
        return out


class ARVMCTN(nn.Module):
    """
    Adaptive Reparameterized Variable Modulation Convolutional Transformer Network (ARVMCTN).
    Combines multi-scale convolutional embedding, Transformer encoder, and parameter-conditioned scaling.
    """
    def __init__(self, in_dim=2, embed_dim=12, kernel_sizes=(3, 5, 7),
                 stride=1, num_heads=2, num_layers=1, out_dim=2,
                 dropout=0.1, seq_len=11, param_dim=3):
        super().__init__()

        # === Embedding ===
        self.embedding = MultiScaleConvEmbedding(
            in_dim=in_dim,
            embed_dim=embed_dim,
            kernel_sizes=kernel_sizes,
            stride=stride
        )
        self.total_embed_dim = embed_dim * len(kernel_sizes)
        self.seq_len = seq_len

        # === Transformer Encoder Layers ===
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.total_embed_dim,
                nhead=num_heads,
                dim_feedforward=self.total_embed_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # === Parameter-conditioned scaling ===
        self.gamma_mlp = nn.Sequential(
            nn.Linear(param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.total_embed_dim * seq_len)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.total_embed_dim * seq_len)
        )

        # === Regressor ===
        self.regressor = nn.Sequential(
            nn.Linear(self.total_embed_dim * seq_len, self.total_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.total_embed_dim // 2, out_dim)
        )

    def forward(self, x, params):
        """
        Args:
            x: Tensor [B, L, in_dim] — input sequence
            params: Tensor [B, param_dim] — external conditioning parameters
        Returns:
            out: Tensor [B, out_dim]
        """
        B = x.size(0)

        # Embedding
        x = self.embedding(x)

        # Transformer encoding
        for layer in self.layers:
            x = layer(x)

        # Flatten sequence
        x = x.reshape(B, -1)

        # Parameter-conditioned affine transformation
        gamma = self.gamma_mlp(params)
        beta = self.beta_mlp(params)
        x = gamma * x + beta

        # Regression output
        out = self.regressor(x)
        return out


if __name__ == "__main__":
    model = ARVMCTN(param_dim=3)
    x = torch.randn(8, 11, 2)      # [batch, seq_len, in_dim]
    params = torch.randn(8, 3)     # [batch, param_dim]
    out = model(x, params)
    print(out.shape)  # Expected: [8, 2]
