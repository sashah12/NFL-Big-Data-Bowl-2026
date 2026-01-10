import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


class AttnEncoderLayer(nn.TransformerEncoderLayer):

    def forward(self, src, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False    # → keep per-head weights
        )
        
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)

        return src, attn_weights

class PlayTransformer(nn.Module):
    def __init__(self, continuous_dim, num_positions=30, embed_dim=8, d_model=256, nhead=16, num_layers=6):
        super().__init__()
        # Increase nhead to 16 to handle the extra context dimensions effectively

        # Learnable embeddings for Offense (5 slots) and Defense (11 slots)
        self.off_pos_embedding = nn.Embedding(num_positions, embed_dim)
        self.def_pos_embedding = nn.Embedding(num_positions, embed_dim)
        
        # New input dimension: 
        # Continuous (170+) + Offense (5*8=40) + Defense (11*8=88)
        combined_input_dim = continuous_dim + (5 * embed_dim) + (11 * embed_dim)
        self.in_proj = nn.Linear(combined_input_dim, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=0.2,
                dim_feedforward=512
            )
            for _ in range(num_layers)
        ])

        self.ball_head = nn.Linear(d_model, 2)

    def forward(self, x_cont, x_off_ids, x_def_ids, mask):
        batch_size, seq_len, _ = x_cont.shape

        # 1. Process Offensive Embeddings: (B, 5) -> (B, 40)
        off_feats = self.off_pos_embedding(x_off_ids).view(batch_size, -1) 
        
        # 2. Process Defensive Embeddings: (B, 11) -> (B, 88)
        def_feats = self.def_pos_embedding(x_def_ids).view(batch_size, -1)
        
        # 3. Concatenate and Broadcast static context across the time dimension
        # Resulting shape: (B, SeqLen, 128)
        static_context = torch.cat([off_feats, def_feats], dim=-1)
        static_context = static_context.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 4. Final Feature Fusion
        x = torch.cat([x_cont, static_context], dim=-1)
        x = self.in_proj(x)

        # 5. Transformer Causal Pass
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = layer(
                x, 
                src_mask=causal_mask, 
                src_key_padding_mask=~mask,
                is_causal=True 
            )

        logits = self.ball_head(x)
        x_coords = torch.sigmoid(logits[..., 0]) * 120
        y_coords = torch.sigmoid(logits[..., 1]) * 53.3
        
        return {"ball_location": torch.stack([x_coords, y_coords], dim=-1)}


