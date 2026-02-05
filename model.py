from torch import nn
import torch

class pressureInsolesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_classes,
        num_encoder_layers=2,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
        seq_len=436
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_dim, latent_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.output_proj = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
    
        x_emb = self.input_proj(x) + self.pos_embedding
        latent_seq = self.encoder(x_emb)
        out = self.output_proj(latent_seq)
        
        return out