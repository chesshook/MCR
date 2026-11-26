"""MCR model trained from random initialization; no pretrained language model."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ByteSemanticEncoder(nn.Module):
    """Non-autoregressive UTF-8 byte encoder with CNN + bidirectional GRU pooling."""
    def __init__(self, embedding_dim: int = 256, conv_channels: int = 512,
                 recurrent_hidden: int = 512, output_dim: int = 1024,
                 dropout: float = 0.15):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embedding_dim, conv_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embedding_dim, conv_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embedding_dim, conv_channels, kernel_size=7, padding=3)
        self.conv_fuse = nn.Linear(conv_channels * 3, 256)
        self.conv_norm = nn.LayerNorm(256)
        self.residual_conv = nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2)
        self.residual_norm = nn.LayerNorm(256)
        self.gru = nn.GRU(
            input_size=256, hidden_size=recurrent_hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        recurrent_dim = recurrent_hidden * 2
        self.attention = nn.Linear(recurrent_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(recurrent_dim * 3, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("ids and mask must be [batch, sequence]")
        lengths = mask.sum(dim=1).clamp(min=1).to(torch.long)
        x = self.embedding(ids).transpose(1, 2)
        c = torch.cat((F.gelu(self.conv3(x)), F.gelu(self.conv5(x)), F.gelu(self.conv7(x))), dim=1)
        x = self.conv_fuse(c.transpose(1, 2))
        x = F.gelu(self.conv_norm(x))
        residual = self.residual_conv(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(self.residual_norm(x + residual))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        sequence, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=ids.shape[1])
        valid = mask.to(torch.bool)
        scores = self.attention(sequence).squeeze(-1).float()
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).to(sequence.dtype)
        attended = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        mask_f = valid.unsqueeze(-1).to(sequence.dtype)
        mean = (sequence * mask_f).sum(dim=1) / lengths.unsqueeze(-1).to(sequence.dtype)
        max_values = sequence.masked_fill(~valid.unsqueeze(-1), torch.finfo(sequence.dtype).min).amax(dim=1)
        pooled = torch.cat((attended, mean, max_values), dim=-1)
        return F.normalize(self.projection(self.dropout(pooled)), p=2, dim=-1)


class MCRModel(nn.Module):
    """Shared semantic space with retrieval and three-way sentence-relation heads."""
    def __init__(self, embedding_dim: int = 256, conv_channels: int = 512,
                 recurrent_hidden: int = 512, output_dim: int = 1024,
                 dropout: float = 0.15):
        super().__init__()
        self.encoder = ByteSemanticEncoder(
            embedding_dim=embedding_dim, conv_channels=conv_channels,
            recurrent_hidden=recurrent_hidden, output_dim=output_dim, dropout=dropout,
        )
        self.relation_head = nn.Sequential(
            nn.Linear(output_dim * 4, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, task: str, **batch):
        if task == "chat":
            query = self.encoder(batch["context_ids"], batch["context_mask"])
            reply = self.encoder(batch["reply_ids"], batch["reply_mask"])
            return query, reply
        if task == "nli":
            premise = self.encoder(batch["premise_ids"], batch["premise_mask"])
            hypothesis = self.encoder(batch["hypothesis_ids"], batch["hypothesis_mask"])
            features = torch.cat((premise, hypothesis, premise * hypothesis,
                                  torch.abs(premise - hypothesis)), dim=-1)
            return self.relation_head(features)
        raise ValueError(f"Unknown MCR training task: {task}")


def contrastive_loss(query: torch.Tensor, reply: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = (query @ reply.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
