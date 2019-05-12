from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_components import Embedder, CNNPooler


class Discriminator(nn.Module):
    def __init__(self,
                 d_t_emb: int,
                 max_seq_len: int,
                 n_kernels: List[int],  # len(n_kernels) == 12
                 w_kernels: List[int],  # len(w_kernels) == 12
                 target_embeddings: torch.Tensor,
                 dropout_rate: float = 0.333,
                 n_class: int = 2
                 ) -> None:
        super(Discriminator, self).__init__()
        self.d_t_emb = d_t_emb
        self.max_seq_len = max_seq_len
        self.n_kernels = n_kernels
        self.w_kernels = w_kernels

        self.target_vocab_size, self.d_t_emb = target_embeddings.size()
        self.target_embed = Embedder(self.target_vocab_size, self.d_t_emb)

        self.poolers = nn.ModuleList(
            OrderedDict([(f'conv{i + 1}', CNNPooler(n_filter=kn, kernel_window=(kw, d_t_emb), max_seq_len=max_seq_len))
                         for i, (kn, kw) in enumerate(zip(n_kernels, w_kernels))])
        )

        # highway architecture
        total = sum(n_kernels)
        self.sigmoid = nn.Sigmoid()
        self.transform_gate = nn.Linear(total, total)
        self.highway = nn.Linear(total, total)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc = nn.Linear(total, n_class)

    # return predictions
    def forward(self,
                xs: torch.Tensor,  # positive and negative latter events
                ts: torch.Tensor,  # tag representing whether the latter is generated or not
                ) -> torch.Tensor:
        embedded = self.target_embed(xs).unsqueeze(1)              # (b, 1, max_seq_len, d_t_emb)

        pooled = self.poolings[0](embedded)
        for pooler in self.poolers[1:]:
            pooled = torch.cat((pooled, pooler(embedded)), dim=1)  # (b, total)

        t = F.sigmoid(self.transform_gate(pooled))
        h = t * F.relu(self.highway_output(pooled)) + (1 - t) * pooled
        h = self.dropout(h)

        dis_predictions = self.fc(h)                               # (b, n_class)
        return dis_predictions

    def get_rewards(self,
                    ys: torch.Tensor,  # generated latter events
                    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            embedded = self.target_embed(ys).unsqueeze(1)               # (b, 1, max_seq_len, d_t_emb)
            pooled = self.poolers[0](embedded)
            for pooler in self.poolers[1:]:
                pooled = torch.cat((pooled, pooler(embedded)), dim=1)  # (b, total)

            t = F.sigmoid(self.transform_gate(pooled))
            h = t * F.relu(self.highway_output(pooled)) + (1 - t) * pooled
            h = self.dropout(h)

            rewards = F.softmax(self.fc(h), dim=1)                      # (b, n_class)
            # classes = (Generated, Real), return probabilities that the events are real
            return rewards[:, -1]
