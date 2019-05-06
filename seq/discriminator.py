from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,
                 n_filter: int,
                 kernel_window: Tuple,
                 max_seq_len: int):
        super(CNN, self).__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=n_filter, kernel_size=kernel_window)
        self.bn = nn.BatchNorm2d(n_filter, 1)
        self.pool = nn.MaxPool2d(max_seq_len)

    def forward(self,
                x: torch.Tensor,  # (batch, 1, max_seq_len, d_emb)
                ) -> torch.Tensor:
        cnn = self.cnn(x)                   # (batch, num_filters, max_seq_len)
        bn = F.relu(self.bn(cnn))           # (batch, num_filters, max_seq_len)
        pooled = self.pool(bn).squeeze(-1)  # (batch, num_filters)
        return pooled


class Discriminator(nn.Module):
    def __init__(self,
                 d_emb: int,
                 embeddings: torch.Tensor,
                 max_seq_len: int,
                 n_kernels: List[int],  # len(n_kernels) == 12
                 w_kernels: List[int],  # len(w_kernels) == 12
                 vocab_size: int,
                 dropout_rate: float = 0.333,
                 n_class: int = 2):
        super(Discriminator, self).__init__()
        self.d_emb = d_emb
        self.embeddings = embeddings
        self.max_seq_len = max_seq_len
        self.n_kernels = n_kernels
        self.w_kernels = w_kernels
        self.vocab = vocab_size

        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.poolings = nn.ModuleList(
            OrderedDict([(f'conv{i+1}', CNN(n_filter=kn, kernel_window=(kw, d_emb), max_seq_len=max_seq_len))
                         for i, (kn, kw) in enumerate(zip(n_kernels, w_kernels))])
        )

        # highway architecture
        total = sum(n_kernels)
        self.sigmoid = nn.Sigmoid()
        self.transform_gate = nn.Linear(total, total)
        self.highway = nn.Linear(total, total)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc = nn.Linear(total, n_class)

        self.softmax = nn.Softmax(dim=-1)

    # return predictions
    def forward(self,
                xs: torch.Tensor,  # positive and negative latter events
                ts: torch.Tensor,  # tag representing whether the latter event is generated or not
                ) -> torch.Tensor:
        embedded = self.embed(xs).unsqueeze(1)  # (batch, 1, max_seq_len, d_emb

        pooled = self.poolings[0](embedded)
        for pooling in self.poolings[1:]:
            pooled = torch.cat((pooled, pooling(embedded)), dim=1)  # (batch, total)

        t = F.sigmoid(self.transform_gate(pooled))
        h = t * F.relu(self.highway_output(pooled)) + (1 - t) * pooled
        h = self.dropout(h)

        disc_predictions = self.fc(h)  # (batch, n_class)
        return disc_predictions

    def get_rewards(self,
                    ys: torch.Tensor,  # generated latter events
                    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            embedded = self.embed(ys).unsqueeze(1)  # (batch, 1, max_seq_len, d_emb)
            pooled = self.poolings[0](embedded)
            for pooling in self.poolings[1:]:
                pooled = torch.cat((pooled, pooling(embedded)), dim=1)  # (batch, total)

            t = F.sigmoid(self.transform_gate(pooled))
            h = t * F.relu(self.highway_output(pooled)) + (1 - t) * pooled
            h = self.dropout(h)

            rewards = self.softmax(self.fc(h))  # (batch, n_class)
            # classes = (Generated, Real), return probabilities that the events are real
            return rewards[:, -1]
