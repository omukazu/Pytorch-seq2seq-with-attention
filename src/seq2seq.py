from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from constants import PAD, EOS
from model_components import Encoder, Decoder


class Seq2seq(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 embeddings: torch.Tensor or None,
                 max_seq_len: int,
                 attention: bool = False,
                 bi_directional: bool = True,
                 dropout_rate: float = 0.333,
                 n_layer: int = 2):
        super(Seq2seq, self).__init__()
        self.attention = attention
        self.bi_directional = bi_directional
        self.d_emb = embeddings.size(1)
        self.max_seq_len = max_seq_len
        self.vocab_size = embeddings.size(0) - 1

        # TODO: specify whether to freeze or not
        self.source_embed = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=True)
        self.target_embed = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=True)

        self.d_out = d_hidden * 2
        self.n_enc_layer = n_layer
        self.n_direction = 2 if bi_directional else 1
        self.n_dec_layer = n_layer

        self.encoder = Encoder(nn.LSTM(input_size=self.d_emb, hidden_size=d_hidden, num_layers=self.n_enc_layer,
                                       batch_first=True, dropout=dropout_rate, bidirectional=bi_directional))
        self.decoder = Decoder(nn.LSTM(input_size=self.d_emb, hidden_size=self.d_out, num_layers=self.n_dec_layer,
                                       batch_first=True, dropout=dropout_rate, bidirectional=False))
        self.w = nn.Linear(self.d_out, self.vocab_size)

    def forward(self,
                source: torch.Tensor,
                source_mask: torch.Tensor,
                target: torch.Tensor,
                target_mask: torch.Tensor
                ) -> torch.Tensor:
        batch_size = source.size(0)
        source_embedded = self.source_embed(source)  # (batch, max_seq_len, d_emb)
        enc_out, h = self.encoder(source_embedded, source_mask, None)
        if self.bi_directional:
            # (n_layer * n_direction, batch, d_hidden) -> (n_layer, batch, d_hidden * n_direction)
            h = [self.transform(batch_size, e) for e in h]

        output = source_embedded.new_zeros(batch_size, self.max_seq_len + 1, self.vocab_size)
        target_embedded = self.target_embed(target).transpose(0, 1).unsqueeze(2)  # (max_seq_len + 1, batch, 1, d_emb)
        # decode per word
        for i in range(self.max_seq_len + 1):
            if self.attention:
                pass  # TODO: calculate attention
            dec_out, h = self.decoder(target_embedded[i], target_mask[:, i], h)
            output[:, i, :] = self.w(dec_out.squeeze(1))  # (batch, vocab_size)
        return output

    def predict(self,
                source: torch.Tensor,       # (batch, max_seq_len, d_emb)
                source_mask: torch.Tensor,  # (batch, max_seq_len)
                ) -> List[List[int]]:       # (batch, max_seq_len + 1)
        self.eval()
        with torch.no_grad():
            batch_size = source.size(0)
            source_embedded = self.source_embed(source)  # (batch, max_seq_len, d_emb)
            enc_out, h = self.encoder(source_embedded, source_mask, None)
            if self.bi_directional:
                h = [self.transform(batch_size, e) for e in h]

            tensor_type = 'torch.cuda.LongTensor' if source.device.index is not None else 'torch.LongTensor'
            target = torch.full((batch_size, 1), EOS).type(tensor_type).to(source.device)
            mask_type = 'torch.cuda.ByteTensor' if source.device.index is not None else 'torch.ByteTensor'
            target_mask = torch.full((batch_size, 1), 1).type(mask_type).squeeze(-1).to(source.device)
            dec_out, h = self.decoder(self.target_embed(target), target_mask, h)
            output = torch.argmax(self.w(dec_out.squeeze(1)), dim=1) + 1
            output = output.unsqueeze(1)
            for i in range(1, self.max_seq_len + 1):
                if self.attention:
                    pass
                dec_out, h = self.decoder(dec_out, target_mask, h)
                prediction = torch.argmax(self.w(dec_out.squeeze(1)), dim=1) + 1  # (batch, vocab_size)
                target_mask = target_mask * prediction.ne(EOS)
                output = torch.cat((output, prediction.unsqueeze(1)), dim=1)
                if sum(target_mask) == 0:
                    break
        return output

    def transform(self,
                  batch_size: int,
                  h: torch.Tensor
                  ) -> torch.Tensor:
        h = h.contiguous().view(self.n_enc_layer, self.n_direction, batch_size, -1)
        # hidden[: n_layer] <- extract last hidden layer
        h = h.transpose(1, 2)[:self.n_dec_layer]
        h = h.contiguous().view(self.n_dec_layer, batch_size, -1)
        return h
