from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import EOS
from model_components import Encoder, Decoder, Embedder


class Seq2seq(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 source_embeddings: torch.Tensor,
                 target_embeddings: torch.Tensor,
                 max_seq_len: int,
                 attention: bool = False,
                 bi_directional: bool = True,
                 dropout_rate: float = 0.333,
                 n_layer: int = 2):
        super(Seq2seq, self).__init__()
        self.attention = attention
        self.bi_directional = bi_directional
        self.d_s_emb = source_embeddings.size(1)
        self.d_t_emb = target_embeddings.size(1)
        self.max_seq_len = max_seq_len
        self.vocab_size = target_embeddings.size(0)

        self.source_embed = Embedder(source_embeddings)
        self.target_embed = Embedder(target_embeddings)

        self.d_dec_hidden = d_hidden
        self.n_enc_layer = n_layer
        self.n_direction = 2 if bi_directional else 1
        self.n_dec_layer = 1

        self.encoder = Encoder(nn.LSTM(input_size=self.d_s_emb, hidden_size=d_hidden,
                                       num_layers=self.n_enc_layer, batch_first=True,
                                       dropout=dropout_rate, bidirectional=bi_directional))
        self.decoder = Decoder(nn.LSTM(input_size=self.d_t_emb, hidden_size=self.d_dec_hidden,
                                       num_layers=self.n_dec_layer, batch_first=True,
                                       dropout=dropout_rate, bidirectional=False))
        self.w = nn.Linear(self.d_dec_hidden, self.vocab_size)

    def forward(self,
                source: torch.Tensor,       # (batch, max_source_len, d_emb)
                source_mask: torch.Tensor,  # (batch, max_source_len)
                target: torch.Tensor,       # (batch, max_target_len, 1), word_ids
                target_mask: torch.Tensor   # (batch, max_target_len)
                ) -> torch.Tensor:          # (batch, max_target_len, d_emb)
        batch_size = source.size(0)
        source_embedded = self.source_embed(source, source_mask, False)  # (batch, max_source_len, d_emb)
        enc_out, h = self.encoder(source_embedded, source_mask)
        # (n_enc_layer * bi_direction, batch, d_hidden) -> (n_dec_layer, batch, d_hidden)
        h = (self.transform(batch_size, h[0]),
             self.transform(batch_size, h[1].new_zeros(h[1].size())))

        max_target_len = target.size(1)
        output = source_embedded.new_zeros(batch_size, max_target_len, self.vocab_size)
        target_embedded = self.target_embed(target, target_mask, False)  # (batch, max_target_len, d_emb)
        target_embedded = target_embedded.transpose(0, 1).unsqueeze(2)   # (max_target_len, batch, 1, d_emb)
        for i in range(max_target_len):
            if self.attention:
                pass  # TODO: calculate attention
            dec_out, h = self.decoder(target_embedded[i], target_mask[:, i], h)
            output[:, i, :] = self.w(dec_out.squeeze(1))  # (batch, vocab_size)
        return output

    def predict(self,
                source: torch.Tensor,       # (batch, max_seq_len, d_emb)
                source_mask: torch.Tensor,  # (batch, max_seq_len)
                ) -> torch.Tensor:          # (batch, seq_len)
        self.eval()
        with torch.no_grad():
            batch_size = source.size(0)
            source_embedded = self.source_embed(source, source_mask, False)  # (batch, max_seq_len, d_emb)
            enc_out, h = self.encoder(source_embedded, source_mask)
            h = (self.transform(batch_size, h[0]),
                 self.transform(batch_size, h[1].new_zeros(h[1].size())))

            target_id = torch.full((batch_size, 1), EOS, dtype=source.dtype).to(source.device)
            target_mask = torch.full([batch_size], 1, dtype=source_mask.dtype).to(source_mask.device)
            output = source_embedded.new_zeros(batch_size, self.max_seq_len, 1)
            for i in range(self.max_seq_len):
                if self.attention:
                    pass
                target_embedded = self.target_embed(target_id, target_mask, True)
                dec_out, h = self.decoder(target_embedded, target_mask, h)
                outs = self.w(dec_out.squeeze(1))
                prediction = torch.argmax(F.softmax(outs, dim=1), dim=1)                # (batch), greedy
                target_mask = target_mask * prediction.ne(EOS).type(target_mask.dtype)
                target_id = prediction.unsqueeze(1)
                output[:, i, :] = prediction.unsqueeze(1)
        return output

    def transform(self,
                  batch_size: int,
                  h: torch.Tensor
                  ) -> torch.Tensor:
        h = h.contiguous().view(self.n_enc_layer, self.n_direction, batch_size, -1)
        if self.bi_directional:
            h = h[:, 0, :, :] + h[:, 1, :, :]
        h = h.squeeze(1)          # (n_enc_layer, batch, d_enc_hidden)
        # hidden[: n_layer] <- extract n-last hidden layer
        h = h[:self.n_dec_layer]  # (n_dec_layer * 1, batch, d_hidden)
        return h
