from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


PAD = 0
EOS = 2


class Encoder(nn.Module):
    def __init__(self,
                 rnn: nn.Module,
                 dropout_rate: float = 0.333):
        super(Encoder, self).__init__()
        self.rnn = rnn
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                h: Optional[Tuple[torch.Tensor, torch.Tensor]]
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        lengths = mask.cumsum(dim=1)[:, -1]
        sorted_lengths, perm_indices = lengths.sort(0, descending=True)
        _, unperm_indices = perm_indices.sort(0)

        # masking
        packed = pack_padded_sequence(x[perm_indices], lengths=sorted_lengths, batch_first=True)
        # (sum(lengths), hid*2)
        output, h = self.rnn(packed, h)
        unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)
        # (batch, max_seq_len, d_hidden * 2), (n_layer * n_direction, batch, d_hidden * 2)
        # hidden[: n_layer] <- extract last hidden layer
        return self.dropout(unpacked[unperm_indices]), list(h)


class Decoder(nn.Module):
    def __init__(self,
                 rnn: nn.Module,
                 dropout_rate: float = 0.333):
        super(Decoder, self).__init__()
        self.rnn = rnn
        self.dropout = nn.Dropout(p=dropout_rate)

    # stateless
    def forward(self,
                x: torch.Tensor,                      # (batch, 1, d_emb)
                mask: torch.Tensor,                   # (batch, 1)
                hs: Tuple[torch.Tensor, torch.Tensor]  # ((n_lay * n_dir, b, d_hid), (n_lay * n_dir, b, d_hid))
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = x.size(0)
        valid_len = mask.sum()
        sorted_lengths, perm_indices = mask.sort(0, descending=True)
        _, unperm_indices = perm_indices.sort(0)

        if valid_len > 0:
            packed = pack_padded_sequence(x[perm_indices][:valid_len], lengths=sorted_lengths[:valid_len], batch_first=True)
            old_hs = [h.index_select(1, perm_indices)[:, valid_len:, :].contiguous() for h in hs]
            hs = [h.index_select(1, perm_indices)[:, :valid_len, :].contiguous() for h in hs]
            output, hs = self.rnn(packed, hs)
            unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)

            if batch_size > valid_len:
                _, _, d_out = unpacked.size()
                zeros = unpacked.new_zeros(batch_size - valid_len, 1, d_out)
                unpacked = torch.cat((unpacked, zeros), dim=0)  # (valid_len, 1, d_out) -> (batch, 1, d_out)
                unpacked = self.dropout(unpacked[unperm_indices])
                hs = tuple([torch.cat((h, old_h), dim=1) for h, old_h in zip(hs, old_hs)])
        else:
            _, batch_size, d_out = hs[0].size()
            unpacked = hs[0].new_zeros(batch_size, 1, d_out)
        return unpacked, hs


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

        self.source_embed = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=False)
        self.target_embed = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=False)

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
        embedded_s = self.source_embed(source)  # (batch, max_seq_len, d_emb)
        enc_out, h = self.encoder(embedded_s, source_mask, None)
        if self.bi_directional:
            # (n_layer * n_direction, batch, d_hidden) -> (n_layer, batch, d_hidden * n_direction)
            h[0] = h[0].contiguous().view(self.n_enc_layer, self.n_direction, batch_size, -1).transpose(1, 2)[:self.n_dec_layer].contiguous().view(self.n_dec_layer, batch_size, -1)
            h[1] = h[1].contiguous().view(self.n_enc_layer, self.n_direction, batch_size, -1).transpose(1, 2)[:self.n_dec_layer].contiguous().view(self.n_dec_layer, batch_size, -1)

        os = embedded_s.new_zeros(batch_size, self.max_seq_len + 1, self.vocab_size)
        embedded_t = self.target_embed(target).transpose(0, 1).unsqueeze(2)  # (max_seq_len + 1, batch, 1, d_emb)
        for i in range(self.max_seq_len + 1):
            if self.attention:
                # TODO: calculate attention
                pass
            dec_out, h = self.decoder(embedded_t[i], target_mask[:, i], h)
            os[:, i, :] += self.w(dec_out.squeeze(1))  # (batch, vocab_size)
        return os

    def predict(self,
                source: torch.Tensor,
                source_mask: torch.Tensor,
                ):
        self.eval()
        with torch.no_grad():
            batch_size = source.size(0)
            embedded_s = self.source_embed(source)  # (batch, max_seq_len, d_emb)
            enc_out, h = self.encoder(embedded_s, source_mask, None)
            if self.bi_directional:
                # (n_layer * n_direction, batch, d_hidden) -> (n_layer, batch, d_hidden * n_direction)
                h[0] = h[0].contiguous().view(self.n_enc_layer, self.n_direction, batch_size, -1).transpose(1, 2)[
                       :self.n_dec_layer].contiguous().view(self.n_dec_layer, batch_size, -1)
                h[1] = h[1].contiguous().view(self.n_enc_layer, self.n_direction, batch_size, -1).transpose(1, 2)[
                       :self.n_dec_layer].contiguous().view(self.n_dec_layer, batch_size, -1)

            tensor_type = 'torch.cuda.LongTensor' if source.device.index is not None else 'torch.LongTensor'
            target = torch.full((batch_size, 1), EOS).type(tensor_type).to(source.device)
            target_mask = torch.full((batch_size, 1), 1).type(tensor_type).squeeze(-1).to(source.device)
            dec_out = self.target_embed(target)
            os = []
            for i in range(self.max_seq_len + 1):
                if self.attention:
                    pass
                dec_out, h = self.decoder(dec_out, target_mask, h)
                prediction = torch.argmax(self.w(dec_out.squeeze(1)), dim=1) + 1  # (batch, vocab_size)
                target_mask = prediction.ne(EOS)
                os.append(prediction)
        return os
