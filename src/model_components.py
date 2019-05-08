from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TargetEmbedder(nn.Module):
    def __init__(self,
                 target_embeddings: torch.Tensor):
        super(TargetEmbedder, self).__init__()
        self.target_embed = nn.Embedding.from_pretrained(embeddings=target_embeddings, freeze=False)
        self.d_t_emb = target_embeddings.size(1)

    def forward(self,
                target: torch.Tensor,       # (batch, max_target_len)
                target_mask: torch.Tensor,  # (batch, max_target_len)
                predict: bool
                ) -> torch.Tensor:
        target = target * target_mask
        target_embedded = self.target_embed(target)
        size = (-1, self.d_t_emb) if predict else (-1, -1, self.d_t_emb)
        target_mask = target_mask.unsqueeze(-1).expand(size).type(target_embedded.dtype)
        target_embedded = target_embedded * target_mask  # TARGET_PAD -> zero vector
        return target_embedded                           # (batch, max_target_len, d_t_emb)


class Encoder(nn.Module):
    def __init__(self,
                 rnn: nn.Module,
                 dropout_rate: float = 0.333):
        super(Encoder, self).__init__()
        self.rnn = rnn
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,
                source: torch.Tensor,       # (batch, max_source_len, d_emb)
                source_mask: torch.Tensor,  # (batch, max_source_len)
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        lengths = source_mask.cumsum(dim=1)[:, -1]
        sorted_lengths, perm_indices = lengths.sort(0, descending=True)
        sorted_input = source.index_select(0, perm_indices)
        _, unperm_indices = perm_indices.sort(0)

        # masking
        packed = pack_padded_sequence(sorted_input, lengths=sorted_lengths, batch_first=True)
        output, h = self.rnn(packed, None)  # (sum(lengths), hid*2)
        unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)
        restored = unpacked.index_select(0, unperm_indices)
        h = [state.index_select(1, unperm_indices) for state in h]
        # (batch, max_source_len, d_enc_hidden * n_direction), (n_layer * n_direction, batch, d_enc_hidden)
        return restored, h


class Decoder(nn.Module):
    def __init__(self,
                 rnn: nn.Module,
                 dropout_rate: float = 0.333):
        super(Decoder, self).__init__()
        self.rnn = rnn
        self.dropout = nn.Dropout(p=dropout_rate)

    # stateless, decode per word
    def forward(self,
                target_word: torch.Tensor,             # (b, 1, d_emb)
                target_mask: torch.Tensor,             # (b, 1)
                hs: Tuple[torch.Tensor, torch.Tensor]  # ((n_d_lay * n_dir, b, d_hid), (n_d_lay * n_dir, b, d_hid))
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        valid_len = target_mask.sum()

        if valid_len > 0:
            batch = target_word.size(0)
            sorted_lengths, perm_indices = target_mask.sort(0, descending=True)
            sorted_input = target_word.index_select(0, perm_indices)
            _, unperm_indices = perm_indices.sort(0)

            packed = pack_padded_sequence(sorted_input[:valid_len],
                                          lengths=sorted_lengths[:valid_len], batch_first=True)
            old_hs = [h.index_select(1, perm_indices)[:, valid_len:, :].contiguous() for h in hs]
            new_hs = [h.index_select(1, perm_indices)[:, :valid_len, :].contiguous() for h in hs]
            output, new_hs = self.rnn(packed, new_hs)
            unpacked, _ = pad_packed_sequence(output, batch_first=True)  # (valid_len, 1, d_d_hid * 1)

            if valid_len < batch:
                n_dec_lay, _, d_dec_hidden = hs[0].size()
                pad = unpacked.new_zeros(batch - valid_len, 1, d_dec_hidden)
                # (valid_len, 1, n_dir * d_d_hid) -> (b, 1, n_dir * d_d_hid)
                unpacked = torch.cat((unpacked, pad), dim=0)
                new_hs = tuple([torch.cat((nh, oh), dim=1) for nh, oh in zip(new_hs, old_hs)])

            unpacked = unpacked.index_select(0, unperm_indices)
            new_hs = tuple([new_h.index_select(1, unperm_indices) for new_h in new_hs])
        # all words are PAD or EOS
        else:
            _, batch, d_dec_hidden = hs[0].size()
            unpacked = hs[0].transpose(0, 1)
            new_hs = hs
        # (b, 1, n_dir * d_d_hid), ((n_d_lay * n_dir, b, d_hid), (n_d_lay * n_dir, b, d_hid))
        return unpacked, new_hs
