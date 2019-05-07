from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        sorted_lengths, perm_indices = (mask.cumsum(dim=1)[:, -1]).sort(0, descending=True)
        _, unperm_indices = perm_indices.sort(0)

        # masking
        packed = pack_padded_sequence(x[perm_indices], lengths=sorted_lengths, batch_first=True)
        output, h = self.rnn(packed, h)  # (sum(lengths), hid*2)
        unpacked, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)
        # (batch, max_seq_len, d_hidden * 2), (n_layer * n_direction, batch, d_hidden * 2)
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
        valid_len = mask.sum()

        if valid_len > 0:
            batch_size = x.size(0)
            sorted_lengths, perm_indices = mask.sort(0, descending=True)
            _, unperm_indices = perm_indices.sort(0)

            packed = pack_padded_sequence(x[perm_indices][:valid_len],
                                          lengths=sorted_lengths[:valid_len], batch_first=True)
            old_hs = [h.index_select(1, perm_indices)[:, valid_len:, :].contiguous() for h in hs]
            hs = [h.index_select(1, perm_indices)[:, :valid_len, :].contiguous() for h in hs]
            output, hs = self.rnn(packed, hs)
            unpacked, _ = pad_packed_sequence(output, batch_first=True)

            if batch_size > valid_len:
                _, _, d_out = x.size()
                zeros = unpacked.new_zeros(batch_size - valid_len, 1, d_out)
                unpacked = torch.cat((unpacked, zeros), dim=0)  # (valid_len, 1, d_out) -> (batch, 1, d_out)
                x = self.dropout(unpacked[unperm_indices])
                hs = tuple([torch.cat((h, old_h), dim=1) for h, old_h in zip(hs, old_hs)])
        return x, hs
