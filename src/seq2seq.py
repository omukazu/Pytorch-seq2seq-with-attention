import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import UNK, BOS, EOS
from model_components import Encoder, Decoder, Embedder, Maxout


class Seq2seq(nn.Module):
    def __init__(self,
                 d_hid: int,
                 source_embeddings: torch.Tensor,
                 target_embeddings: torch.Tensor,
                 max_seq_len: int,
                 attention: bool = False,
                 bi_directional: bool = True,
                 dropout_rate: float = 0.333,
                 n_layer: int = 2):
        super(Seq2seq, self).__init__()
        self.max_seq_len = max_seq_len

        self.source_vocab_size, self.d_s_emb = source_embeddings.size()
        self.target_vocab_size, self.d_t_emb = target_embeddings.size()
        self.vocab_size = target_embeddings.size(0)
        self.source_embed = Embedder(self.source_vocab_size, self.d_s_emb)
        self.source_embed.set_initial_embedding(source_embeddings, freeze=True)
        self.target_embed = Embedder(self.target_vocab_size, self.d_t_emb)
        self.target_embed.set_initial_embedding(target_embeddings, freeze=True)

        self.n_e_lay = n_layer
        self.bi_directional = bi_directional
        self.n_dir = 2 if bi_directional else 1
        self.attention = attention
        self.d_d_hid = d_hid * self.n_dir
        self.n_d_lay = 1
        self.d_out = self.d_d_hid // self.n_dir

        self.encoder = Encoder(nn.LSTM(input_size=self.d_s_emb, hidden_size=d_hid,
                                       num_layers=self.n_e_lay, batch_first=True,
                                       dropout=dropout_rate, bidirectional=bi_directional))
        self.decoder = Decoder(nn.LSTM(input_size=self.d_t_emb, hidden_size=self.d_d_hid,
                                       num_layers=self.n_d_lay, batch_first=True,
                                       dropout=dropout_rate, bidirectional=False))
        self.maxout = Maxout(self.d_d_hid, self.d_out, self.n_dir)
        self.w = nn.Linear(self.d_out, self.vocab_size)

    def forward(self,
                source: torch.Tensor,       # (b, max_sou_len, d_emb)
                source_mask: torch.Tensor,  # (b, max_sou_len)
                target: torch.Tensor,       # (b, max_tar_len, 1), word_ids
                target_mask: torch.Tensor   # (b, max_tar_len)
                ) -> torch.Tensor:          # (b, max_tar_len, d_emb)
        b = source.size(0)
        source_embedded = self.source_embed(source, source_mask, False)  # (b, max_sou_len, d_emb)
        e_out, states = self.encoder(source_embedded, source_mask)
        # (n_e_lay * n_dir, b, d_hid) -> (n_d_lay, b, d_hid)
        states = (self.transform(b, states[0]),
                  self.transform(b, states[1].new_zeros(states[1].size())))

        max_tar_len = target.size(1)
        output = source_embedded.new_zeros((b, max_tar_len, self.vocab_size))
        target_embedded = self.target_embed(target, target_mask, False)  # (b, max_tar_len, d_emb)
        target_embedded = target_embedded.transpose(1, 0).unsqueeze(2)   # (max_tar_len, b, 1, d_emb)
        # decode per word
        for i in range(max_tar_len):
            if self.attention:
                pass  # TODO: calculate attention
            d_out, h = self.decoder(target_embedded[i], target_mask[:, i], states)
            output[:, i, :] = self.w(self.maxout(d_out.squeeze(1)))      # (b, vocab_size)
        return output

    def predict(self,
                source: torch.Tensor,       # (b, max_seq_len, d_emb)
                source_mask: torch.Tensor,  # (b, max_seq_len)
                ) -> torch.Tensor:          # (b, seq_len)
        self.eval()
        with torch.no_grad():
            b = source.size(0)
            source_embedded = self.source_embed(source, source_mask, False)  # (b, max_seq_len, d_emb)
            e_out, states = self.encoder(source_embedded, source_mask)
            states = (self.transform(b, states[0]),
                      self.transform(b, states[1].new_zeros(states[1].size())))

            target_id = torch.full((b, 1), BOS, dtype=source.dtype).to(source.device)
            target_mask = torch.full([b], 1, dtype=source_mask.dtype).to(source_mask.device)
            predictions = source_embedded.new_zeros(b, self.max_seq_len, 1)
            for i in range(self.max_seq_len):
                if self.attention:
                    pass
                target_embedded = self.target_embed(target_id, target_mask, True)
                d_out, states = self.decoder(target_embedded, target_mask, states)

                output = self.w(self.maxout(d_out.squeeze(1)))              # (b, vocab_size)
                output[:, UNK] -= 1e6                                       # mask unknown
                if i == 0:
                    output[:, EOS] -= 1e6
                prediction = torch.argmax(F.softmax(output, dim=1), dim=1)  # (b), greedy
                target_mask = target_mask * prediction.ne(EOS).type(target_mask.dtype)
                target_id = prediction.unsqueeze(1)
                predictions[:, i, :] = prediction.unsqueeze(1)
        return predictions

    def transform(self,
                  b: int,
                  state: torch.Tensor
                  ) -> torch.Tensor:
        state = state.contiguous().view(self.n_e_lay, self.n_dir, b, -1)
        state = state.permute(0, 2, 3, 1)                        # (n_e_lay, b, d_e_hid, self.n_dir)
        state = state.contiguous().view(self.n_e_lay, b, -1)  # (n_e_lay, b, d_e_hid * self.n_dir)
        # hidden_state[: n_layer] <- extract n-last hidden layer
        state = state[:self.n_d_lay]   # (n_d_lay, b, d_e_hid * self.n_dir)
        return state
