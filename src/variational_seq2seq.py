from typing import Tuple

import torch
from torch.distributions import Normal as Gaussian
import torch.nn as nn
import torch.nn.functional as F

from constants import UNK, BOS, EOS
from model_components import Embedder, Encoder, Decoder, Maxout


class VariationalSeq2seq(nn.Module):
    def __init__(self,
                 d_e_hid: int,
                 max_seq_len: int,
                 source_embeddings: torch.Tensor,  # TODO: Optional embeddings
                 target_embeddings: torch.Tensor,
                 attention: bool = True,
                 bi_directional: bool = True,
                 dropout_rate: float = 0.333,
                 freeze: bool = False,
                 n_e_layer: int = 2,
                 n_d_layer: int = 1
                 ) -> None:
        super(VariationalSeq2seq, self).__init__()
        self.max_seq_len = max_seq_len

        self.source_vocab_size, self.d_s_emb = source_embeddings.size()
        self.target_vocab_size, self.d_t_emb = target_embeddings.size()
        self.source_embed = Embedder(self.source_vocab_size, self.d_s_emb)
        self.source_embed.set_initial_embedding(source_embeddings, freeze=freeze)
        self.target_embed = Embedder(self.target_vocab_size, self.d_t_emb)
        self.target_embed.set_initial_embedding(target_embeddings, freeze=freeze)

        self.n_e_lay = n_e_layer
        self.bi_directional = bi_directional
        self.n_dir = 2 if bi_directional else 1
        self.encoder = Encoder(rnn=nn.LSTM(input_size=self.d_s_emb, hidden_size=d_e_hid,
                                           num_layers=self.n_e_lay, batch_first=True,
                                           dropout=dropout_rate, bidirectional=bi_directional))

        self.z_mu = nn.Linear(d_e_hid * self.n_dir, d_e_hid * self.n_dir)
        self.z_ln_var = nn.Linear(d_e_hid * self.n_dir, d_e_hid * self.n_dir)

        self.attention = attention
        self.d_d_hid = d_e_hid * self.n_dir
        self.n_d_lay = n_d_layer
        assert self.d_d_hid % self.n_dir == 0, 'invalid d_e_hid'
        self.d_c_hid = self.d_d_hid if attention else 0
        self.d_out = (self.d_d_hid + self.d_c_hid) // self.n_dir
        self.decoder = Decoder(rnn=nn.LSTMCell(input_size=self.d_t_emb, hidden_size=self.d_d_hid))

        self.c_tanh = nn.Tanh()
        self.c_linear = nn.Linear(self.d_c_hid, self.d_c_hid)

        self.maxout = Maxout(self.d_d_hid + self.d_c_hid, self.d_out, self.n_dir)
        self.w = nn.Linear(self.d_out, self.target_vocab_size)

    def forward(self,
                source: torch.Tensor,       # (b, max_sou_seq_len)
                source_mask: torch.Tensor,  # (b, max_sou_seq_len)
                target: torch.Tensor,       # (b, max_tar_seq_len)
                target_mask: torch.Tensor,  # (b, max_tar_seq_len)
                label: torch.Tensor,        # (b, max_tar_seq_len)
                annealing: float
                ) -> Tuple[torch.Tensor, Tuple]:                  # (b, max_tar_seq_len, d_emb)
        b = source.size(0)
        source_embedded = self.source_embed(source, source_mask)  # (b, max_sou_seq_len, d_s_emb)
        e_out, (hidden, _) = self.encoder(source_embedded, source_mask)

        h = self.transform(hidden, True)  # (n_e_lay * b, d_e_hid * n_dir)
        z_mu = self.z_mu(h)               # (n_e_lay * b, d_e_hid * n_dir)
        z_ln_var = self.z_ln_var(h)       # (n_e_lay * b, d_e_hid * n_dir)
        hidden = Gaussian(z_mu, z_ln_var).rsample()  # reparameterization trick
        # (n_e_lay * b, d_e_hid * n_dir) -> (b, d_e_hid * n_dir), initialize cell state
        states = (self.transform(hidden, False), self.transform(hidden.new_zeros(hidden.size()), False))

        max_tar_seq_len = target.size(1)
        output = source_embedded.new_zeros((b, max_tar_seq_len, self.target_vocab_size))
        target_embedded = self.target_embed(target, target_mask)  # (b, max_tar_seq_len, d_t_emb)
        target_embedded = target_embedded.transpose(1, 0)         # (max_tar_seq_len, b, d_t_emb)
        total_context_loss = 0
        # decode per word
        for i in range(max_tar_seq_len):
            d_out, states = self.decoder(target_embedded[i], target_mask[:, i], states)
            if self.attention:
                context, cs = self.calculate_context_vector(e_out, states[0], source_mask, True)  # (b, d_d_hid)
                total_context_loss += self.calculate_context_loss(cs)
                d_out = torch.cat((d_out, context), dim=-1)                                       # (b, d_d_hid * 2)
            output[:, i, :] = self.w(self.maxout(d_out))  # (b, d_d_hid) -> (b, d_out) -> (b, tar_vocab_size)
        loss, details = self.calculate_loss(output, target_mask, label,
                                            z_mu, z_ln_var, total_context_loss, annealing)
        if torch.isnan(loss).any():
            raise ValueError('nan detected')
        return loss, details

    def predict(self,
                source: torch.Tensor,       # (b, max_sou_seq_len)
                source_mask: torch.Tensor,  # (b, max_sou_seq_len)
                sampling: bool = True
                ) -> torch.Tensor:          # (b, max_seq_len)
        self.eval()
        with torch.no_grad():
            b = source.size(0)
            source_embedded = self.source_embed(source, source_mask)                        # (b, max_seq_len, d_s_emb)
            e_out, (hidden, _) = self.encoder(source_embedded, source_mask)

            h = self.transform(hidden, True)
            z_mu = self.z_mu(h)
            z_ln_var = self.z_ln_var(h)
            hidden = Gaussian(z_mu, z_ln_var).sample() if sampling else z_mu
            states = (self.transform(hidden, False), self.transform(hidden.new_zeros(hidden.size()), False))

            target_id = torch.full((b, 1), BOS, dtype=source.dtype).to(source.device)
            target_mask = torch.full((b, 1), 1, dtype=source_mask.dtype).to(source_mask.device)
            predictions = source_embedded.new_zeros(b, self.max_seq_len, 1)
            for i in range(self.max_seq_len):
                target_embedded = self.target_embed(target_id, target_mask).squeeze(1)      # (b, d_t_emb)
                d_out, states = self.decoder(target_embedded, target_mask[:, 0], states)
                if self.attention:
                    context, _ = self.calculate_context_vector(e_out, states[0], source_mask, False)
                    d_out = torch.cat((d_out, context), dim=-1)

                output = self.w(self.maxout(d_out))                                         # (b, tar_vocab_size)
                output[:, UNK] -= 1e6                                                       # mask <UNK>
                if i == 0:
                    output[:, EOS] -= 1e6                                                   # avoid 0 length output
                prediction = torch.argmax(F.softmax(output, dim=1), dim=1).unsqueeze(1)     # (b, 1), greedy
                target_mask = target_mask * prediction.ne(EOS).type(target_mask.dtype)
                target_id = prediction
                predictions[:, i, :] = prediction
        return predictions

    def transform(self,
                  state: torch.Tensor,  # (n_e_lay * n_dir, b, d_e_hid) or (n_e_lay * b, d_e_hid * n_dir)
                  switch: bool
                  ) -> torch.Tensor:
        if switch:
            b = state.size(1)
            state = state.contiguous().view(self.n_e_lay, self.n_dir, b, -1)
            state = state.permute(0, 2, 3, 1)                      # (n_e_lay, b, d_e_hid, n_dir)
            state = state.contiguous().view(self.n_e_lay * b, -1)  # (n_e_lay * b, d_e_hid * n_dir)
        else:
            b = state.size(0) // self.n_e_lay
            state = state.contiguous().view(self.n_e_lay, b, -1)
            # extract hidden layer
            state = state[0]
        return state

    # variational soft attention, score is calculated by dot product
    def calculate_context_vector(self,
                                 encoder_hidden_states: torch.Tensor,          # (b, max_sou_seq_len, d_e_hid * n_dir)
                                 previous_decoder_hidden_state: torch.Tensor,  # (b, d_d_hid)
                                 source_mask: torch.Tensor,                    # (b, max_sou_seq_len)
                                 is_training: bool
                                 ) -> Tuple[torch.Tensor, Tuple]:
        b, max_sou_seq_len, d_d_hid = encoder_hidden_states.size()
        # (b, max_sou_seq_len, d_d_hid)
        previous_decoder_hidden_states = previous_decoder_hidden_state.unsqueeze(1).expand(b, max_sou_seq_len, d_d_hid)

        alignment_weights = (encoder_hidden_states * previous_decoder_hidden_states).sum(dim=-1)
        alignment_weights.masked_fill_(source_mask.ne(1), -1e6)
        alignment_weights = F.softmax(alignment_weights, dim=-1).unsqueeze(-1)   # (b, max_sou_seq_len, 1)
        context_vector = (alignment_weights * encoder_hidden_states).sum(dim=1)  # (b, d_d_hid)

        c_mu = context_vector
        c_ln_var = (self.c_linear(self.c_tanh(context_vector))).exp()
        context_vector = Gaussian(c_mu, c_ln_var).rsample() if is_training else Gaussian(c_mu, c_ln_var).sample()
        return context_vector, (c_mu, c_ln_var)

    @staticmethod
    def calculate_context_loss(cs: Tuple[torch.Tensor, torch.Tensor]
                               ) -> torch.Tensor:
        c_mu, c_ln_var = cs
        b = c_mu.size(0)
        kl_divergence = (c_mu ** 2 + c_ln_var.exp() - c_ln_var - 1) * 0.5
        context_loss = kl_divergence.sum() / b
        return context_loss

    @staticmethod
    def calculate_loss(output: torch.Tensor,       # (b, max_tar_len, vocab_size)
                       target_mask: torch.Tensor,  # (b, max_tar_len)
                       label: torch.Tensor,        # (b, max_tar_len)
                       mu: torch.Tensor,           # (n_e_lay * b, d_e_hid * n_dir)
                       ln_var: torch.Tensor,       # (n_e_lay * b, d_e_hid * n_dir)
                       total_context_loss: torch.Tensor,
                       annealing: float,
                       gamma: float = 10,
                       ) -> Tuple[torch.Tensor, Tuple]:
        b, max_tar_len, vocab_size = output.size()
        label = label.masked_select(target_mask.eq(1))

        prediction_mask = target_mask.unsqueeze(-1).expand(b, max_tar_len, vocab_size)  # (b, max_tar_len, vocab_size)
        prediction = output.masked_select(prediction_mask.eq(1)).contiguous().view(-1, vocab_size)
        reconstruction_loss = F.cross_entropy(prediction, label, reduction='none').sum() / b
        kl_divergence = (mu ** 2 + ln_var.exp() - ln_var - 1) * 0.5
        regularization_loss = kl_divergence.sum() / b
        return reconstruction_loss + annealing * (regularization_loss + gamma * total_context_loss), \
            (reconstruction_loss, regularization_loss, total_context_loss)
