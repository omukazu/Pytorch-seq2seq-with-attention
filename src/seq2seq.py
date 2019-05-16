import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import UNK, BOS, EOS
from model_components import Embedder, Encoder, Decoder, Maxout


class Seq2seq(nn.Module):
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
        super(Seq2seq, self).__init__()
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

        self.attention = attention
        self.d_d_hid = d_e_hid * self.n_dir
        self.n_d_lay = n_d_layer
        assert self.d_d_hid % self.n_dir == 0, 'invalid d_e_hid'
        self.d_c_hid = self.d_d_hid if attention else 0
        self.d_out = (self.d_d_hid + self.d_c_hid) // self.n_dir
        self.decoder = Decoder(rnn=nn.LSTMCell(input_size=self.d_t_emb, hidden_size=self.d_d_hid))

        self.maxout = Maxout(self.d_d_hid + self.d_c_hid, self.d_out, self.n_dir)
        self.w = nn.Linear(self.d_out, self.target_vocab_size)

    def forward(self,
                source: torch.Tensor,       # (b, max_sou_seq_len)
                source_mask: torch.Tensor,  # (b, max_sou_seq_len)
                target: torch.Tensor,       # (b, max_tar_seq_len)
                target_mask: torch.Tensor   # (b, max_tar_seq_len)
                ) -> torch.Tensor:          # (b, max_tar_seq_len, d_emb)
        b = source.size(0)
        source_embedded = self.source_embed(source, source_mask)  # (b, max_sou_seq_len, d_s_emb)
        e_out, states = self.encoder(source_embedded, source_mask)
        if self.attention:
            states = None
        else:
            # (n_e_lay * n_dir, b, d_e_hid) -> (b, d_e_hid * n_dir), initialize cell state
            states = (self.transform(states[0]), self.transform(states[1].new_zeros(states[1].size())))

        max_tar_seq_len = target.size(1)
        output = source_embedded.new_zeros((b, max_tar_seq_len, self.target_vocab_size))
        target_embedded = self.target_embed(target, target_mask)  # (b, max_tar_seq_len, d_t_emb)
        target_embedded = target_embedded.transpose(1, 0)         # (max_tar_seq_len, b, d_t_emb)
        # decode per word
        for i in range(max_tar_seq_len):
            d_out, states = self.decoder(target_embedded[i], target_mask[:, i], states)
            if self.attention:
                context = self.calculate_context_vector(e_out, states[0], source_mask)  # (b, d_d_hid)
                d_out = torch.cat((d_out, context), dim=-1)                             # (b, d_d_hid * 2)
            output[:, i, :] = self.w(self.maxout(d_out))  # (b, d_d_hid) -> (b, d_out) -> (b, tar_vocab_size)
        return output

    def predict(self,
                source: torch.Tensor,       # (b, max_sou_seq_len)
                source_mask: torch.Tensor,  # (b, max_sou_seq_len)
                ) -> torch.Tensor:          # (b, max_seq_len)
        self.eval()
        with torch.no_grad():
            b = source.size(0)
            source_embedded = self.source_embed(source, source_mask)                        # (b, max_seq_len, d_s_emb)
            e_out, states = self.encoder(source_embedded, source_mask)
            states = (self.transform(states[0]), self.transform(states[1].new_zeros(states[1].size())))

            target_id = torch.full((b, 1), BOS, dtype=source.dtype).to(source.device)
            target_mask = torch.full((b, 1), 1, dtype=source_mask.dtype).to(source_mask.device)
            predictions = source_embedded.new_zeros(b, self.max_seq_len, 1)
            for i in range(self.max_seq_len):
                target_embedded = self.target_embed(target_id, target_mask).squeeze(1)      # (b, d_t_emb)
                d_out, states = self.decoder(target_embedded, target_mask[:, 0], states)
                if self.attention:
                    context = self.calculate_context_vector(e_out, states[0], source_mask)  # (b, d_d_hid)
                    d_out = torch.cat((d_out, context), dim=-1)                             # (b, d_d_hid * 2)

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
                  state: torch.Tensor  # (n_e_lay * n_dir, b, d_e_hid)
                  ) -> torch.Tensor:
        b = state.size(1)
        state = state.contiguous().view(self.n_e_lay, self.n_dir, b, -1)
        state = state.permute(0, 2, 3, 1)                     # (n_e_lay, b, d_e_hid, n_dir)
        state = state.contiguous().view(self.n_e_lay, b, -1)  # (n_e_lay, b, d_e_hid * n_dir)
        # extract last hidden layer
        state = state[0]                                      # (b, d_e_hid * n_dir)
        return state

    @staticmethod
    # soft attention, score is calculated by dot product
    def calculate_context_vector(encoder_hidden_states: torch.Tensor,          # (b, max_sou_seq_len, d_e_hid * n_dir)
                                 previous_decoder_hidden_state: torch.Tensor,  # (b, d_d_hid)
                                 source_mask: torch.Tensor                     # (b, max_sou_seq_len)
                                 ) -> torch.Tensor:
        b, max_sou_seq_len, d_d_hid = encoder_hidden_states.size()
        # (b, max_sou_seq_len, d_d_hid)
        previous_decoder_hidden_states = previous_decoder_hidden_state.unsqueeze(1).expand(b, max_sou_seq_len, d_d_hid)

        alignment_weights = (encoder_hidden_states * previous_decoder_hidden_states).sum(dim=-1)
        alignment_weights.masked_fill_(source_mask.ne(1), -1e6)
        alignment_weights = F.softmax(alignment_weights, dim=-1).unsqueeze(-1)   # (b, max_sou_seq_len, 1)

        context_vector = (alignment_weights * encoder_hidden_states).sum(dim=1)  # (b, d_d_hid)
        return context_vector
