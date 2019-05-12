import os
import sys
from typing import Dict, List, Tuple

import numpy
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors

from constants import PAD, BOS, EOS, UNK
from data_loader import Seq2seqDataLoader
from seq2seq import Seq2seq


def calculate_loss(output: torch.Tensor,        # (b, max_tar_len, vocab_size)
                   target_mask: torch.Tensor,   # (b, max_tar_len)
                   label: torch.Tensor,         # (b, max_tar_len)
                   ) -> torch.Tensor:
    b, max_tar_len, vocab_size = output.size()
    label = label.masked_select(target_mask.eq(1))

    prediction_mask = target_mask.unsqueeze(-1).expand(b, max_tar_len, vocab_size)  # (b, max_tar_len, vocab_size)
    prediction = output.masked_select(prediction_mask.eq(1)).contiguous().view(-1, vocab_size)
    loss = F.cross_entropy(prediction, label, reduction='none').sum() / b
    return loss


def load_vocabulary(source_path: str,
                    target_path: str
                    ) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    with open(source_path, "r") as source, open(target_path, "r") as target:
        source_lines = [line for line in source]
        target_lines = [line for line in target]

    source_word_to_id = {f'{key.strip()}': i + 1 for i, key in enumerate(source_lines)}
    source_word_to_id['<UNK>'] = UNK
    source_id_to_word = {i + 1: f'{key.strip()}' for i, key in enumerate(source_lines)}
    source_id_to_word[UNK] = '<UNK>'

    target_word_to_id = {f'{key.strip()}': i + 3 for i, key in enumerate(target_lines)}
    target_word_to_id['<UNK>'] = UNK
    target_word_to_id['<BOS>'] = BOS
    target_word_to_id['<EOS>'] = EOS
    target_id_to_word = {i + 3: f'{key.strip()}' for i, key in enumerate(target_lines)}
    target_id_to_word[UNK] = '<UNK>'
    target_id_to_word[BOS] = '<BOS>'
    target_id_to_word[EOS] = '<EOS>'
    return source_word_to_id, source_id_to_word, target_word_to_id, target_id_to_word


def ids_to_embeddings(word_to_id: Dict[str, int],
                      w2v: KeyedVectors
                      ) -> torch.Tensor:
    embeddings = numpy.zeros((len(word_to_id), w2v.vector_size), 'f')  # (vocab_size, d_emb)
    unk_indices = []
    for w, i in word_to_id.items():
        if w in w2v.vocab:
            embeddings[i] = w2v.word_vec(w)
        else:
            unk_indices.append(i)
    if len(unk_indices) > 0:
        embeddings[unk_indices] = numpy.sum(embeddings, axis=0) / (len(word_to_id) - len(unk_indices))
    return torch.tensor(embeddings)


def load_setting(config: Dict[str, Dict[str, str or int]],
                 args  # argparse.Namespace
                 ):
    torch.manual_seed(config['arguments']['seed'])

    path = 'debug' if args.debug else 'data'
    source_word_to_id, source_id_to_word, target_word_to_id, target_id_to_word \
        = load_vocabulary(config[path]['s_vocab'], config[path]['t_vocab'])
    w2v = KeyedVectors.load_word2vec_format(config[path]['w2v'], binary=True, unicode_errors='ignore')
    source_embeddings = ids_to_embeddings(source_word_to_id, w2v)
    target_embeddings = ids_to_embeddings(target_word_to_id, w2v)

    if config['arguments']['model_name'] == 'Seq2seq':
        model = Seq2seq(d_e_hid=config['arguments']['d_hidden'],
                        max_seq_len=config['arguments']['max_seq_len'],
                        source_embeddings=source_embeddings,
                        target_embeddings=target_embeddings)
    else:
        print(f'Unknown model name: {config["arguments"]["model_name"]}', file=sys.stderr)
        return

    # setup device
    if args.gpu and torch.cuda.is_available():
        assert all([int(gpu_number) >= 0 for gpu_number in args.gpu.split(',')]), 'invalid input'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if len(args.gpu) > 1:
            ids = list(map(int, args.gpu.split(',')))
            device = torch.device(f'cuda')
            model = torch.nn.DataParallel(model, device_ids=ids)
        else:
            device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    model.to(device)

    # setup data_loader instances
    train_data_loader = Seq2seqDataLoader(config[path]['train'], source_word_to_id, target_word_to_id,
                                          batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)
    valid_data_loader = Seq2seqDataLoader(config[path]['valid'], source_word_to_id, target_word_to_id,
                                          batch_size=config['arguments']['batch_size'], shuffle=False, num_workers=2)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['arguments']['learning_rate'])

    return source_id_to_word, target_id_to_word, model, device, train_data_loader, valid_data_loader, optimizer


def translate(predictions: torch.Tensor,
              id_to_word: Dict[int, str]
              ) -> List[List[str]]:
    length = predictions.size(0)
    place_holder = [[] for _ in range(length)]
    for index, prediction in enumerate(predictions):
        for p in prediction:
            if int(p) in {PAD, BOS}:
                pass
            else:
                place_holder[index].append(id_to_word[int(p)])
                if int(p) == EOS:
                    break
    return place_holder
