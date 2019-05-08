import os
import sys
from typing import Any, Dict, List, Tuple

import numpy
from gensim.models import KeyedVectors
import torch
import torch.nn.functional as F

from data_loader import EPDataLoader
from seq2seq import Seq2seq
from constants import BOS, EOS, UNK


def calculate_loss(output: torch.Tensor,        # (batch, max_target_len, vocab_size)
                   target_mask: torch.Tensor,   # (batch, max_target_len)
                   label: torch.Tensor,         # (batch, max_target_len)
                   loss_function: torch.nn.Module
                   ) -> torch.Tensor:
    batch, max_target_len, vocab_size = output.size()
    label = label.masked_select(target_mask.eq(1))

    prediction = F.softmax(output, dim=1)  # (batch, max_target_len, vocab_size)
    prediction_mask = target_mask.unsqueeze(-1).expand(-1, -1, vocab_size)
    prediction = prediction.masked_select(prediction_mask.eq(1)).contiguous().view(-1, vocab_size)
    loss = loss_function(prediction, label)
    return loss


def load_vocabulary(source_path: str,
                    target_path: str
                    ) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    with open(source_path, "r") as source, open(target_path, "r") as target:
        s_lines = [line for line in source]
        t_lines = [line for line in target]
    s_word_to_id = {f'{key.strip()}': i + 1 for i, key in enumerate(s_lines)}
    s_word_to_id['<UNK>'] = UNK
    s_id_to_word = {i + 1: f'{key.strip()}' for i, key in enumerate(s_lines)}
    s_id_to_word[UNK] = '<UNK>'
    t_word_to_id = {f'{key.strip()}': i + 3 for i, key in enumerate(t_lines)}
    t_word_to_id['<UNK>'] = UNK
    t_word_to_id['<BOS>'] = BOS
    t_word_to_id['<EOS>'] = EOS
    t_id_to_word = {i + 2: f'{key.strip()}' for i, key in enumerate(t_lines)}
    t_id_to_word[UNK] = '<UNK>'
    t_id_to_word[BOS] = '<BOS>'
    t_id_to_word[EOS] = '<EOS>'
    return s_word_to_id, s_id_to_word, t_word_to_id, t_id_to_word


def ids_to_embeddings(word_to_id: Dict[str, int],
                      w2v: KeyedVectors
                      ) -> torch.Tensor:
    embeddings = numpy.zeros((len(word_to_id) + 1, w2v.vector_size), 'f')  # (vocab_size + 1, d_emb)
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
    s_word_to_id, s_id_to_word, t_word_to_id, t_id_to_word \
        = load_vocabulary(config[path]['s_vocab'], config[path]['t_vocab'])
    w2v = KeyedVectors.load_word2vec_format(config[path]['w2v'], binary=True, unicode_errors='ignore')
    source_embeddings = ids_to_embeddings(s_word_to_id, w2v)
    target_embeddings = ids_to_embeddings(t_word_to_id, w2v)

    if config['arguments']['model_name'] == 'Seq2seq':
        model = Seq2seq(d_hidden=config['arguments']['d_hidden'],
                        source_embeddings=source_embeddings,
                        target_embeddings=target_embeddings,
                        max_seq_len=config['arguments']['max_seq_len'])
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
    train_data_loader = EPDataLoader(config[path]['train'], s_word_to_id, t_word_to_id,
                                     batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)
    valid_data_loader = EPDataLoader(config[path]['valid'], s_word_to_id, t_word_to_id,
                                     batch_size=config['arguments']['batch_size'], shuffle=False, num_workers=2)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['arguments']['learning_rate'])

    return s_id_to_word, t_id_to_word, model, device, train_data_loader, valid_data_loader, optimizer


def load_tester(config: Dict[str, Dict[str, str or int]],
                args  # argparse.Namespace
                ):
    path = 'debug' if args.debug else 'data'
    s_word_to_id, s_id_to_word, t_word_to_id, t_id_to_word \
        = load_vocabulary(config[path]['s_vocab'], config[path]['t_vocab'])
    w2v = KeyedVectors.load_word2vec_format(config[path]['w2v'], binary=True, unicode_errors='ignore')
    source_embeddings = ids_to_embeddings(s_word_to_id, w2v)
    target_embeddings = ids_to_embeddings(t_word_to_id, w2v)

    # build model architecture first
    if config['arguments']['model_name'] == 'Seq2seq':
        model = Seq2seq(d_hidden=config['arguments']['d_hidden'],
                        source_embeddings=source_embeddings,
                        target_embeddings=target_embeddings,
                        max_seq_len=config['arguments']['max_seq_len'])
    else:
        print(f'Unknown model name: {config["arguments"]["model_name"]}', file=sys.stderr)
        return

    # setup device
    if args.gpu and torch.cuda.is_available():
        assert all([int(gpu_number) >= 0 for gpu_number in args.gpu.split(',')]), 'invalid input'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if len(args.gpu) > 1:
            ids = list(map(int, args.gpu.split(',')))
            device = torch.device('cuda')
            model = torch.nn.DataParallel(model, device_ids=ids)
        else:
            device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    # load state dict
    state_dict = torch.load(config['arguments']['load_path'], map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)

    test_data_loader = EPDataLoader(config[path]['test'], s_word_to_id, config['arguments']['max_seq_len'],
                                    batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)

    # build optimizer
    return t_id_to_word, model, device, test_data_loader


def translate(predictions: torch.Tensor,
              id_to_word: Dict[int, str]
              ) -> List[List[Any]]:
    return [[id_to_word[int(p)] for p in prediction if int(p) not in {BOS, EOS}]
            for prediction in predictions]
