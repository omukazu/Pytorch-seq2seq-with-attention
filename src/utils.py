import os
import sys
from typing import Any, Dict, List, Tuple

import numpy
from gensim.models import KeyedVectors
import torch
import torch.nn.functional as F

from data_loader import EPDataLoader
from seq2seq import Seq2seq
from constants import PAD, UNK, EOS


def calculate_loss(output: torch.Tensor,  # (batch, max_seq_len + 1, vocab_size)
                   truth: torch.Tensor,   # (batch, max_seq_len + 1)
                   loss_function: torch.nn.Module
                   ) -> torch.Tensor:
    vocab_size = output.size(2)
    prediction = F.softmax(output, dim=1).contiguous().view(-1, vocab_size)
    truth = truth.view(-1)
    loss = loss_function(prediction, truth)
    return loss


def load_vocabulary(path: str
                    ) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r") as f:
        lines = [line for line in f]
    word_to_id = {f'{key.strip()}': i + 3 for i, key in enumerate(lines)}
    word_to_id['<PAD>'] = PAD
    word_to_id['<UNK>'] = UNK
    word_to_id['<EOS>'] = EOS
    id_to_word = {i + 3: f'{key.strip()}' for i, key in enumerate(lines)}
    id_to_word[PAD] = '<PAD>'
    id_to_word[UNK] = '<UNK>'
    id_to_word[EOS] = '<EOS>'
    return word_to_id, id_to_word


def ids_to_embeddings(word_to_id: Dict[str, int],
                      w2v: KeyedVectors
                      ) -> torch.Tensor:
    embeddings = numpy.zeros((len(word_to_id), w2v.vector_size), 'f')  # (vocab_size, d_emb)
    for w, i in word_to_id.items():
        if w == '<PAD>':
            pass  # zero vector
        elif w in w2v.vocab:
            embeddings[i] = w2v.word_vec(w)
        else:
            embeddings[i] = w2v.word_vec('<UNK>')
    return torch.tensor(embeddings)


def load_setting(config: Dict[str, Dict[str, str or int]],
                 args  # argparse.Namespace
                 ):
    torch.manual_seed(config['arguments']['seed'])

    path = 'debug' if args.debug else 'data'
    word_to_id, id_to_word = load_vocabulary(config[path]['vocabulary'])
    w2v = KeyedVectors.load_word2vec_format(config[path]['w2v'], binary=True, unicode_errors='ignore')
    embeddings = ids_to_embeddings(word_to_id, w2v)

    if config['arguments']['model_name'] == 'Seq2seq':
        model = Seq2seq(d_hidden=config['arguments']['d_hidden'],
                        embeddings=embeddings,
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
    train_data_loader = EPDataLoader(config[path]['train'], word_to_id, config['arguments']['max_seq_len'],
                                     batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)
    valid_data_loader = EPDataLoader(config[path]['valid'], word_to_id, config['arguments']['max_seq_len'],
                                     batch_size=config['arguments']['batch_size'], shuffle=False, num_workers=2)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['arguments']['learning_rate'])

    return id_to_word, model, device, train_data_loader, valid_data_loader, optimizer


def load_tester(config: Dict[str, Dict[str, str or int]],
                args  # argparse.Namespace
                ):
    # build model architecture first
    if config['arguments']['model_name'] == 'Seq2seq':
        model = Seq2seq(d_hidden=config['arguments']['d_hidden'],
                        embeddings=None,
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

    # setup data_loader instances
    path = 'debug' if args.debug else 'evpairs'
    word_to_id, id_to_word = load_vocabulary(config[path]['vocabulary'])

    test_data_loader = EPDataLoader(config[path]['test'], word_to_id, config['arguments']['max_seq_len'],
                                    batch_size=config['arguments']['batch_size'], shuffle=True, num_workers=2)

    # build optimizer
    return id_to_word, model, device, test_data_loader


def translate(predictions: torch.Tensor,
              id_to_word: Dict[int, str]
              ) -> List[List[Any]]:
    return [[id_to_word[int(p)] for p in prediction if int(p) not in {PAD, EOS}] for prediction in predictions]
