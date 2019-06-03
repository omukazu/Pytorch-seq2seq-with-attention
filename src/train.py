import json
import os
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import numpy as np
from numpy.random import choice
from progressbar import ProgressBar
import torch

from seq2seq import Seq2seq
from utils import load_setting, sigmoid, translate


def main():
    parser = ArgumentParser(description='train a seq2seq model', formatter_class=RawTextHelpFormatter)
    parser.add_argument('CONFIG', default=None, type=str, help='path to config file')
    parser.add_argument('--gpu', '-g', default=None, type=str, help='gpu numbers\nto specify')
    parser.add_argument('--debug', default=False, action='store_true', help='switch to debug mode')
    args = parser.parse_args()

    with open(args.CONFIG, "r") as f:
        config = json.load(f)

    os.makedirs(os.path.dirname(config['arguments']['save_path']), exist_ok=True)

    source_id_to_word, target_id_to_word, model, device, train_data_loader, valid_data_loader, optimizer = \
        load_setting(config, args)

    n_pred = 5
    n_sample = 1 if model == Seq2seq else 5
    threshold = 10

    bar = ProgressBar(0, len(train_data_loader))
    for epoch in range(1, config['arguments']['epoch'] + 1):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        annealing = sigmoid(epoch - threshold)
        total_loss = 0
        total_rec_loss = 0
        total_reg_loss = 0
        total_c_loss = 0
        for batch_idx, (source, source_mask, target_inputs, target_outputs, target_mask) \
                in enumerate(train_data_loader):
            bar.update(batch_idx)
            source = source.to(device)
            source_mask = source_mask.to(device)
            target = target_inputs.to(device)
            target_mask = target_mask.to(device)
            label = target_outputs.to(device)

            # Forward pass
            loss, details = model(source, source_mask, target, target_mask, label, annealing)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_rec_loss += details[0]
            total_reg_loss += details[1]
            total_c_loss += details[2]
        else:
            print('')
            print(f'train_loss={total_loss / (batch_idx + 1):.3f}'
                  f'/rec:{total_rec_loss / (batch_idx + 1):.3f}'
                  f'/reg:{total_reg_loss / (batch_idx + 1):.3f}'
                  f'/c:{total_c_loss / (batch_idx + 1):.3f}')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_rec_loss = 0
            total_reg_loss = 0
            total_c_loss = 0
            for batch_idx, (source, source_mask, target_inputs, target_outputs, target_mask) \
                    in enumerate(valid_data_loader):
                source = source.to(device)
                source_mask = source_mask.to(device)
                target = target_inputs.to(device)
                target_mask = target_mask.to(device)
                label = target_outputs.to(device)

                loss, details = model(source, source_mask, target, target_mask, label, annealing)
                total_loss += loss
                total_rec_loss += details[0]
                total_reg_loss += details[1]
                total_c_loss += details[2]
            else:
                print(f'valid_loss={total_loss / (batch_idx + 1):.3f}'
                      f'/rec:{total_rec_loss / (batch_idx + 1):.3f}'
                      f'/reg:{total_reg_loss / (batch_idx + 1):.3f}'
                      f'/c:{total_c_loss / (batch_idx + 1):.3f}')
                random_indices = choice(np.arange(len(source)), n_pred, replace=False)
                print(random_indices)
                s_translation = translate(source[random_indices], source_id_to_word, is_target=False)
                t_translation = translate(target[random_indices], target_id_to_word, is_target=True)
                p_translation = \
                    [translate(model.predict(source, source_mask)[random_indices], target_id_to_word,is_target=True)
                     for _ in range(n_sample)]
                p_translation = list(zip(*p_translation))
                for s, t, ps in zip(s_translation, t_translation, p_translation):
                    print(f'source:{" ".join(s)} / target:{" ".join(t)}')
                    for i, p in enumerate(ps):
                        print(f'predict{i+1}:{" ".join(p)}')

    # TODO: add metrics
    torch.save(model.state_dict(), os.path.join(config['arguments']['save_path'], f'sample.pth'))


if __name__ == '__main__':
    main()
