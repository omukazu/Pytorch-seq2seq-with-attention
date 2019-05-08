import json
import os
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from utils import calculate_loss, load_setting, translate


def main():
    parser = ArgumentParser(description='train a classifier', formatter_class=RawTextHelpFormatter)
    parser.add_argument('CONFIG', default=None, type=str, help='path to config file')
    parser.add_argument('--gpu', '-g', default=None, type=str, help='gpu numbers\nto specify')
    parser.add_argument('--debug', default=False, action='store_true', help='switch to debug mode')
    args = parser.parse_args()

    with open(args.CONFIG, "r") as f:
        config = json.load(f)

    os.makedirs(os.path.dirname(config['arguments']['save_path']), exist_ok=True)

    s_id_to_word, t_id_to_word, model, device, train_data_loader, valid_data_loader, optimizer = \
        load_setting(config, args)

    best_acc = 0
    place_holder = CrossEntropyLoss(reduction='mean')

    for epoch in range(1, config['arguments']['epoch'] + 1):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        for batch_idx, (source, source_mask, targets, target_mask) in tqdm(enumerate(train_data_loader)):
            source = source.to(device)
            source_mask = source_mask.to(device)
            target = targets[0].to(device)
            target_mask = target_mask.to(device)
            label = targets[1].to(device)

            # Forward pass
            output = model(source, source_mask, target, target_mask)
            loss = calculate_loss(output, target_mask, label, place_holder)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        else:
            predict = model.predict(source, source_mask)  # (batch, max_seq_len)
            s_translation = translate(source[:5], s_id_to_word)
            t_translation = translate(targets[1][:5], t_id_to_word)
            p_translation = translate(predict[:5], t_id_to_word)
            for p in p_translation:
                print(' '.join(p))
        print(f'train_loss={total_loss / train_data_loader.n_samples:.6f}', end=' ')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            # num_iter = 0
            for batch_idx, (source, source_mask, targets, target_mask) in tqdm(enumerate(valid_data_loader)):
                source = source.to(device)
                source_mask = source_mask.to(device)
                target = targets[0].to(device)
                target_mask = target_mask.to(device)
                label = targets[1].to(device)

                output = model(source, source_mask, target, target_mask)
                # for s, t, p in zip(s_translation, t_translation, p_translation):
                #     print(f'source:{" ".join(s)} / target:{" ".join(t)} / predict:{" ".join(p)}')

                total_loss += calculate_loss(output, target_mask, label, place_holder)
                # num_iter = batch_idx + 1
        print(f'valid_loss={total_loss / valid_data_loader.n_samples:.6f}', end=' ')
        # if valid_acc > best_acc:
    torch.save(model.state_dict(),
               os.path.join(config['arguments']['save_path'], f'sample.pth'))
        # best_acc = valid_acc


if __name__ == '__main__':
    main()
