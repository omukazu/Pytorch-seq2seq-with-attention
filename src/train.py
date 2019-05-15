import json
import os
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

from numpy.random import randint
import torch

from utils import load_setting, calculate_loss, translate


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

    best_acc = 0
    n_pred = 10

    for epoch in range(1, config['arguments']['epoch'] + 1):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        for batch_idx, (source, source_mask, target_inputs, target_outputs, target_mask) \
                in enumerate(train_data_loader):
            source = source.to(device)
            source_mask = source_mask.to(device)
            target = target_inputs.to(device)
            target_mask = target_mask.to(device)
            label = target_outputs.to(device)

            # Forward pass
            output = model(source, source_mask, target, target_mask)
            loss = calculate_loss(output, target_mask, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        else:
            print(f'train_loss={total_loss / (batch_idx + 1):.3f}')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            # num_iter = 0
            for batch_idx, (source, source_mask, target_inputs, target_outputs, target_mask) \
                    in enumerate(valid_data_loader):
                source = source.to(device)
                source_mask = source_mask.to(device)
                target = target_inputs.to(device)
                target_mask = target_mask.to(device)
                label = target_outputs.to(device)

                output = model(source, source_mask, target, target_mask)

                total_loss += calculate_loss(output, target_mask, label)
                # num_iter = batch_idx + 1
            else:
                print(f'valid_loss={total_loss / (batch_idx + 1):.3f}')
                predict = model.predict(source, source_mask)  # (b, max_seq_len)
                random_indices = randint(0, len(predict), n_pred)
                s_translation = translate(source[random_indices], source_id_to_word, is_target=False)
                t_translation = translate(target[random_indices], target_id_to_word, is_target=True)
                p_translation = translate(predict[random_indices], target_id_to_word, is_target=True)
                for s, t, p in zip(s_translation, t_translation, p_translation):
                    print(f'source:{" ".join(s)} / target:{" ".join(t)} / predict:{" ".join(p)}')

    # TODO: add metrics
    torch.save(model.state_dict(), os.path.join(config['arguments']['save_path'], f'sample.pth'))


if __name__ == '__main__':
    main()
