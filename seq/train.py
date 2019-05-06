import json
import os
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from utils import calculate_loss, load_setting, create_save_file_name, create_config


def main():
    parser = ArgumentParser(description='train a classifier', formatter_class=RawTextHelpFormatter)
    parser.add_argument('CONFIG', default=None, type=str, help='path to config file')
    parser.add_argument('--gpu', '-g', default=None, type=str, help='gpu numbers\nto specify')
    parser.add_argument('--debug', default=False, action='store_true', help='switch to debug mode')
    args = parser.parse_args()

    with open(args.CONFIG, "r") as f:
        config = json.load(f)

    os.makedirs(os.path.dirname(config['arguments']['save_path']), exist_ok=True)

    model, device, train_data_loader, valid_data_loader, optimizer = load_setting(config, args)
    # params = model.module.params if len(args.gpu) > 1 else model.params
    # file_name = create_save_file_name(config, params)
    # with open(os.path.join(config['arguments']['save_path'], f'best_{file_name}.config'), "w") as f:
    #     json.dump(create_config(config, params), f, indent=4)

    best_acc = 0
    place_holder = CrossEntropyLoss(ignore_index=-1, reduction='mean')

    for epoch in range(1, config['arguments']['epoch'] + 1):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        for batch_idx, (source, mask_xs, targets, mask_ys) in tqdm(enumerate(train_data_loader)):
            source = source.to(device)
            mask_xs = mask_xs.to(device)
            target = targets[0].to(device)
            truth = (targets[1] - 1).to(device)
            mask_ys = mask_ys.to(device)

            # Forward pass
            output = model(source, mask_xs, target, mask_ys)
            loss = calculate_loss(output, truth, place_holder)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'train_loss={total_loss / train_data_loader.n_samples:.3f}', end=' ')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            # num_iter = 0
            for batch_idx, (source, mask_xs, targets, mask_ys) in tqdm(enumerate(valid_data_loader)):
                source = source.to(device)
                mask_xs = mask_xs.to(device)
                target = targets[0].to(device)
                truth = (targets[1] - 1).to(device)
                mask_ys = mask_ys.to(device)

                output = model(source, mask_xs, target, mask_ys)

                total_loss += calculate_loss(output, truth, place_holder)
                # num_iter = batch_idx + 1
        print(f'valid_loss={total_loss / valid_data_loader.n_samples:.3f}', end=' ')
        # if valid_acc > best_acc:
    torch.save(model.state_dict(),
               os.path.join(config['arguments']['save_path'], f'sample.pth'))
        # best_acc = valid_acc


if __name__ == '__main__':
    main()
