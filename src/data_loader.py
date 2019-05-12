from operator import itemgetter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from constants import PAD, BOS, EOS, UNK


class Seq2seqDataset(Dataset):
    def __init__(self,
                 path: str,
                 source_word_to_id: Dict[str, int],
                 target_word_to_id: Dict[str, int]
                 ) -> None:
        self.source_word_to_id = source_word_to_id
        self.target_word_to_id = target_word_to_id
        self.sources, self.targets = self._load(path)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self,
                    idx: int
                    ) -> Tuple[List, List, List, List, List]:
        source = self.sources[idx]
        source_mask = [1] * len(source)
        target = self.targets[idx]
        target_input, target_output = target[0], target[1]
        target_mask = [1] * len(target[0])
        return source, source_mask, target_input, target_output, target_mask

    def _load(self,
              path: str,
              delimiter: str = '\t'
              ) -> Tuple[List, List]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                former, latter = line.strip().split(delimiter)
                source_ids: List[int] = []
                target_inp_ids: List[int] = []
                target_out_ids: List[int] = []
                for mrph in former.split():
                    if mrph in self.source_word_to_id.keys():
                        source_ids.append(self.source_word_to_id[mrph])
                    else:
                        source_ids.append(UNK)
                sources.append(source_ids)

                target_inp_ids.append(BOS)
                for mrph in latter.split():
                    if mrph in self.target_word_to_id.keys():
                        target_inp_ids.append(self.target_word_to_id[mrph])
                        target_out_ids.append(self.target_word_to_id[mrph])
                    else:
                        target_inp_ids.append(UNK)
                        target_out_ids.append(UNK)
                target_out_ids.append(EOS)
                targets.append([target_inp_ids, target_out_ids])
        return sources, targets


class Seq2seqDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 source_word_to_id: Dict[str, int],
                 target_word_to_id: Dict[str, int],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int
                 ) -> None:
        self.dataset = Seq2seqDataset(path, source_word_to_id, target_word_to_id)
        self.n_samples = len(self.dataset)
        super(Seq2seqDataLoader, self).__init__(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                collate_fn=seq2seq_collate_fn)


def seq2seq_collate_fn(batch: List[Tuple]
                       ) -> Tuple[torch.LongTensor, torch.LongTensor,
                                  torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    sources, source_masks, target_inputs, target_outputs, target_masks = [], [], [], [], []
    cache = [(len(sample[0]), len(sample[2])) for sample in batch]
    max_source_length = max(cache, key=itemgetter(0))[0]
    max_target_length = max(cache, key=itemgetter(1))[1]
    for sample in batch:
        source, source_mask, target_input, target_output, target_mask = sample
        source_length, target_length = len(source), len(target_input)

        source_padding = [PAD] * (max_source_length - source_length)
        sources.append(source + source_padding)
        source_mask_padding = [0] * (max_source_length - source_length)
        source_masks.append(source_mask + source_mask_padding)

        target_padding = [PAD] * (max_target_length - target_length)
        target_inputs.append(target_input+target_padding)
        target_outputs.append(target_output+target_padding)
        target_mask_padding = [0] * (max_target_length - target_length)
        target_masks.append(target_mask + target_mask_padding)
    return torch.LongTensor(sources), torch.LongTensor(source_masks), \
        torch.LongTensor(target_inputs), torch.LongTensor(target_outputs), torch.LongTensor(target_masks)
