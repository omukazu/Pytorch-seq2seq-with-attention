from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader

from constants import PAD, BOS, EOS, UNK


class EPDataset(Dataset):
    def __init__(self,
                 path: str,
                 s_word_to_id: Dict[str, int],
                 t_word_to_id: Dict[str, int]):
        self.s_word_to_id = s_word_to_id
        self.t_word_to_id = t_word_to_id
        self.sources, self.targets = self._load(path)
        self.max_source_len: int = max(len(phrase) for phrase in self.sources)
        self.max_target_len: int = max(len(phrase[0]) for phrase in self.targets)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self,
                    idx
                    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        source_len = len(self.sources[idx])
        source_pad: List[int] = [PAD] * (self.max_source_len - source_len)
        source = np.array(self.sources[idx] + source_pad)
        source_mask = np.array([1] * source_len + [0] * (self.max_source_len - source_len))

        target_len = len(self.targets[idx][0])
        target_pad: List[int] = [PAD] * (self.max_target_len - target_len)
        target_inp = np.array(self.targets[idx][0] + target_pad)
        target_out = np.array(self.targets[idx][1] + target_pad)
        targets = (target_inp, target_out)
        target_mask = np.array([1] * target_len + [0] * (self.max_target_len - target_len))
        return source, source_mask, targets, target_mask

    def _load(self,
              path: str,
              delimiter: str = '\t'
              ) -> Tuple[List[List[int]], List[Tuple[List[int], List[int]]]]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                former, latter = line.strip().split(delimiter)
                source_ids: List[int] = []
                target_inp_ids: List[int] = []
                target_out_ids: List[int] = []
                for mrph in former.split():
                    if mrph in self.s_word_to_id.keys():
                        source_ids.append(self.s_word_to_id[mrph])
                    else:
                        source_ids.append(UNK)
                sources.append(source_ids)

                target_inp_ids.append(BOS)
                for mrph in latter.split():
                    if mrph in self.t_word_to_id.keys():
                        target_inp_ids.append(self.t_word_to_id[mrph])
                        target_out_ids.append(self.t_word_to_id[mrph])
                    else:
                        target_inp_ids.append(UNK)
                        target_out_ids.append(UNK)
                target_out_ids.append(EOS)
                targets.append((target_inp_ids, target_out_ids))
        return sources, targets


class EPDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 s_word_to_id: Dict[str, int],
                 t_word_to_id: Dict[str, int],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int):
        self.dataset = EPDataset(path, s_word_to_id, t_word_to_id)
        self.n_samples = len(self.dataset)
        super(EPDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
