from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader

PAD = 0
EOS = 2


class EPDataset(Dataset):
    def __init__(self,
                 path: str,
                 word_to_id: Dict[str, int],
                 max_seq_len: Optional[int],
                 ) -> None:
        self.word_to_id = word_to_id
        self.max_seq_len = max_seq_len
        self.sources, self.targets = self._load(path)
        self.max_phrase_len: int = self.max_seq_len if self.max_seq_len is not None \
            else max(len(phrase) for phrase in self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self,
                    idx
                    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        source_len = len(self.sources[idx])
        target_len = len(self.targets[idx][0])
        source_pad: List[int] = [PAD] * (self.max_seq_len - source_len)
        target_pad: List[int] = [PAD] * (self.max_seq_len + 1 - target_len)
        source = np.array(self.sources[idx] + source_pad)
        target_inp = np.array(self.targets[idx][0] + target_pad)
        target_out = np.array(self.targets[idx][1] + target_pad)
        targets = (target_inp, target_out)
        mask_xs = np.array([1] * source_len + [0] * (self.max_seq_len - source_len))
        mask_ys = np.array([1] * target_len + [0] * (self.max_seq_len + 1 - target_len))
        return source, mask_xs, targets, mask_ys

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
                    if mrph in self.word_to_id.keys():
                        source_ids.append(self.word_to_id[mrph])
                    else:
                        source_ids.append(self.word_to_id['<UNK>'])
                if self.max_seq_len is not None and len(source_ids) > self.max_seq_len:
                    source_ids = source_ids[:self.max_seq_len]      # limit sequence length from end of a sentence
                sources.append(source_ids)

                target_inp_ids.append(EOS)
                for mrph in latter.split():
                    if mrph in self.word_to_id.keys():
                        target_inp_ids.append(self.word_to_id[mrph])
                        target_out_ids.append(self.word_to_id[mrph])
                    else:
                        target_inp_ids.append(self.word_to_id['<UNK>'])
                        target_out_ids.append(self.word_to_id['<UNK>'])
                target_out_ids.append(EOS)

                if self.max_seq_len is not None and len(target_inp_ids) > self.max_seq_len + 1:
                    target_inp_ids = target_inp_ids[:self.max_seq_len + 1]
                    target_out_ids = target_out_ids[:self.max_seq_len] + [EOS]
                targets.append((target_inp_ids, target_out_ids))
        return sources, targets


class EPDataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 word_to_id: Dict[str, int],
                 max_seq_len: Optional[int],
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int
                 ):
        self.dataset = EPDataset(path, word_to_id, max_seq_len)
        self.n_samples = len(self.dataset)
        super(EPDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
