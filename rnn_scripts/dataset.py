import os
import numpy as np
from tqdm import tqdm
from utils.tokenizer import Tokenizer
from torch import is_tensor, from_numpy
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    def __init__(self, config, pattern='pretrain') -> None:
        self.config = config
        self.pattern = pattern
        self.tok = Tokenizer()

        assert self.pattern in ['pretrain', 'finetune']
        if self.pattern == 'pretrain':
            self.smis = self._load(self.config.pretrain_data)
        else:
            self.smis = self._load(self.config.finetune_data)

        self.tokenized_smis = self._tokenize(self.smis)

    def __len__(self):
        return len(self.tokenized_smis)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        tokens = self.tokenized_smis[idx]
        tokens_padded = self._pad(tokens)

        self.X = [self.tok.token_to_idx[token] for token in tokens_padded[:-1]]
        self.y = [self.tok.token_to_idx[token] for token in tokens_padded[1:]]
        
        self.X = from_numpy(np.array(self.X, dtype=np.float32))
        self.y = from_numpy(np.array(self.y, dtype=np.float32))

        sample = {"input_smi":self.X, "label_smi":self.y}

        return sample

    def _load(self, data_path):
        print(f'Loading SMILES from {os.path.basename(data_path)} ...')
        with open(data_path) as f:
            data = [s.rstrip() for s in f]
        print('Loading done.')
        return data

    def _tokenize(self, smis):
        print('Tokenizing SMILES ...')
        tokenized_smis = [self.tok.tokenize(smi) for smi in tqdm(smis)]
        print('Tokenization done.')

        self.max_len = max([len(l) for l in tokenized_smis])
        self.config.tokens_max_len = self.max_len

        return tokenized_smis

    def _pad(self, tokenized_smi):
        return ['G'] + tokenized_smi + ['E'] + [
            'A' for _ in range(self.max_len - len(tokenized_smi))
        ]
    
    def get_config(self):
        return self.config
