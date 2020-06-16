from torch import dtype
from lawbert.lawbert.dataset import tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
import os
import logging
import torch
from dataclasses import dataclass
from typing import List, Dict 


class LineByLineDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int
    ) -> None:
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if len(line) > 0 and line.isspace()
            ]

        batch_encoding = tokenizer.batch_encode_plus(
            lines, add_special_tokens=True, max_length=block_size
        )
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.examples[index], dtype=torch.float)




@dataclass
class DataCollatorForLanguageModeling:
    """
        - collate batches of tensors, with pad_token
        - preprocesses batches for masked modeling 
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True 
    mlm_probability: float = 0.15

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
