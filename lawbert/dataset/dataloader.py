from torch import dtype
from lawbert.dataset import tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
import os
import logging
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from torch.nn.utils.rnn import pad_sequence


class LineByLineDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int
    ) -> None:
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if len(line) > 0 and not line.isspace()
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
class DataCollatorForLanguageModeling(object):
    """
        - collate batches of tensors, with pad_token
        - preprocesses batches for masked modeling 
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            return {"input_ids": batch, "labels": batch}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        ## Check for same length tensors
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

    def mask_tokens(self, inputs: torch.Tensor) -> Any:
        """ 
            Prepare masked tokens inputs/labels for masked language modeling: 80% mask, 10% random, 10% original 

        Args:
            self ([type]): [description]
            torch ([type]): [description]
        """
        labels = inputs.clone()  ## THis methods gets recorded ion the gradients
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]

        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only comput loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        ## 10 % of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        ## REst 10% time no change
        return inputs, labels
