import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Union, List


class BertDatset(Dataset):

    def __init__(self, df: pd.DataFrame, tokenizer, max_seq_len: int=100,
                 text_columns: List=['text'], label_column: str=None):
        """
        :param text_columns: если text_columns list, to
        inputs = self.tokenizer.encode_plus(
                 text=text
                 ...) руками делается в
        inputs = self.tokenizer.encode_plus(
                 text=text[0],
                 text_pair=text[1]
                 ...)
        """
        self.df = df
        self.tokenizer = tokenizer
        self.label_column = label_column
        self.max_seq_len = max_seq_len
        self.text_columns = text_columns

    def __len__(self):
        return self.df.shape[0]

    def _get_inputs(self, iloc):
        text = self.df[self.text_columns].iloc[iloc]
        if len(self.text_columns)==1:
            inputs = self.tokenizer.encode_plus(
                text=text[0],
                max_length=self.max_seq_len,
                pad_to_max_length=True,
                truncation=True
            )
            return inputs
        inputs = self.tokenizer.encode_plus(
            text=text[0],
            text_pair=text[1],
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            truncation=True
        )
        return inputs

    def __getitem__(self, iloc):
        inputs = self._get_inputs(iloc)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        if self.label_column is None:
            return {
                'df_ids': torch.tensor(iloc, dtype=torch.long),
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            }
        label = self.df[self.label_column].iloc[iloc]
        return {
                'df_ids': torch.tensor(iloc, dtype=torch.long),
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float32),
            }