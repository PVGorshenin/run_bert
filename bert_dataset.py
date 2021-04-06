import torch
from torch.utils.data import Dataset

class BertDatset(Dataset):

    def __init__(self, df, tokenizer, max_seq_len=100, text_columns='text', label_column=None,
                 do_lower_case=True):
        self.df = df
        self.tokenizer = tokenizer
        self.label_column = label_column
        self.max_seq_len = max_seq_len
        self.text_columns = text_columns
        self.do_lower_case = do_lower_case

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, iloc):
        text = self.df[self.text_columns[0]].iloc[iloc]
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_len,
            pad_to_max_length=True
        )
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
                'label': torch.tensor(label, dtype=torch.float),
            }