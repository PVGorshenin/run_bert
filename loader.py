from .bert_dataset import BertDatset
from torch.utils.data import DataLoader

def get_dataloader(df, tokenizer, label_column=None, text_columns=['text'],
                   batch_size=4, max_seq_len=200, shuffle=False):

    ds = BertDatset(df=df,
                  tokenizer=tokenizer,
                  max_seq_len=max_seq_len,
                  label_column=label_column,
                  text_columns=text_columns,
                  do_lower_case=False)

    loader = DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle)
    loader.num = df.shape[0]
    return loader