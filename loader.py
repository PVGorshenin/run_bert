from .bert_dataset import BertDatset
from torch.utils.data import DataLoader

def get_dataloader(df, tokenizer, label_column='category_1_le', text_columns=['text'], batch_size=4):

    ds = BertDatset(df=df,
                  tokenizer=tokenizer,
                  max_seq_len=512,
                  label_column=label_column,
                  text_columns=text_columns,
                  do_lower_case=False)

    loader = DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True)
    loader.num = df.shape[0]
    return loader