import torch.nn as nn


class MyBertModel(nn.Module):
    def __init__(self, bert_model, device, n_classes=30, hidden_dim=768, dropout=.2):
        super(MyBertModel, self).__init__()
        self.bert_model = bert_model.to(device)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, token_type_ids, attention_mask):
        layers, pool_out = self.bert_model(input_ids=ids,
                                           token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)
        logits = self.sigmoid(self.fc1(pool_out))
        return logits
