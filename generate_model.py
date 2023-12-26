import torch
import torch.nn as nn
import torch.nn.functional as F
embedding_dim=64

class Generate_model_transformer(nn.Module):
    def __init__(self,volcabulary_size) -> None:
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=volcabulary_size,embedding_dim=embedding_dim)
        self.transformer=nn.Transformer(d_model=embedding_dim,batch_first=True)
        
    def forward():
        pass

hidden_size=128
num_layers=1
class Generate_model_lstm(nn.Module):
    def __init__(self, volcabulary_size) -> None:
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=volcabulary_size,embedding_dim=embedding_dim)
        self.lstm=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers =num_layers ,batch_first=True)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, volcabulary_size)
    def forward(self, word_ids, lstm_hidden=None)->(torch.Tensor,torch.Tensor):
        embedded = self.embedding(word_ids)
        lstm_out, lstm_hidden = self.lstm(embedded, lstm_hidden)
        out = self.h2h(lstm_out)
        out = self.h2o(out)
        return out, lstm_hidden


