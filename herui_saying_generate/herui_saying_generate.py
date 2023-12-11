import torch
import torch.utils
import torch.optim as optim
import requests
from torch.utils.data import Dataset,dataloader

import model

class Hr_dataset(Dataset):
    def __init__(self) -> None:
        self.word2index={"<START>":0,"<END>":1}
        self.index2word={}
        self.data=[]
        index_start=len(self.word2index)
        hr_list=requests.get("https://57uu.github.io/herui_saying_text/").text.split("\n")
        for lines in hr_list:
            for c in lines:
                if c not in self.word2index:
                    pass