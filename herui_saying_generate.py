import torch
import torch.utils
import torch.optim as optim
import requests
from torch.utils.data import DataLoader, Dataset,dataloader, random_split

import model

sequence_length=8
stride=3
validation_num=20
batch_size=64



class Hr_dataset(Dataset):
    def __init__(self) -> None:
        string_list=[]
        self.word2index={"<START>":0,"<END>":1}
        self.index2word={}
        self.data=[]
        index_start=len(self.word2index)
        with open("saying.txt") as f:
            hr_list=[i for i in f.read().split("\n") if i!=""]
        for lines in hr_list:
            string_list.append(self.word2index["<START>"])
            for c in lines:
                if c not in self.word2index:
                    self.word2index[c]=index_start
                    index_start+=1
                string_list.append(self.word2index[c])
            string_list.append(self.word2index["<END>"])
        self.index2word={index:word for word,index in self.word2index.items()}
        self.data=torch.tensor(string_list)
        print("Read Done")
    def __len__(self):
        return (len(self.data)-1)//stride
    def __getitem__(self, index) -> (torch.Tensor,torch.Tensor):
        start=index*stride
        end=start+sequence_length
        return (self.data[start:end],self.data[start+1:end+1])

hr_dataset_instance=Hr_dataset()
train_dataset, test_dataset = random_split(
    dataset=hr_dataset_instance,
    lengths=[len(hr_dataset_instance)-validation_num, validation_num],
)

train_dataloder=DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_dataloder=DataLoader(dataset=test_dataset,batch_size=batch_size)
