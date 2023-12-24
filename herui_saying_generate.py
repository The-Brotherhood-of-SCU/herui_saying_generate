import torch
import torch.utils
import torch.optim as optim
import requests
from torch.utils.data import DataLoader, Dataset,dataloader, random_split
import torch.nn.functional as F

import generate_model

sequence_length=8
stride=3
validation_num=20
batch_size=64
device="cuda"



def get_list(file_name:str)->list:
    with open(file_name,encoding="utf-8") as f:
        return [i for i in f.read().split("\n") if i!=""]

class Hr_dataset(Dataset):
    def __init__(self) -> None:
        string_list=[]
        self.START="<START>"
        self.END="<END>"
        self.word2index={self.START:0,self.END:1}
        self.index2word={}
        self.data=[]
        index_start=len(self.word2index)
        hr_list=get_list("saying.txt")
        hr_list+=get_list("saying_fake.txt")

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
        self.volcabulary_size=len(self.index2word)
        print("Read Done,volcabulary",self.volcabulary_size,"entries",len(self))
    def __len__(self):
        return (len(self.data)-1)//stride
    def __getitem__(self, index) -> (torch.Tensor,torch.Tensor):
        start=index*stride
        end=start+sequence_length
        return (self.data[start:end],self.data[start+1:end+1])


hr_dataset=Hr_dataset()
train_dataset, test_dataset = random_split(
    dataset=hr_dataset,
    lengths=[len(hr_dataset)-validation_num, validation_num],
)

train_dataloder=DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_dataloder=DataLoader(dataset=test_dataset,batch_size=batch_size)

model=generate_model.Generate_model_lstm(volcabulary_size=hr_dataset.volcabulary_size)
model.to(device=device)

def training(optimizer,epoch):
    for j in range(epoch):
        for i, (input_, target) in enumerate(train_dataloder):
            model.train()

            input_, target = input_.to(device), target.to(device)

            output, _ = model(input_)
            loss = F.cross_entropy(output.reshape(-1, hr_dataset.volcabulary_size), target.flatten())

            optimizer.zero_grad()  # Make sure gradient does not accumulate
            loss.backward()  # Compute gradient
            optimizer.step()  # Update NN weights

            print(
                "Training: Epoch=%d, Batch=%d/%d, Loss=%.4f"
                % (j, i, len(train_dataloder), loss.item())
            )

def generate(start_phrases=[],isContinue=lambda: input("Press Enter to Continue")=="",length=30):
    hidden = None
    def next_word(input_word):
        nonlocal hidden

        if(input_word in hr_dataset.word2index):
            input_word_index = hr_dataset.word2index[input_word]
            input_ = torch.Tensor([[input_word_index]])
        else:
            input_=torch.zero_(1,hr_dataset.volcabulary_size)
        input_=input_.long().to(device)
        output, hidden = model(input_, hidden)
        top_word_index = output[0].topk(1).indices.item()
        return hr_dataset.index2word[top_word_index]

    result_all = []  # a list of output words
    cur_word = hr_dataset.START

    while(True):
        for char in range(length):
            result=[]
            if cur_word == hr_dataset.END:  # end of a sentence
                
                result.append(cur_word)
                next_word(cur_word)

                if len(start_phrases) == 0:
                    continue

                for w in start_phrases.pop(0):
                    result_all.append(w)
                    cur_word = next_word(w)

            else:
                result.append(cur_word)
                cur_word = next_word(cur_word)
        result_all+=result
        if(not isContinue()):
            break

    # Convert a list of generated words to a string
    result_all = "".join(result_all)
    result_all = result_all.strip("/")  # remove trailing slashes
    # print("result",result)
    # if(input("IS continue")=="y"):
    #         result+=generate()
    
    return result_all

if __name__=="__main__":
    learing_rate=0.1
    optimizer=optim.Adam(model.parameters(), lr=learing_rate)
    training(optimizer,epoch=10)