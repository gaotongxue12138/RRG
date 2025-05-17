import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import Dataset
from transformers import AdamW
from tqdm import tqdm
from transformers import T5ForConditionalGeneration,AutoTokenizer
import argparse
class Align_Dataset(Dataset):
    def __init__(self, data_path, block_size,url,mode):
        self.adapter_tokenizer = AutoTokenizer.from_pretrained(url)
        self.block_size = block_size
        self.mode = mode
        self.inputs_ids = []
        self.inputs_mask = []
        self.targets_ids = []
        self.targets_mask = []
        x = pd.read_json(data_path, lines= True)
        for i in range(len(x)):                   
            source_code = x['nl'][i] + x['relevant'][i]
            target_code = x['code'][i] 
            tokenizer = self.adapter_tokenizer
            input_ = tokenizer(source_code,max_length = self.block_size ,pad_to_max_length=True,truncation = True,return_tensors="pt")
            target_ = tokenizer(target_code,max_length = self.block_size ,pad_to_max_length=True,truncation = True,return_tensors="pt")
            self.inputs_ids.append(input_['input_ids'])
            self.inputs_mask.append(input_['attention_mask'])
            self.targets_ids.append(target_['input_ids'])
            self.targets_mask.append(target_['attention_mask'])
    def __len__(self):
        return len(self.inputs_ids)
    def __getitem__(self, item):   
        return self.inputs_ids[item],self.inputs_mask[item],self.targets_ids[item],self.targets_mask[item]
def evalution_bleu(model,tokenizer,eval_url):
    model.eval()
    dataset = Align_Dataset(eval_url, 512, model_name_or_path,"eval")
    dataloader = DataLoader(dataset, batch_size = 1)
    results = []
    for step, (batch ,token_labels,_,_) in tqdm(enumerate(dataloader)):
        inputs = batch.to(device)
        ans = model.generate(input_ids =inputs[0],max_length = 512,use_cache=True)
        text = tokenizer.decode(ans[0], skip_special_tokens=True)
        results.append(text)
    df = pd.read_json(eval_url, lines=True)
    df['ans'] = results
    EM , BLEU ,CodeBLEU= get_EM_BLEU(df['code'],df['ans'])
    return EM, BLEU, CodeBLEU           

def stage1_train(model,dataloader,epochs,optimizer,criterion,log,tokenizer):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (input_ids ,input_mask,target_ids,_ ) in tqdm(enumerate(dataloader)):
            y = target_ids.to(device, dtype = torch.long).squeeze(dim = 1)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = input_ids.to(device, dtype = torch.long).squeeze(dim = 1)
            mask = input_mask.to(device, dtype = torch.long).squeeze(dim = 1)
            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_loss = total_loss / len(dataloader)
    return model

parser = argparse.ArgumentParser(description='描述你的脚本')
parser.add_argument('--dataset_name', type=str, help='帮助信息')
parser.add_argument('--device_id', type=int, help='帮助信息')
args = parser.parse_args()
name = args.dataset_name
device_id = args.device_id
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
train_url = '*'
test_url = '*'
model_path = '*'
model_name_or_path = "*"
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model.to(device)
EPOCH = 10
BATCH_SIZE = 16
optimizer = AdamW(model.parameters(), lr = 1e-4)
align_dataset  = Align_Dataset(train_url,512,model_name_or_path,"train")
align_dataloader = DataLoader(align_dataset, batch_size = BATCH_SIZE)
criterion = nn.CrossEntropyLoss()
model = stage1_train(model,align_dataloader,EPOCH, optimizer,criterion,log,tokenizer)
torch.save(model.state_dict(), model_path)
evalution_bleu(model,tokenizer,test_url)