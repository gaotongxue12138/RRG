import torch
from transformers import  AutoModelForCausalLM
import torch.nn as nn
import pandas as pd
import torch
from datasets import load_dataset,Dataset
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
class base_Dataset(Dataset):
    def __init__(self, data_path, block_size,generator_tokenizer, mode):
        self.generator_tokenizer = generator_tokenizer
        self.block_size = block_size
        self.mode = mode
        self.inputs = []
        self.token_labels = []
        self.excc = 0
        x = pd.read_json(data_path, lines= True)
        if self.generator_tokenizer.pad_token_id==None:
            self.generator_tokenizer.pad_token_id = self.generator_tokenizer.bos_token_id
        for i in range(len(x)): 
            code = generator_tokenizer.encode(x["code"][i])
            nl = x['nl'][i]            
            nl = generator_tokenizer.encode(nl)
            relevant = generator_tokenizer.encode(x['relevant'][i])
            input_ids, input_labels  = self.pad_and_get_mask(code,relevant, nl,generator_tokenizer)          
            self.inputs.append(input_ids)
            self.token_labels.append(input_labels)
    def pad_and_get_mask(self, code,relevant, nl, tokenizer):
        if len(relevant) >=codetokens:
            relevant = relevant[:codetokens]
        else:
            relevant += (codetokens - len(relevant)) * [self.generator_tokenizer.bos_token_id]        
        while (len(code) + len(relevant) + len(nl) + 2 > self.block_size):
            if (len(code) > len(nl)):
                code = code[:-1]
            else:
                nl = nl[:-1]    
        if self.mode == 'train':    
            inputs =  nl + relevant  +  [self.generator_tokenizer.bos_token_id]  + code + [self.generator_tokenizer.eos_token_id]
            labels = [1] * len(nl) + [1] * len(relevant) + [2] * (len(code)+1) + [2]
        else:
            inputs =  nl +  relevant  + [self.generator_tokenizer.bos_token_id]
            labels =[1] * len(nl) + [1] * len(relevant) + [2]
            return inputs, labels
        assert len(inputs) <= self.block_size
        pad_len = self.block_size - len(inputs)
        inputs += [self.generator_tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)
        return inputs , labels
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item]) 
def evalution_bleu(model,tokenizer,eval_url):
    model.eval()
    dataset = base_Dataset(eval_url, 512, tokenizer,"eval")
    dataloader = DataLoader(dataset, batch_size = 1)
    results = []
    for step, (batch,_) in tqdm(enumerate(dataloader)):
        old_text = tokenizer.decode(batch[0], skip_special_tokens=True)
        inputs = batch.to(device)
        ans = model.generate(input_ids =inputs,max_length = 512,eos_token_id = tokenizer.eos_token_id)
        ans = tokenizer.decode(ans[0], skip_special_tokens=True).replace(old_text,"") 
        results.append(ans)
    df = pd.read_json(eval_url, lines=True)
    df['ans'] = results
    EM , BLEU ,CodeBLEU= get_EM_BLEU(df['code'],df['ans'])
    print("EM Score:", EM)    
    print("BLEU",BLEU)
    print("CodeBLEU",CodeBLEU)
    return EM, BLEU, CodeBLEU
def train(model,tokenizer,train_url):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr = 0.00015)
    dataset = base_Dataset(train_url, 512, tokenizer,"train")
    train_dataloader = DataLoader(dataset, batch_size = BATCHSIZE , shuffle=True)
    for epoch in range(EPOCH):
        model.train()
        total_loss = 0.0
        for step, (batch ,token_labels) in tqdm(enumerate(train_dataloader)):
            inputs = batch.to(device)
            attn_mask = (token_labels.clone().detach() != 0).to(dtype=torch.uint8, device=device)
            loss_mask = (token_labels.clone().detach() == 2).to(dtype=torch.uint8, device=device)                 
            outputs = model(input_ids = inputs,attention_mask = attn_mask)    
            logits = outputs.logits
            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)   
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
            total_loss += loss.item()
            loss.backward()  
            optimizer.step()
            optimizer.zero_grad()
        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{EPOCH}, Average Loss: {average_loss}')
    EM,BLEU,CodeBLEU= evalution_bleu(model,tokenizer,test_url)
    return model
import argparse
parser = argparse.ArgumentParser(description='描述你的脚本')
parser.add_argument('--dataset_name', type=str, help='帮助信息')
parser.add_argument('--model_name', type=str, help='帮助信息')
parser.add_argument('--device_id', type=int, help='帮助信息')
args = parser.parse_args()
name = args.dataset_name
model_name = args.model_name
device_id = args.device_id
model_name_or_path = '*'
train_url = '*'
test_url = '*'
model_path = '*'
EPOCH = 10
BATCHSIZE = 16
if name == "concode":
    codetokens = 64
else:
    codetokens = 168
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model.to(device)
model = train(model,tokenizer,train_url)
torch.save(model.state_dict(), model_path)