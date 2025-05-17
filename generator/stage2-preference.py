import transformers
from transformers import AutoTokenizer
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
logging.getLogger().setLevel(logging.ERROR)
from trl import AutoModelForSeq2SeqLMWithValueHead
from typing import Dict, Optional, Sequence, List
import torch
import pandas as pd
import torch
from datasets import load_dataset,Dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import warnings
import logging
import sys
from trl import PPOConfig,AutoModelForSeq2SeqLMWithValueHead
import logging
import sys
sys.path.append('*')
from utils.get_bleu import get_EM_BLEU
from nltk.translate.bleu_score import sentence_bleu
import math
import re
from codebleu import calc_codebleu
import argparse
parser = argparse.ArgumentParser(description='描述你的脚本')
parser.add_argument('--epochs', type=int, help='帮助信息')
parser.add_argument('--dataset_name', type=str, help='帮助信息')
parser.add_argument('--learn_rate', type=float, help='帮助信息')
parser.add_argument('--model_name', type=str, help='帮助信息')
parser.add_argument('--rl_size', type=int, help='帮助信息')
parser.add_argument('--reward_mode', type=str, help='帮助信息')
parser.add_argument('--data_mode', type=str, help='帮助信息')
parser.add_argument('--device_id', type=str, help='帮助信息')
parser.add_argument('--code_tokens', type=int, help='帮助信息')
args = parser.parse_args()
epochs = args.epochs
name = args.dataset_name
lr = args.learn_rate
model_name = args.model_name
rl_size = args.rl_size
reward_mode = args.reward_mode
data_mode = args.data_mode
device_id = args.device_id
code_tokens = args.code_tokens
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
if name =="CSN_py":
    lang_ = 'python'
else:
    lang_ = 'java'    
class Adapter_Dataset(Dataset):
    def __init__(self, data_path, block_size,reward_tokenizer,tokenizer,mode, code_tokens):
        self.adapter_tokenizer = tokenizer
        self.generator_tokenizer = reward_tokenizer
        self.block_size = block_size
        self.mode = mode
        self.code_tokens = code_tokens
        self.inputs = []
        self.token_labels = []
        self.code_prompts = []
        x = pd.read_json(data_path, lines= True)
        if mode == "train":
            l = rl_size
        else:
            l = len(x)
        if self.generator_tokenizer.pad_token_id==None:
            self.generator_tokenizer.pad_token_id = self.generator_tokenizer.bos_token_id
        for i in range(l):                   
            code = self.generator_tokenizer.encode(x["code"][i])
            nl = self.generator_tokenizer.encode(x["nl"][i])
            learn_code_prompt = self.generator_tokenizer.encode(x["relevant"][i])
            input_ids, input_labels = self.pad_and_get_mask(code, nl,learn_code_prompt,self.generator_tokenizer)            
            code_prompt = self.adapter_tokenizer(x['nl'][i] + x["relevant"][i],max_length=self.block_size,pad_to_max_length=True,truncation=True,return_tensors="pt")['input_ids']
            self.inputs.append(input_ids)
            self.token_labels.append(input_labels)
            self.code_prompts.append(code_prompt)
    def pad_and_get_mask(self, code, nl,learn_code_prompt,tokenizer):
        while (len(code) + len(nl) + 2 + self.code_tokens > self.block_size):
            if (len(code) > len(nl)):
                code = code[:-1]
            else:
                nl = nl[:-1]
        while len(learn_code_prompt) > self.code_tokens:
            learn_code_prompt = learn_code_prompt[:-1]
        pad_len = self.code_tokens - len(learn_code_prompt)
        learn_code_prompt += [self.generator_tokenizer.bos_token_id] * pad_len
        pad_code = learn_code_prompt
        inputs =  nl +  pad_code +   [self.generator_tokenizer.bos_token_id] + code + [self.generator_tokenizer.eos_token_id]
        labels = [1] * len(nl) + [1] * self.code_tokens + [2]  + [2] * len(code) + [2]         
        assert len(inputs) <= self.block_size
        pad_len = self.block_size - len(inputs)
        inputs += [self.generator_tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)
        return inputs , labels 
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, item):   
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item]) , torch.tensor(self.code_prompts[item])
generate_url = '*'
reward_url = '*'
train_path='*'
test_path='*'
pretrained_generator = '*'
pretrained_refactorer = '*'
tokenizer = AutoTokenizer.from_pretrained(generate_url)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(generate_url)
pretrained_state_dict = torch.load(pretrained_refactorer,map_location = device)
new_state_dict = {}
for key in pretrained_state_dict.keys():
    new_key = 'pretrained_model.' + key
    new_state_dict[new_key] = pretrained_state_dict[key]
model.load_state_dict(new_state_dict, strict=False)
model = model.to(device) 
reward_tokenizer = AutoTokenizer.from_pretrained(reward_url)
if reward_tokenizer.pad_token_id == None:
    reward_tokenizer.pad_token_id = reward_tokenizer.bos_token_id
reward_model = AutoModelForCausalLM.from_pretrained(reward_url)
pretrained_state_dict = torch.load(pretrained_generator,map_location=device)
reward_model.load_state_dict(pretrained_state_dict)
reward_model = reward_model.to(device) 
block_size = 512
batch_size = 16
init_kl_coef = 0.5
train_dataset = Adapter_Dataset(train_path, block_size, reward_tokenizer,tokenizer,"train",code_tokens)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size,drop_last = True)
test_dataset = Adapter_Dataset(test_path, block_size, reward_tokenizer,tokenizer,"test",code_tokens)
test_dataloader = DataLoader(test_dataset, batch_size = 1)
from trl import PPOTrainer
config = PPOConfig(
    model_name="*",
    learning_rate=lr,
    batch_size = batch_size,
    mini_batch_size =batch_size,
    init_kl_coef = init_kl_coef
)
optimizer = AdamW(model.parameters(),lr = lr)
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    optimizer = optimizer,
)
def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens
def get_reward(ans,output,ans_len):
    output_bleu = calc_codebleu([str(ans)], [str(output)], lang=lang_, weights=(1/4, 1/4, 1/4, 1/4))['codebleu']
    ans_l = ans_len  
    return output_bleu * math.sqrt(ans_l)   
generation_kwargs = {
    "max_length" : code_tokens,
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": -1,
}
criterion = nn.CrossEntropyLoss()
for epoch in range(epochs):
    reward_model.eval()
    total_loss = 0.0
    total_bleu = 0.0
    for step, (batch ,token_labels ,adapter_data) in tqdm(enumerate(train_dataloader)):
        adapter_data = adapter_data.to(device)
        adapter_data = torch.chunk(adapter_data, batch_size, dim=0)
        adapter_data = [t.squeeze() for t in adapter_data]
        querys = [tokenizer.decode(r.squeeze(),skip_special_tokens= True) for r in adapter_data]
        inputs_id = adapter_data
        response_tensors = ppo_trainer.generate(adapter_data, **generation_kwargs)
        relevants = [tokenizer.decode(r.squeeze(),skip_special_tokens= True) for r in response_tensors]
        rewards = []
        attn_mask = (token_labels.clone().detach() != 0).to(dtype=torch.uint8, device=device)
        nl_mask = (token_labels.clone().detach() == 1).to(dtype=torch.uint8, device=device)
        loss_mask = (token_labels.clone().detach() == 2).to(dtype=torch.uint8, device=device)
        inputs = batch.to(device)
        for i in range(len(relevants)):
            print("1. output:",relevants[i])
            relevant = reward_tokenizer.encode(relevants[i])
            if len(relevant) >= code_tokens:
                relevant = relevant[:code_tokens]
            if name =="CSN":
               pass
            elif name =="CSN_py":
                pass
            else:
                pad = code_tokens -len(relevant)
                relevant += [reward_tokenizer.bos_token_id] * pad
            ones_indices = nl_mask[i].nonzero(as_tuple=True)[0]
            end_idx = ones_indices[-1]
            inputs[i, 1 + end_idx - len(relevant):end_idx + 1] = torch.tensor(relevant, dtype=torch.long, device=inputs.device)
            input_reward = inputs[i,:end_idx+2]
            text = reward_tokenizer.decode(input_reward,skip_special_tokens= True)
            code = inputs[i,end_idx+1:]
            ans = reward_tokenizer.decode(code,skip_special_tokens= True)
            outputs = reward_model.generate(input_ids = input_reward.unsqueeze(dim=0),min_length = 10,max_length = 512,eos_token_id = reward_tokenizer.eos_token_id,use_cache = True)
            outputs = reward_tokenizer.decode(outputs[0],skip_special_tokens=True)
            outputs = outputs.replace(text,"")
            ans_len = len(reward_tokenizer.encode(ans))
            reward =  get_reward(ans,outputs,ans_len)
            total_bleu+=reward
            rewards.append(torch.tensor(reward))
        stats = ppo_trainer.step(inputs_id, response_tensors, rewards)
        ppo_trainer.log_stats(stats, {"query":querys,"inputs_id":inputs_id,"response":relevants}, reward)
reward_model_save_url = '*'
ppo_trainer_save_url = '*'
reward_model.save_pretrained(reward_model_save_url)
ppo_trainer.save_pretrained(ppo_trainer_save_url)
adapter_model_url = ppo_trainer_save_url
adapter_tokenzier_url  ="*"
adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_tokenzier_url)
adapter_model = T5ForConditionalGeneration.from_pretrained(adapter_model_url).to(device)
reward_model_url =  reward_model_save_url
reward_url = '*'
generator_tokenizer = AutoTokenizer.from_pretrained(reward_url)
generator_model = AutoModelForCausalLM.from_pretrained(reward_model_url).to(device)
if generator_tokenizer.pad_token_id == None:
    generator_tokenizer.pad_token_id = generator_tokenizer.bos_token_id
adapter_model.eval()
total_bleu = 0.0
ans0 = []
ans1 = []
for step, (batch ,token_labels ,adapter_data) in tqdm(enumerate(test_dataloader)):
    adapter_data = adapter_data.to(device)
    response_tensors = adapter_model.generate(adapter_data.squeeze(0),max_length=512)
    relevants = [adapter_tokenizer.decode(r.squeeze(),skip_special_tokens= True) for r in response_tensors]
    rewards = []
    attn_mask = (token_labels.clone().detach() != 0).to(dtype=torch.uint8, device=device)
    nl_mask = (token_labels.clone().detach() == 1).to(dtype=torch.uint8, device=device)
    loss_mask = (token_labels.clone().detach() == 2).to(dtype=torch.uint8, device=device)
    inputs = batch.to(device)
    for i in range(len(relevants)):
        relevant = generator_tokenizer.encode(relevants[i])
        if len(relevant) >= code_tokens:
            relevant = relevant[:code_tokens]
        if name =="CSN":
                pass
        elif name =="CSN_py":
            pass
        else:
            pad = code_tokens -len(relevant)
            relevant += [generator_tokenizer.bos_token_id] * pad
        ones_indices = nl_mask[i].nonzero(as_tuple=True)[0]
        end_idx = ones_indices[-1]
        inputs[i, 1 + end_idx - len(relevant):end_idx + 1] = torch.tensor(relevant, dtype=torch.long, device=inputs.device)
        input_reward = inputs[i,:end_idx+2]
        text = generator_tokenizer.decode(input_reward,skip_special_tokens= True)
        code = inputs[i,end_idx+1:]
        ans = generator_tokenizer.decode(code,skip_special_tokens= True)
        outputs = generator_model.generate(input_ids = input_reward.unsqueeze(dim=0),min_length = 10,max_length = 512,eos_token_id = generator_tokenizer.eos_token_id)
        outputs = generator_tokenizer.decode(outputs[0],skip_special_tokens=True)
        outputs = outputs.replace(text,"")
        ans0.append(ans)
        ans1.append(outputs)
        bleu = calc_codebleu([str(ans)], [str(outputs)], lang="java", weights=(1/4, 1/4, 1/4, 1/4), tokenizer=tokenize_for_bleu_eval)['codebleu']
        total_bleu += bleu
eval_ = pd.read_json(test_path,lines = True)        
a,b,c = get_EM_BLEU(eval_['code'],ans1)
print('em',a,'bleu',b,"codebleu",c)