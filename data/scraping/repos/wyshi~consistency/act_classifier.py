
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import regex as re
import random
import itertools
import tqdm
import time
import os

import config as cfg
from torch.nn import Identity
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from allennlp.training.checkpointer import Checkpointer
# from gpt_model import GPT2SimpleLM
from pytorch_pretrained_bert import OpenAIAdam
# from torchfly.criterions import SequenceFocalLoss, SequenceCrossEntropyLoss
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AdamW#, WarmupLinearSchedule
# from transformers.modeling_gpt2 import SequenceSummary
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss
import pickle as pkl
import pdb
# In[2]:
def save_pkl(obj, dir):
    with open(dir, "wb") as fh:
        pkl.dump(obj, fh)

def load_pkl(dir):
    with open(dir, "rb") as fh:
        obj = pkl.load(fh)
    return obj

import tqdm
def split_train_val():
    import pickle as pkl
    df = pd.read_excel("training/data/300_dialog.xlsx")
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(df[df['B4']==0]['er_label_1'])
    df['er_label_1_num'] = None#labels
    df['er_label_1_num'].loc[df['B4']==0] = labels
    save_pkl(le, "training/data/labelencoder.pkl")
    def extract_data(df_dialogs):
        data = []
        for i in tqdm.trange(len(df_dialogs)):
            line = df.iloc[i]
            if line["B4"] == 0:
                text = line["Unit"].strip()
                data.append([text, line['er_label_1_num']])
            # else:
            #     text = "B:" + line["Unit"].strip()
            #     data[line["B2"]].append([text, line['ee_label_1']])                
        return data
    all_data = extract_data(df)
    import random
    random.seed(123)
    random.shuffle(all_data)
    train_data = all_data[:int(len(all_data)*0.8)]
    val_data = all_data[int(len(all_data)*0.8):]
    save_pkl(train_data, "training/data/train_data.pkl")
    save_pkl(val_data, "training/data/val_data.pkl")

class SequenceSummary(nn.Module):
    r""" Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """
    def __init__(self, config):
        super().__init__()
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                print(f"num_class: {config.num_labels}")
                num_classes = config.num_labels
            else:
                print(f"num_class here: {config.hidden_size}")
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        self.activation = Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()
        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)
    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, ..., seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2] - 1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output



torch.backends.cudnn.benchmark = True
torch.manual_seed(123)
np.random.seed(123)

class PersuadeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        # tokenizer weird behavior
        self.turn_ending = tokenizer.cls_token_id#[628, 198]
        # tokenizer.encode("\n\n\n")        
    def __len__(self):
        return len(self.data)    
    def __getitem__(self, index):
        dial_tokens = tokenizer.encode(self.data[index][0]) + [self.turn_ending]
        cls_token_location = dial_tokens.index(self.tokenizer.cls_token_id)
        dial_act = self.data[index][1]
        return dial_tokens, cls_token_location, dial_act        
    def collate(self, unpacked_data):
        return unpacked_data

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.add_special_tokens({'cls_token': '[CLS]'})


class GPT2DoubleHeadsModel_modified(GPT2DoubleHeadsModel):
    def __init__(self, config):
        super().__init__(config)
        # config.num_labels = 1
        config.num_labels = le.classes_.shape[0]
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.init_weights()

config = GPT2Config()
config = config.from_pretrained('gpt2-medium')
le = load_pkl("training/data/labelencoder.pkl")
config.num_labels = le.classes_.shape[0]
model_A = GPT2DoubleHeadsModel_modified(config)
model_A.resize_token_embeddings(len(tokenizer)) 
# model_B = GPT2DoubleHeadsModel.from_pretrained('gpt2')

device = torch.device("cuda:5")
torch.cuda.set_device(device)
model_A = model_A.to(device)

model_A_states, model_B_states = torch.load(cfg.old_medium_model_dir)
model_A_states['transformer.wte.weight'] = torch.cat([model_A_states['transformer.wte.weight'][:50257,:],
                                                     torch.randn([1, 1024]).to(device)], dim=0)
model_A.load_state_dict(model_A_states, strict=False)

# model_B_states['transformer.wte.weight'] = model_B_states['transformer.wte.weight'][:50257,:]


# load training data
train_data = load_pkl("training/data/train_data.pkl")
val_data = load_pkl("training/data/val_data.pkl")

train_dataset = PersuadeDataset(train_data, tokenizer)
val_dataset = PersuadeDataset(val_data, tokenizer)

batch_size = 1

train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=train_dataset.collate)
val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=train_dataset.collate)





# define the losses
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
# eval_criterion = SequenceCrossEntropyLoss()

# In[9]:


def train_one_iter(batch, update_count, fp16=False):
    dial_tokens, cls_token_location, dial_act = batch
    input_ids = torch.LongTensor(dial_tokens).unsqueeze(0).to(device)
    # [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
    mc_token_ids = torch.tensor(cls_token_location).unsqueeze(0).to(device)
    mc_labels = torch.tensor(dial_act).unsqueeze(0).to(device)

    outputs = model_A(input_ids, mc_token_ids=mc_token_ids)
    lm_prediction_scores, mc_logits = outputs[:2]
    loss = criterion(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))

    # past = None
    # all_logits = []
   
    # for turn_num, (dial_turn_inputs, dialog_turn_act) in enumerate(zip(dial_inputs, dialog_acts)):
    #     # if role_ids[turn_num] == 0:
    #     #     # breakpoint()
    #     #     logits, past = model_A(dial_turn_inputs, past=past)
    #     #     all_logits.append(logits)
    #     # else:
    #     #     # breakpoint()
    #     #     logits, past = model_B(dial_turn_inputs, past=past)
    #     #     all_logits.append(logits)

    # all_logits = torch.cat(all_logits, dim=1) # torch.Size([1, 505, 50260]), 505 = sum of tokens from 21 sentences
    
    
    
    # # target
    # all_logits = all_logits[:, :-1].contiguous() # torch.Size([1, 504, 50260])
    # target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()# torch.Size([1, 504])
    # target_mask = torch.ones_like(target).float()# torch.Size([1, 504])
    
    loss /= num_gradients_accumulation
    
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
        
    record_loss = loss.item() * num_gradients_accumulation
    # print("record_loss: {}".format(record_loss))
    # perplexity = np.exp(record_loss)
    
    return record_loss#, perplexity

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from utils import print_cm

def validate(dataloader):
    with torch.no_grad():
        pbar = progress_bar(dataloader)

        correct = 0
        total = 0
        y_true, y_pred = [], []

        for batch in pbar:
            
            dial_tokens, cls_token_location, dial_act = batch[0]
            input_ids = torch.LongTensor(dial_tokens).unsqueeze(0).to(device)
            # [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
            mc_token_ids = torch.tensor(cls_token_location).unsqueeze(0).to(device)
            labels = torch.tensor(dial_act).unsqueeze(0).to(device)

            outputs = model_A(input_ids, mc_token_ids=mc_token_ids)
            lm_prediction_scores, mc_logits = outputs[:2]
            y_true.extend([dial_act])
            _, predicted_labels = torch.max(outputs, 1)
            y_pred.extend(predicted_labels.tolist())

            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
        f1 = f1_score(y_true, y_pred, average="binary")
        print(f"Epcoh {ep} Validation accuracy: {correct/total}, f1: {f1}")
        # print_cm(confusion_matrix(y_true, y_pred, labels=range(len(le.classes_))), labels=[l[-5:] for l in le.classes_.tolist()])
        return correct/total, f1


# ### Training

# In[10]:


checkpointer = Checkpointer(serialization_dir="Checkpoint_act_clf", 
                            keep_serialized_model_every_num_seconds=3600*2, 
                            num_serialized_models_to_keep=5)


# In[11]:


# optimizer
num_epochs = 10
num_gradients_accumulation = 1
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

param_optimizer = list(model_A.named_parameters()) 
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


optimizer = OpenAIAdam(optimizer_grouped_parameters,
                       lr=2e-5,
                       warmup=0.1,
                       max_grad_norm=1.0,
                       weight_decay=0.01,
                       t_total=num_train_optimization_steps)


# In[12]:


# support fp16
# [model_A, model_B], optimizer = amp.initialize([model_A, model_B], optimizer, opt_level="O1")


# In[13]:

import tqdm 
update_count = 0
progress_bar = tqdm.tqdm
start = time.time()
best_acc = -float('Inf')
best_f1 = -float('Inf')

for ep in tqdm.tqdm(range(num_epochs)):

    "Training"
    pbar = progress_bar(train_dataloader)
    model_A.train()
    # model_B.train()
    
    for batch in pbar:
        batch = batch[0]
        # without relative position
        # if sum([len(item) for item in batch[1]]) > 1024:
        #     continue
            
        record_loss = train_one_iter(batch, update_count, fp16=False)
        
        update_count += 1

        if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
            # update for gradient accumulation
            optimizer.step()
            optimizer.zero_grad()
            
            # speed measure
            end = time.time()
            speed = batch_size * num_gradients_accumulation / (end - start)
            start = end
            
            # show progress
            pbar.set_postfix(loss=record_loss, speed=speed)

    "Evaluation"
    model_A.eval()
    # model_B.eval()
    val_acc, val_f1 = validate(val_dataloader)
    print(f"val f1: {val_f1}, valid acc: {val_acc}")
    is_best_so_far = val_f1 > best_f1
    # if is_best_so_far:
    #     best_acc = val_acc
    #     # torch.save(model_clf.state_dict(), f"Checkpoint_clf/best_acc_{best_acc}_f1_{val_f1}_with_past.pth")
    if is_best_so_far:
        best_f1 = val_f1
        torch.save(model_A.state_dict(), f"Checkpoint_act_clf/best_acc_{best_acc}_f1_{best_f1}.pth")
        # checkpointer.save_checkpoint(ep, model_A.state_dict(), {"None": None}, is_best_so_far)

print("best acc: {}, best f1: {}".format(best_acc, best_f1))





# In[ ]:




