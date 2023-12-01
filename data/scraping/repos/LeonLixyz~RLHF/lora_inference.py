import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_loader import pairwise_data_tokenized
from enn_net import Epinet, generate_rd
from reward_model import RewardENN, LoraRewardENN
import os
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig
import openai
from torch.optim.lr_scheduler import StepLR
from accelerate import Accelerator
import os

def load_config(config_path):
    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)
    return config

def load_reward(config, model):
    hid_rep_size = model.lm_head.in_features
    EtaReward = Epinet(hid_rep_size, config["ref_size"], config["hidden_size"], config["output_size"], gain=config["gain"])
    PReward = Epinet(hid_rep_size, config["ref_size"], config["hidden_size"], config["output_size"], gain=config["gain"])
    for param in PReward.model.parameters():
        param.requires_grad = False

    reward_head = torch.nn.Linear(hid_rep_size, 1)
    torch.nn.init.xavier_uniform_(reward_head.weight, gain=config["gain"])
    torch.nn.init.zeros_(reward_head.bias)

    reward_ENN = LoraRewardENN(model, EtaReward, PReward, reward_head, scale_factor=config["scale_factor"])
    return reward_ENN

def load_model(config):

    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])
    peft_model_id = 'lora_model/base_model'
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    base_model = PeftModel.from_pretrained(base_model, peft_model_id)
    reward_model = load_reward(config, base_model)


    reward_model.reward_head.load_state_dict(torch.load('lora_model/reward_head.pth'))
    reward_model.EtaNetwork.load_state_dict(torch.load('lora_model/EtaNetwork.pth'))
    reward_model.PNetwork.load_state_dict(torch.load('lora_model/PNetwork.pth'))

    reward_model.eval()  # switch the model to evaluation mode
    return reward_model, tokenizer

def load_data(config, tokenizer):
    with open(config["data_dir"]) as f:
        annotated = json.load(f)[:20]
    dataset = pairwise_data_tokenized(annotated, tokenizer, config["num_dimensions"], config["num_samples"])
    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=config["shuffle"])
    return training_dataloader

def main():
    config = load_config("config.json")
    reward_model, tokenizer = load_model(config)
    training_dataloader = load_data(config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)

    for i, batch in enumerate(training_dataloader):
        print('ith batch', i)
        data, z_samples = batch
        z_samples = z_samples.to(device)

        p_1, p_2, p_1_att, p_2_att, label = data
        p_1 = p_1.to(device)
        p_2 = p_2.to(device)
        p_1_att = p_1_att.to(device)
        p_2_att = p_2_att.to(device)
        label = label.to(device)
        z = z_samples

        reward_1 = reward_model(p_1, p_1_att, z)
        reward_2 = reward_model(p_2, p_2_att, z)

        print('preference is', label)
        print('reward_1 is', reward_1)
        print('reward_2 is', reward_2)
        break

if __name__ == "__main__":
    main()