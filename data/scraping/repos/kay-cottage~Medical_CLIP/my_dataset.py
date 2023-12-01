
# -*- coding: utf-8 -*-
"""
Created on Wed Aug.17 09:12:41 2022
@author: gw.kayak
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils

#images to text
class image_caption_dataset(Dataset):
    def __init__(self, df):
        self.images = df["image"].tolist()
        self.caption = df["caption"].tolist()
        # args from open-ai clip
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     #transforms.RandomHorizontalFlip(), # 水平翻转
                                     #transforms.Grayscale(3),
                                     transforms.ToTensor(), # 转为张量
                                     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (.26862954, 0.26130258, 0.27577711)),#args from openai-clip
                                     ])
        '''
        self.vit_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     #transforms.Grayscale(3),
                                     transforms.ToTensor(), # 转为张量
                                     
                                     ])
        '''

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        img=Image.open(self.images[idx])
        images=self.transform(img)
        #vit_img=self.vit_transform(img)

        caption = self.caption[idx]
        return images, caption


# images to images
class image_to_images_caption_dataset(Dataset):
    def __init__(self, df):
        self.images = df["image"].tolist()
        self.caption = df["caption"].tolist()
        # args from open-ai clip
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(), # 水平翻转
                                     transforms.ToTensor(), # 转为张量
                                     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (.26862954, 0.26130258, 0.27577711))])


    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        img=Image.open(self.images[idx])
        images=self.transform(img)

        caption = Image.open(self.caption[idx])
        caption = self.transform(caption)
        return images, caption

    
    '''
    # 需要导入模块: import transformers [as 别名]
    # 或者: from transformers import BertConfig [as 别名]
    def main():
        with open("build/data/bert_tf_v1_1_large_fp32_384_v2/bert_config.json") as f:
            config_json = json.load(f)

        config = BertConfig(
            attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
            hidden_act=config_json["hidden_act"],
            hidden_dropout_prob=config_json["hidden_dropout_prob"],
            hidden_size=config_json["hidden_size"],
            initializer_range=config_json["initializer_range"],
            intermediate_size=config_json["intermediate_size"],
            max_position_embeddings=config_json["max_position_embeddings"],
            num_attention_heads=config_json["num_attention_heads"],
            num_hidden_layers=config_json["num_hidden_layers"],
            type_vocab_size=config_json["type_vocab_size"],
            vocab_size=config_json["vocab_size"])

        model = load_from_tf(config, "build/data/bert_tf_v1_1_large_fp32_384_v2/model.ckpt-5474")
        torch.save(model.state_dict(), "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")
        save_to_onnx(model)
    '''
