import clip.clip as clip
import os
import json
import argparse
import datetime
from functools import partial

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption, evaluate_KIE, evaluate_MRR, evaluate_embodied, evaluate_zero_shot_image_classification
from task_datasets import ocrDataset, dataset_class_dict
from models import get_model, get_image
torch.hub.set_dir('/fs/nexus-scratch/kwyang3/models')
import pdb
import collections
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from collections import defaultdict
from imagenet_classnames import openai_classnames
imagenet_templates = [
    'a photo of a {}.',
    'a photo of {}.',
]
import pdb
import wandb
from datasets import Dataset, load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=-1)

    # datasets
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K SVT IC13 IC15 SVTP CUTE80 COCO-Text Total-Text WordArt CTW HOST WOST")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_seed", type=int, default=0)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    # eval choices
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--eval_ocr", action="store_true", help="Whether to evaluate on ocr.")
    parser.add_argument("--eval_vqa", action="store_true", help="Whether to evaluate on vqa.")
    parser.add_argument("--eval_caption", action="store_true", help="Whether to evaluate on caption.")
    parser.add_argument("--eval_kie", action="store_true", default=False, help="Whether to evaluate on kie.")
    parser.add_argument("--eval_mrr", action="store_true", default=False, help="Whether to evaluate on mrr.")
    parser.add_argument("--eval_embod", action="store_true", default=False, help="Whether to evaluate on embodied.")
    parser.add_argument("--eval_cls", action="store_true", default=False, help="Whether to evaluate on zero-shot classification.")

    args = parser.parse_args()
    return args


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():

        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset


def zeroshot_classifier(clip_model, classnames, templates):
	with torch.no_grad():
		zeroshot_weights = []
		i = 0
		for classname in tqdm(classnames):
			texts = [template.format(classname) for template in templates] #format with class
			texts = clip.tokenize(texts).cuda() #tokenize
			class_embeddings = clip_model.encode_text(texts) #embed with text encoder
			class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
			class_embedding = class_embeddings.mean(dim=0)
			class_embedding /= class_embedding.norm()
			zeroshot_weights.append(class_embedding)
			i += 1
		zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
	return zeroshot_weights


def main(args):
    wandb.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    clip_model, train_preprocess, val_preprocess = clip.load("ViT-L/14", args.device, jit=False)
    clip_model.eval()
    clip_model.cuda()

    dataset = dataset_class_dict[args.dataset_name]()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    classnames = dataset.classnames
    zeroshot_weights_base = zeroshot_classifier(clip_model, classnames, imagenet_templates)
    	 
    outputs=[]
    targets=[]
    dataset_dict = collections.defaultdict(list)
    for batch in tqdm(dataloader):
        image_path, target = batch['image_path'], batch['label']

        images = [get_image(img) for img in image_path]
        images = [val_preprocess(x) for x in images]
        images = torch.stack(images, dim=0).to(args.device)
        target = torch.LongTensor(target).cuda()

        # predict
        with torch.no_grad():
             image_features = clip_model.encode_image(images)
             image_features /= image_features.norm(dim=-1, keepdim=True)

        logits_base = image_features @ zeroshot_weights_base
        outputs.append(logits_base.cpu())
        targets.append(target.cpu())
	    
        toplogits, y_pred = logits_base.topk(k=20, dim=1)

        for i in range(target.size(0)):
            dataset_dict['image_path'].append(image_path[i])
            dataset_dict['label'].append(batch['label'][i])
            dataset_dict['clip_top10'].append([classnames[ind] for ind in y_pred[i].tolist()])
            dataset_dict['confidence'].append(float(toplogits[i][0]))
            dataset_dict['gt_answers'].append(classnames[batch['label'][i]])

    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk("../data/imagenet_val_dict")

    acc = accuracy(torch.cat(outputs,dim=0), torch.cat(targets,dim=0), (1,2,3,4,5,6,7,8,9,10))
    print(acc)


if __name__ == "__main__":
    args = parse_args()
    main(args)
