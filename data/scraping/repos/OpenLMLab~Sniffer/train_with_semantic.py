import json
import random
import httpx
import msgpack
import numpy as np
import openai
import time
import torch
import torch.nn as nn

from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

import os
import sys

project_path = os.path.abspath('project_path')
if project_path not in sys.path:
    sys.path.append(project_path)
import sniffer_model_info


class MixedFeatureModel(nn.Module):

    def __init__(self, semantic_feat_dim, modelwise_feat_dim, class_num):
        super(MixedFeatureModel, self).__init__()
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(semantic_feat_dim, 16),
                                       torch.nn.Dropout(0.5), torch.nn.ReLU())

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(modelwise_feat_dim + 16, 300),
            torch.nn.Dropout(0.5), torch.nn.ReLU(), torch.nn.Linear(300, 300),
            torch.nn.Dropout(0.5), torch.nn.ReLU(),
            torch.nn.Linear(300, class_num))

    def forward(self, semantic_feat, modelwise_feat):
        semantic_feat = self.fc1(semantic_feat)
        feat = torch.cat((semantic_feat, modelwise_feat), dim=-1)
        outputs = self.fc2(feat)
        return outputs


def train(samples_train,
          samples_test,
          semantic_feat_dim=768,
          model_num=4,
          feat_num=4,
          class_num=5,
          ckpt_name='linear_en.pt'):
    # model_num `loss`s  and  C(model_num, 2) `feature`s
    modelwise_feat_dim = model_num + int(model_num *
                                         (model_num - 1) / 2) * feat_num
    linear_model = MixedFeatureModel(semantic_feat_dim, modelwise_feat_dim,
                                     class_num)
    linear_model.to('cuda')

    train_semantic_feats = [x[0] for x in samples_train]
    test_semantic_feats = [x[0] for x in samples_test]

    train_modelwise_feats = [x[1] for x in samples_train]
    test_modelwise_feats = [x[1] for x in samples_test]

    train_outputs = [x[2] for x in samples_train]
    test_outputs = [x[2] for x in samples_test]

    mask_re = []
    mask_sum = []
    for output in samples_test:
        if output[-1] == 'gpt3re':
            mask_re.append(1)
        else:
            mask_re.append(0)

        if output[-1] == 'gpt3sum':
            mask_sum.append(1)
        else:
            mask_sum.append(0)

    mask_re = torch.tensor(mask_re).to('cuda')
    mask_sum = torch.tensor(mask_sum).to('cuda')

    train_semantic_feats = torch.tensor(train_semantic_feats).to('cuda')
    test_semantic_feats = torch.tensor(test_semantic_feats).to('cuda')

    train_modelwise_feats = torch.tensor(train_modelwise_feats).to('cuda')
    test_modelwise_feats = torch.tensor(test_modelwise_feats).to('cuda')

    train_outputs = torch.tensor(train_outputs).to('cuda')
    test_outputs = torch.tensor(test_outputs).to('cuda')

    # train the linear model
    training(linear_model, train_semantic_feats, test_semantic_feats,
             train_modelwise_feats, test_modelwise_feats, train_outputs,
             test_outputs, class_num, [mask_re, mask_sum])

    # save the model and load at detector-backend
    torch.save(linear_model.cpu(), ckpt_name)

    # testing
    saved_model = torch.load(ckpt_name)
    linear_model = linear_model.load_state_dict(saved_model.state_dict())


def training(model,
             train_semantic_feats,
             test_semantic_feats,
             train_modelwise_feats,
             test_modelwise_feats,
             y_train,
             y_test,
             class_num,
             masks=[]):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 10000

    num = [sum(y_test == i) for i in range(class_num)]
    mask_re = masks[0].cpu().numpy()
    mask_sum = masks[1].cpu().numpy()
    print(num)

    for it in tqdm(range(n_epochs)):
        outputs = model(train_semantic_feats, train_modelwise_feats)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % 100 == 0:
            with torch.no_grad():
                outputs = model(test_semantic_feats, test_modelwise_feats)
                loss_test = criterion(outputs, y_test)
                prob = torch.nn.functional.softmax(outputs, dim=-1)  # N, label
                pred_labels = torch.argmax(prob, dim=-1)

                true_labels = y_test.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()
                acc_label = precision_score(y_true=true_labels,
                                            y_pred=pred_labels,
                                            average=None)
                rec_label = recall_score(y_true=true_labels,
                                         y_pred=pred_labels,
                                         average=None)
                acc = (true_labels == pred_labels).astype(
                    np.float32).mean().item()

                rec_label_re = (sum(
                    (pred_labels == true_labels) * mask_re)) / sum(mask_re)
                rec_label_sum = (sum(
                    (pred_labels == true_labels) * mask_sum)) / sum(mask_sum)

                print('*' * 120)
                print(
                    f'In this epoch {it+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}'
                )
                print("Total acc: {}".format(acc))
                print("The accuracy of each class:")
                print(acc_label)
                print("The recall of each class:")
                print(rec_label)
                print("rec_label_re: {}, rec_label_sum: {}".format(
                    rec_label_re, rec_label_sum))
    return


def consturct_train_features(samples_train):
    convert_train = []
    for item in samples_train:
        label_int = item['label_int']
        label = item['label']
        values = item['values']
        semantic_feats = values["roberta_feature"]
        modelwise_feats = values['losses'] + values[
            'lt_zero_percents'] + values['std_deviations'] + values[
                'pearson_list'] + values['spearmann_list']
        convert_train.append(
            [semantic_feats, modelwise_feats, label_int, label])
    return convert_train


if __name__ == "__main__":
    with open(
            '',
            'r') as f:
        samples_train = [json.loads(line) for line in f]
    with open(
            '',
            'r') as f:
        samples_test = [json.loads(line) for line in f]

    # [values, label_int, label]
    samples_train = consturct_train_features(samples_train)
    samples_test = consturct_train_features(samples_test)

    semantic_feat_dim = len(samples_train[0][0])

    train(
        samples_train=samples_train,
        samples_test=samples_test,
        semantic_feat_dim=semantic_feat_dim,
        model_num=len(sniffer_model_info.en_model_names),
        feat_num=sniffer_model_info.cur_feat_num,
        class_num=sniffer_model_info.en_class_num,
        ckpt_name=
        '')
    