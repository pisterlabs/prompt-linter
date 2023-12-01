'''
!pip install transformers
!pip install sentence_transformers
!pip install torchmetrics
!pip install openai
'''

# importing
# import sys

import numpy as np  # to handle data
import pandas as pd  # to handle and save data
import os
# import pickle
from datetime import datetime as d  # to generate timestamps to save models
import math
import random
# import json

# from sentence_transformers import SentenceTransformer  # for word embedding
from transformers import AutoTokenizer, AutoModel

import torch  # for AI
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
from torchmetrics import R2Score

import matplotlib.pyplot as plt  # to plot training
import openai  # to generate training data

import seaborn as sns  # to analyse data
from wordcloud import WordCloud

# import nltk
# nltk.download('stopwords')  # uncomment this line to use the NLTK Downloader
from nltk.corpus import stopwords

openai.api_key = os.getenv('OPENAI_API_KEY')


class CustomTopicDataset(Dataset):  # the dataset class
    def __init__(self, sentences, labels):
        self.x = sentences
        self.y = labels
        if isinstance(self.x, list):
            self.length = len(self.x)
            self.shape = len(self.x[0])
        else:
            self.length = self.x.shape[0]
            self.shape = self.x[0].shape[0]
        self.feature_names = ['sentences', 'labels']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NeuralNetwork(nn.Module):  # the NN with linear relu layers and one sigmoid in the end
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack_with_sigmoid = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack_with_sigmoid(x)
        return logits


class History:  # The history object to keep track of metrics during training and plot graphs to it.
    def __init__(self, val_set, train_set, model, **kwargs):  # kwargs are the metrics to keep track of.
        self.val_set = val_set
        self.train_set = train_set
        self.model = model
        self.kwargs = kwargs
        self.history = {'steps': []}
        for i in kwargs.keys():
            self.history.update({'val_' + i: []})
            self.history.update({'tra_' + i: []})
        self.valloader = None
        self.trainloader = None

    def save(self, step):  # this function is called in the training loop to save the current state of the model.
        short_history = {}
        for i in self.kwargs.keys():
            short_history.update({'val_' + i: []})
            short_history.update({'tra_' + i: []})
        # generate two dataloader with each k entries from either the training or the validation data.
        k = 500
        short_train_set, waste = torch.utils.data.random_split(self.train_set, [k, len(self.train_set) - k])
        short_val_set, waste = torch.utils.data.random_split(self.val_set, [k, len(self.val_set) - k])
        self.valloader = DataLoader(dataset=short_val_set, batch_size=5, shuffle=True, num_workers=2)
        self.trainloader = DataLoader(dataset=short_train_set, batch_size=5, shuffle=True, num_workers=2)
        # iterate over both dataloaders simultaneously
        for i, ((val_in, val_label), (tra_in, tra_label)) in enumerate(zip(self.valloader, self.trainloader)):
            with torch.no_grad():
                self.model.eval()
                # predict outcomes for training and validation.
                val_pred = self.model(val_in)
                tra_pred = self.model(tra_in)
                for j in self.kwargs.keys():  # iterate over the metrics
                    # calculate metric and save to short history
                    if len(val_pred) > 1:
                        val_l = self.kwargs[j](val_pred, val_label).item()
                        tra_l = self.kwargs[j](tra_pred, tra_label).item()
                        short_history['val_' + j].append(val_l)
                        short_history['tra_' + j].append(tra_l)
                self.model.train()
        # iterate over metrics and save the average of the short history to the history.
        for i in self.kwargs.keys():
            self.history['val_' + i].append(sum(short_history['val_' + i]) / len(short_history['val_' + i]))
            self.history['tra_' + i].append(sum(short_history['tra_' + i]) / len(short_history['tra_' + i]))
        self.history['steps'].append(step)  # save steps for the x-axis

    # this function is called after training to generate graphs.
    # When path is given, the graphs are saved and plt.show() is not called.
    def plot(self, path=None):
        figures = []
        for i in self.kwargs.keys():  # iterate over the metrics and generate graphs for each.
            fig, ax = plt.subplots()
            ax.plot(self.history['steps'], self.history['val_' + i], 'b')
            ax.plot(self.history['steps'], self.history['tra_' + i], 'r')
            ax.set_title(i.upper())
            ax.set_ylabel(i)
            ax.set_xlabel('Epochs')
            figures.append(fig)
            if path is None:
                plt.show()  # depending on the setup the graphs might still be shown even without this function called.
            else:
                plt.savefig(f"{path}/{i}")
            plt.clf()
        return figures


# this function is copied from https://huggingface.co/sentence-transformers/all-roberta-large-v1
# it returns embedded versions of the sentences its passed.
def long_roberta(sentences):
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    # sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
    model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # test if this works with truncation=False

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


class DYNAMIC_BINARY_AI:
    def __init__(self, name):
        self.name = name
        self.running_loss = None
        self.optimizer = None
        self.dataloader = None
        self.model = None
        self.loss = None
        self.dataframe = None
        self.val_set = None
        self.train_set = None
        self.labels = None
        self.sentences = None
        self.embedded_data = None
        self.raw_data = None
        self.dataset = None

    # generates raw training data; prompt_nr*answer_nr samples are created
    def generate_training_data(self, true_prompt, false_prompt, prompt_nr=100, answer_nr=100, load=False):
        def ask_ai(prompt, nr):  # get nr of answers from a prompt. Prompt should end with '\n\n1.'.
            response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=1,
                                                max_tokens=10 * nr)
            response = '1.' + response['choices'][0]['text'] + '\n'
            li = []
            for i in range(nr):
                pos = response.find(str(i + 1))
                beg = pos + len(str(i + 1)) + 2
                end = response[beg:].find('\n')
                li.append(response[beg:beg + end])
            return li

        def gen_sentences(master_prompt):  # generates nr keywords to the prompt and 50*factor sentences to each
            prompt_variations = ask_ai(master_prompt, prompt_nr)
            sentences = []
            for i in prompt_variations:
                sentences.extend(ask_ai(prompt=f'Give me {answer_nr} possible responses to this prompt: "{i}"\n\n1.', nr=answer_nr))
                #with open('sentences_quicksave.json', 'w') as f:
                #    f.write(json.dumps({'nr': nr,'all_sentences': all_sentences, 'sentences': sentences}))
            return sentences

        '''
        nr = 0
        all_sentences = []
        if load and os.path.exists("sentences_quicksave.json"):  # Or folder, will return true or false
            with open('sentences_quicksave.json', 'r') as f:
                a = json.loads(f.read())
            all_sentences = a['all_sentences']
            sentences = a['sentences']
            nr = a['nr']
            if nr == 0:
        else:
        '''
        true_master_prompt = f'Prompt: "{true_prompt}"\nGive me {prompt_nr} variations of this prompt.\n\n1.'
        all_sentences = gen_sentences(master_prompt=true_master_prompt)
        false_master_prompt = f'Prompt: "{false_prompt}"\nGive me {prompt_nr} variations of this prompt.\n\n1.'
        all_sentences.extend(gen_sentences(master_prompt=false_master_prompt))
        labels = []
        for i in range(len(all_sentences)):
            if i < len(all_sentences) / 2:
                labels.append(True)
            else:
                labels.append(False)
        data = [all_sentences, labels]
        data = np.array(data).transpose()
        mapping = []
        uni = np.unique(data)
        for i in uni:
            mapping.append(np.where(data == i)[0][0])
        data = data[mapping[1:]]
        pd.DataFrame(data).to_csv(f"{self.name.replace(' ', '_')}_generated_data.csv", index=False,
                                  header=['sentences', 'labels'])
        self.raw_data = pd.read_csv(f"{self.name.replace(' ', '_')}_generated_data.csv")

    # embeds the raw_data
    def embed_data(self):
        def transpose(lst):
            return list(map(list, zip(*lst)))

        self.embedded_data = []
        k = 100
        for i in range(math.ceil(len(self.raw_data['sentences']) / k)):
            sentences = long_roberta(list(self.raw_data['sentences'][k * i:k * i + k]))
            labels = list(self.raw_data['labels'][k * i:k * i + k])
            self.embedded_data.extend(transpose([sentences, labels]))
            torch.save(self.embedded_data, f'embedded_data_{self.name}.pt')
            print(f'saved {i + 1} / {len(self.raw_data["sentences"]) / k}')
        self.to_dataset()

    # convert embedded data to a proper dataset
    def to_dataset(self):
        def transpose(lst):
            return list(map(list, zip(*lst)))

        self.embedded_data = transpose(self.embedded_data)
        self.labels = self.embedded_data[1]
        self.sentences = self.embedded_data[0]
        self.labels = [torch.tensor([1.]) if i else torch.tensor([0.]) for i in self.labels]
        self.dataset = CustomTopicDataset(self.sentences, self.labels)

    # analyse the data including: balance, common words (wordcloud), sample lengths, duplicates and null values
    def analyse_training_data(self):
        print('Analysing training data...')
        print('General information')
        self.raw_data.info()
        self.raw_data.groupby(['labels']).describe()
        print(f'Number of unique sentences: {self.raw_data["sentences"].nunique()}')
        duplicates = self.raw_data[self.raw_data.duplicated()]
        print(f'Number of duplicate rows:\n{len(duplicates)}')
        print(f'Check for null values:\n{self.raw_data.isnull().sum()}')
        sns.countplot(x=self.raw_data['labels'])  # plotting distribution for easier understanding
        print('The start of the dataset:')
        print(self.raw_data.head(3))

        print('A few random examples from the dataset:')
        # let's see how data is looklike
        random_index = random.randint(0, self.raw_data.shape[0] - 3)
        for row in self.raw_data[['sentences', 'labels']][random_index:random_index + 3].itertuples():
            _, text, label = row
            print(f'TEXT: {text}')
            print(f'LABEL: {label}')

        true_data = self.raw_data[self.raw_data['labels'] == 1]
        true_data = true_data['sentences']
        false_data = self.raw_data[self.raw_data['labels'] == 0]
        false_data = false_data['sentences']

        def wordcloud_draw(data, color, s):
            words = ' '.join(data)
            cleaned_word = " ".join([word for word in words.split() if (word != 'movie' and word != 'film')])
            wordcloud = WordCloud(stopwords=stopwords.words('english'), background_color=color, width=2500,
                                  height=2000).generate(cleaned_word)
            plt.imshow(wordcloud)
            plt.title(s)
            plt.axis('off')

        plt.figure(figsize=[20, 10])  # first value is to the side, second is height.

        plt.subplot(1, 2, 1)
        wordcloud_draw(true_data, 'white', 'Most-common words about the topic')

        plt.subplot(1, 2, 2)
        wordcloud_draw(false_data, 'white', 'Most-common words not about the topic')
        plt.show()  # end wordcloud

        self.raw_data['text_word_count'] = self.raw_data['sentences'].apply(lambda x: len(x.split()))

        plt.figure(figsize=(15, 10))  # plt.figure(figsize=(20, 3))
        plt.subplot(1, 1, 1)  # plt.subplot(1, 3, i + 1)
        sns.histplot(data=self.raw_data, x='text_word_count', hue='labels', bins=50)
        plt.title(f"Distirbution of Various word counts with respect to target")
        plt.tight_layout()
        plt.show()

    # this function trains a model and returns it as well as the history object of its training process.
    def train(self, epochs=10, lr=0.001, val_frac=0.1, batch_size=25, loss=nn.BCELoss()):
        # get_acc measures the accuracy and is passed as a metric to the history object.
        def get_acc(pred, target):
            pred_tag = torch.round(pred)

            correct_results_sum = (pred_tag == target).sum().float()
            acc = correct_results_sum / target.shape[0]
            acc = torch.round(acc * 100)

            return acc

        # generate validation dataset with the fraction of entries of the full set passed as val_frac
        val_len = int(round(len(self.dataset) * val_frac))
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset,
                                                                     [len(self.dataset) - val_len, val_len])
        self.dataloader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True)
        self.model = NeuralNetwork(self.dataset.shape)

        self.loss = loss  # the loss passed to this train function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # define metrics to be monitored by the history object during training.
        r2loss = R2Score()
        mseloss = nn.MSELoss()
        bceloss = nn.BCELoss()
        accuracy = get_acc

        history = History(self.val_set, self.train_set, self.model, r2loss=r2loss, mseloss=mseloss, accuracy=accuracy,
                          bceloss=bceloss)  # define history object

        # main training loop
        for epoch in range(epochs):
            self.running_loss = []
            print(f'Starting new epoch {epoch + 1}/{epochs}')
            for step, (inputs, labels) in enumerate(self.dataloader):
                y_pred = self.model(inputs)
                lo = self.loss(y_pred, labels)
                lo.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.running_loss.append(lo.item())
                if (step + 1) % math.floor(len(self.dataloader) / 5 + 2) == 0:  # if (step+1) % 100 == 0:
                    print(f'current loss:\t\t{sum(self.running_loss) / len(self.running_loss)}')
                    self.running_loss = []
                    history.save(epoch + step / len(self.dataloader))
                    # save current state of the model to history
        # generate folder with timestamp and save the model there.
        now = str(d.now().isoformat()).replace(':', 'I').replace('.', 'i').replace('-', '_')
        os.mkdir(f"model_{now}")
        torch.save(self.model, f"model_{now}/model.pt")
        print(f'Model saved to "model_{now}/model.pt"')
        history.plot(f"model_{now}")  # save graphs to the folder
        return history, self.model  # return history and model


# with this function you can pass custom sentences to the model
def try_model(model):
    a = input('Please enter your input sentence: ')
    a = long_roberta(a)
    pred = model(a)
    print(pred.item())
    print('Where 1 is the first, true prompt: ""\nand 0 is the second, false prompt: "".\n')


if __name__ == "__main__":
    ti = DYNAMIC_BINARY_AI('topic_identifier')
    true_prompt = 'Write a short question about biology.'
    false_prompt = 'Write a short factual statement about shakespeare.'
    # ti.generate_training_data(true_prompt, false_prompt, prompt_nr=2, answer_nr=3)
    ti.raw_data = pd.read_csv(f"pre_prepared_data/binary_bio_shake_generated_data.csv")
    print('ANALYSE DATA BEGINN')
    ti.analyse_training_data()
    print('ANALYSE DATA END')
    # ti.embed_data()
    ti.embedded_data = torch.load('embedded_data_binary_bio_shake.pt')
    history, model = ti.train(epochs=10, lr=0.0001, val_frac=0.1, batch_size=10, loss=nn.BCELoss())
    history.plot()
