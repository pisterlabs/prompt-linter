import math
import torch
import pandas as pd
### !pip install pytorch-pretrained-bert
### !pip install spacy
### !pip instal ftfy
### python3 -m spacy download en
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
# Load pre-trained model (weights)

data = pd.read_csv('/media/daniel/Python/Projects/CommonLit/data/train.csv')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)

data['perplexity_gpt'] = data['excerpt'].apply(score)

print(data.head())

