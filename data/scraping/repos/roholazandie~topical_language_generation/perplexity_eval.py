import math
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import torch


with torch.no_grad():
    # Load pre-trained model (weights)
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss, _ = model(tensor_input, labels=tensor_input)
    return math.exp(loss.item())


a = ['there is a book on the desk',
    'there is a plane on the desk',
   'there is a book in the desk']

print([score(i) for i in a])