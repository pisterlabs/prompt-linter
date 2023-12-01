import math
import torch

#https://github.com/huggingface/transformers/issues/473
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


a=['Tom loves reading books.He prefers reading books at library.So he always goes to library',
   'Tom loves reading books.He missed his lunch today.So he always goes to library.',
   'Tom loves reading books. Patatoes, boil them mash them put them in a stew. So he always goes to library.',
   'I went to the cinema. In the cinema, I watched a movie. After the movie, I went home.',
   'I went to the cinema. I like me some dogs. After the movie, I went home.']

print('Scores - Higher is less likely')
print([score(i) for i in a])
