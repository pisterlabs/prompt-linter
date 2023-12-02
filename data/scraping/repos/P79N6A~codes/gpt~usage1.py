import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import logging

logging.basicConfig(level=logging.INFO)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model (weights)
model = OpenAIGPTModel.from_pretrained('openai-gpt')
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    hidden_states = model(tokens_tensor)

# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()

# If you have a GPU, put everything on cuda

# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor)

# get the predicted last token
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)