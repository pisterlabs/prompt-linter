import openai
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Set up OpenAI API credentials
openai.api_key = "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"

# Define the name of your custom model and its training parameters
model_name = "gpt-3.5-turbo"
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=5000,
    save_total_limit=2,
)

# Define the tokenizer and pre-trained GPT-3 model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Load your custom training dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="C:\Davidlocal\PMD_AI_TRANNING2\exampleconversation2.txt",
    block_size=128
)

# Prepare the data for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Train the model using the OpenAI API
model_url = f"https://api.openai.com/v1/models/{model_name}/fine_tunes"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai.api_key}"}
body = {"model": {"architecture": "text-davinci-002", "max_batch_size": 1}, "data": {"text": dataset}}
r = requests.post(model_url, headers=headers, json=body)
model_id = r.json()["model"]
model_url = f"https://api.openai.com/v1/models/{model_id}"
print(model_id)

# Load the trained model
model = AutoModelForCausalLM.from_pretrained(model_url)


# Generate text using your fine-tuned model
prompt = "Give me a recipe of Espresso Martini and tell me how to make it."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=500, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
