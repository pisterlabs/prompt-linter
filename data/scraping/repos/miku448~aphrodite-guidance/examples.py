from transformers import AutoTokenizer
import guidance
from aphrodite_client.openai import AphroditeOpenAIClient
from aphrodite_client.utils import print_example

# Set up model to use
model_name = "mistralai/Mistral-7B-v0.1"

# Create tokenizer for model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = None
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
tokenizer.special_tokens_map.values()

# Create Aphrodite client
llm = AphroditeOpenAIClient(
    model_name,
    api_base="http://localhost:2242/v1", # Aphrodite endpoint
    api_key="sk-EMPTY", # Use an api key defined in aphrodite's --api-keys argument
    tokenizer=tokenizer,
    chat_mode=False,
    rest_call=False,
    temperature=0.9,
    caching=False
)

# Use aphrodite client for guidance
guidance.llm = llm

# EXAMPLE 1: Simple generation (2-shot)
prompt = guidance('''Common sense question and answer, with short answers
Question: What is your favorite food?
Answer: "Sushi."
Question: What is your favorite color?
Answer: "Blue."
Question: What is your favorite animal?
Answer: "{{gen "response" stop='"'}}"''') ## TODO: It we put an space after "Answer:" it doesn't work. Need to fix this.
result = prompt()
print_example('Example 1', prompt.text, result.variables())

# EXAMPLE 2
prompt = guidance('''I will show you an email and a response, and you will tell me if it's offensive.
Email: {{email}}.
Response: I also don't{{gen "response" stop="."}}
Is the response above offensive in any way? Please answer with a single word, either "Yes" or "No".
Answer:{{#select "answer" logprobs='logprobs'}} Yes{{or}} No{{/select}}''')
result = prompt(email='I hate tacos')
print_example('Example 2', prompt.text, result.variables())

# EXAMPLE 3
prompt = guidance("""Common sense question and answer
Question: What is the besto waifu in domekano?
Answer: {{#select "waifu"}} Hina{{or}} Rui{{or}} Momo{{/select}}""")
result = prompt()
print_example('Example 3', prompt.text, result.variables())

# EXAMPLE 4
prompt = guidance("""RPG Game Character specification
  {
    "name": "{{name}}",
    "job": "{{gen 'job' stop='",'}}",
    "armor": "{{#select 'armor'}}silver{{or}}leather{{/select}}",
    "weapon": "{{select 'weapon' options=valid_weapons}}"
  }
""")
result = prompt(name="Rudeus the silver", valid_weapons=["axe", "mace", "spear", "sword", "bow", "crossbow"])
print_example('Example 4', prompt.text, result.variables())
