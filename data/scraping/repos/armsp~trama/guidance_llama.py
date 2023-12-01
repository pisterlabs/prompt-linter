import guidance
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# replace your_path with a version of the LLaMA model
model_path = "/cluster/work/lawecon/Work/raj/llama2models/13b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(
    model_path,
    # use_auth_token=True,
)
# guidance.llm = guidance.llms.transformers.LLaMA(model_path, device="cpu", temperature=0.5) # use cuda for GPU if you have >27GB of VRAM
guidance.llm = guidance.llms.transformers.LLaMAChat(model_path, tokenizer=tokenizer, device_map="auto", token_healing=True, torch_dtype=torch.bfloat16,  load_in_8bit=False,
    load_in_4bit=False, temperature=0.7)
# define the few shot examples
# examples = [
#     {'input': 'I wrote about shakespeare',
#     'entities': [{'entity': 'I', 'time': 'present'}, {'entity': 'Shakespeare', 'time': '16th century'}],
#     'reasoning': 'I can write about Shakespeare because he lived in the past with respect to me.',
#     'answer': 'No'},
#     {'input': 'Shakespeare wrote about me',
#     'entities': [{'entity': 'Shakespeare', 'time': '16th century'}, {'entity': 'I', 'time': 'present'}],
#     'reasoning': 'Shakespeare cannot have written about me, because he died before I was born',
#     'answer': 'Yes'},
#     {'input': 'A Roman emperor patted me on the back',
#     'entities': [{'entity': 'Roman emperor', 'time': '1st-5th century'}, {'entity': 'I', 'time': 'present'}],
#     'reasoning': 'A Roman emperor cannot have patted me on the back, because he died before I was born',
#     'answer': 'Yes'}
# ]

# # define the guidance program
# structure_prompt = guidance(
# '''How to solve anachronism problems:
# Below we demonstrate how to test for an anachronism (i.e. whether it could have happened or not based on the time periods associated with the entities).
# ----

# {{~! display the few-shot examples ~}}
# {{~#each examples}}
# Sentence: {{this.input}}
# Entities and dates:{{#each this.entities}}
# {{this.entity}}: {{this.time}}{{/each}}
# Reasoning: {{this.reasoning}}
# Anachronism: {{this.answer}}
# ---
# {{~/each}}

# {{~! place the real question at the end }}
# Sentence: {{input}}
# Entities and dates:{{#geneach 'entities' stop="\\nReasoning:"}}
# {{gen 'this.entity' stop=":"}}: {{gen 'this.time' stop="\\n"}}{{/geneach}}
# Reasoning:{{gen "reasoning" stop="\\n"}}
# Anachronism: {{#select "answer"}}Yes{{or}}No{{/select}}''')

# out = structure_prompt(examples=examples, input='The T-rex bit my dog')

# # the entities generated are in the output
# print(out['entities'])
# # ...as is the reasoning
# print(out["reasoning"])
# # ...and the answer
# print(out["answer"])
# print("######################################")
# print(out)



# experts = guidance('''
# {{#system~}}
# You are a helpful and terse assistant.
# {{~/system}}

# {{#user~}}
# I want a response to the following question:
# {{query}}
# Name 3 world-class experts (past or present) who would be great at answering this?
# {{~/user}}

# {{#assistant~}}
# {{gen 'expert_names' temperature=0.6 max_tokens=300}}
# {{~/assistant}}
# ''')

# experts(query='How can I be more productive?')
# print(experts)

experts = guidance('''
{{#system~}}
You are a helpful and terse assistant.
{{~/system}}

{{#user~}}
I want a response to the following question:
{{query}}
Name 3 world-class experts (past or present) who would be great at answering this?
Don't answer the question yet.
{{~/user}}

{{#assistant~}}
{{gen 'expert_names' temperature=0.7 max_tokens=300}}
{{~/assistant}}

{{#user~}}
Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0.7 max_tokens=500}}
{{~/assistant}}''')
print(experts(query='How can I be more productive?', caching=False))

experts = guidance(
'''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
I want a response to the following question:
{{query}}
Who are 3 world-class experts (past or present) who would be great at answering this?
Please don't answer the question or comment on it yet.
{{~/user}}
{{#assistant~}}
{{gen 'experts' temperature=0.6 max_tokens=300}}
{{~/assistant}}
{{#user~}}
Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
In other words, their identity is not revealed, nor is the fact that there is a panel of experts answering the question.
If the experts would disagree, just present their different positions as alternatives in the answer itself (e.g. 'some might argue... others might argue...').
Please start your answer with ANSWER:
{{~/user}}
{{#assistant~}}
{{gen 'answer' temperature=0.6 max_tokens=500}}
{{~/assistant}}''')
print(experts(query='What is the meaning of life?'))

# # we can pre-define valid option sets
# valid_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]

# # define the prompt
# program = guidance("""The following is a character profile for an RPG game in JSON format.
# ```json
# {
#     "description": "{{description}}",
#     "name": "{{gen 'name' temperature=0.6}}",
#     "age": {{gen 'age' pattern='[0-9]+' stop=',' temperature=0.6}},
#     "armor": "{{#select 'armor'}}leather{{or}}chainmail{{or}}plate{{/select}}",
#     "weapon": "{{select 'weapon' options=valid_weapons}}",
#     "class": "{{gen 'class' temperature=0.6}}",
#     "mantra": "{{gen 'mantra' temperature=0.6}}",
#     "strength": {{gen 'strength' pattern='[0-9]+' stop=',' temperature=0.6}},
#     "items": [{{#geneach 'items' num_iterations=3 }}
#         "{{gen 'this' temperature=0.6}}",{{/geneach}}
#     ]
# }```""")

# # execute the prompt
# out = program(description="A quick and nimble fighter.", valid_weapons=valid_weapons)
# # we have now generated a valid json string that is mixure of generated and inserted text
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(str(out).split("```json")[1][:-3])
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(out.variables())
executed_program = guidance("""Write a story about a person and use \{{character_name}} whenever you need to write their name:
STORY
{{parse (gen max_tokens=100 temperature=0.7) name="story"}}""", character_name="Jill")(silent=True)

print(executed_program["story"])


program = guidance('''
{{#system}}You are an expert unix systems admin.{{/system}}

{{#user~}}
What are the most common commands used in the {{os}} operating system?
{{~/user}}

{{#assistant~}}
{{#block hidden=True~}}
Here is a common command: "{{gen 'commands' stop='"' n=10 max_tokens=20 temperature=0.7}}"
{{~/block~}}

{{#each (unique commands)}}
{{@index}}. {{this}}
{{~/each}}

Perhaps the most useful command from that list is: "{{gen 'cool_command' temperature=0.7}}", because{{gen 'cool_command_desc' max_tokens=100 stop="\\n" temperature=0.7}}
On a scale of 1-10, it has a coolness factor of: {{gen 'coolness' pattern="[0-9]+" temperature=0.7}}.
{{~/assistant}}
''')
out = program(os="Linux", unique=lambda x: list(set(x)), caching=False)
print(out)