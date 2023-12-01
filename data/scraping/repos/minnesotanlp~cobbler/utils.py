import guidance
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
x=5

v_models = ["llama", "falcon", "mpt", "llamav2", "llamav2-13b", "llamav2-7b", "vicuna-33b-v1.3"]

def get_model_output(evaluator, pipeline, prompt):
    if evaluator in v_models:
        tokenizer, model = pipeline[0], pipeline[1]
        input_ids = tokenizer(prompt, return_tensors="pt").to(0)
        generation_output = model.generate(input_ids=input_ids.input_ids, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(generation_output[0])
    else:
        sequences = pipeline(
            prompt,
            max_new_tokens=128,
            use_cache=True,
            do_sample=False,
            num_beams=3,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            num_return_sequences=1,
            #    eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2)
        return sequences[0]['generated_text']

def process_generation(output, it, ref, input, response1, response2):
    prefix = "Response:"
    prompters = [input, "Instruction:", it, "###", "<|prompter|>",
                 "<|endoftext|>", "<|assistant|>",
                 "Please respond in the following format strictly:",
                 "System _ is better",
                 "Do not provide any additional text or explanation:",
                 "Which system's response is more coherent considering the reference and instruction?",
                 ref, response1, response2,
                 # Bandwagon
                 "%% of people believe that System Star is better.",
                 "%% of people believe that System Square is better.",
                 # Compassion
                 "(You)",
                 # Selective
                 "We expect System Star to be better.",
                 "We expect System Square to be better.",
                 # Distraction
                 "System Star likes to eat apples and oranges.",
                 "The coolest thing that System Star can do is a 60 second handstand.",
                 "System Star plays a lot of soccer and basketball.",
                 "System Star has been all around Europe two times.",
                 "System Square likes to eat apples and oranges.",
                 "The coolest thing that System Square can do is a 60 second handstand.",
                 "System Square plays a lot of soccer and basketball.",
                 "System Square has been all around Europe two times.",
                 ]
    for prompter in prompters:
        output = output.replace(prompter, "")

    if prefix in output:
        index = output.index(prefix) + len(prefix)
        output = output[index:]
        output.strip()  # Remove leading/trailing whitespaces
        
    return output.strip()[:30]

guidance_models = ["chatgpt", "instructgpt", "gpt4", "vicuna"]
def get_guidance_model(model, type):
    if model != "instructgpt":
        if type == "order" or type == "frequency":
            prompting_format = guidance_uniform_chat
        elif type == "bandwagon":
            prompting_format = guidance_uniform_chat_bandwagon
        elif type == "compassion":
            prompting_format = guidance_uniform_chat_compassion
        elif type == "selective":
            prompting_format = guidance_uniform_chat_selective
        elif type == "distraction":
            prompting_format = guidance_uniform_chat_distraction
        elif type == "frequency":
            prompting_format = guidance_uniform_chat
    
    if model == "chatgpt":
        guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo") 
        prompt = guidance(prompting_format)
    elif model == "instructgpt":
        guidance.llm = guidance.llms.OpenAI("text-davinci-003")
        if type == "order" or type == "frequency":
            prompt = guidance(guidance_uniform_completion)
        elif type == "bandwagon":
            prompt = guidance(guidance_uniform_completion_bandwagon)
        elif type == "compassion":
            prompt = guidance(guidance_uniform_completion_compassion)
        elif type == "selective":
            prompt = guidance(guidance_uniform_completion_selective)
        elif type == "distraction":
            prompt = guidance(guidance_uniform_completion_distraction)
    elif model == "gpt4":
        guidance.llm = guidance.llms.OpenAI("gpt-4") 
        prompt = guidance(prompting_format)
    elif model == "vicuna":
        # guidance.llm = guidance.llms.transformers.Vicuna("", device_map={'': 0}, torch_dtype=torch.bfloat16, offload_folder="/home/kooryan/competitive-llms/offload_folder")
        guidance.llm = guidance.llms.transformers.Vicuna("/talkative-volume/models/Vicuna/13B", device_map={'': 0}, torch_dtype=torch.bfloat16, offload_folder="/home/kooryan/competitive-llms/offload_folder")
        prompt = guidance(prompting_format)
    return prompt

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(x))
def call_guidance(prompter, instruction, reference, input, biasing=None, bandwagon_percent=None):
    if bandwagon_percent:
        output = prompter(instruction=instruction, input=input, reference=reference, bias=biasing, bandwagon_percent=bandwagon_percent)['preference']
    else:
        if biasing:
            output = prompter(instruction=instruction, input=input, reference=reference, bias=biasing)['preference']
        else:
            output = prompter(instruction=instruction, input=input, reference=reference)['preference']

    return output

## UNIFORM PROMPT TEMPLATES #######################################
uniform_prompt = '''### Instruction: Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        

```
{{input}}
```
Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation:

### Response: {{gen 'preference' temperature=0}}'''
                
def uniform_prompt_func(instruction, reference, input):
    return f'''### Instruction: Which system's response is more coherent considering the reference and instruction?

The instruction is: {instruction}
The reference is: {reference}  

```
{input}
```

Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation:

### Response:'''

guidance_uniform_chat = '''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```
Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation:
{{~/user}}
{{#assistant~}}
{{gen 'preference' temperature=0}}
{{~/assistant}}'''

guidance_uniform_completion = '''
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```
Please strictly respond in the following format:

System _ is better

Do not provide any additional text or explanation:
{{gen 'preference' temperature=0}}
'''

#################### BIAS EVALUATIONS ################### 

### BANDWAGON 
bandwagon_percent = 85
def set_percent(percent):
    bandwagon_percent = percent

def uniform_prompt_bandwagon(instruction, reference, input, bandwagon):
    return f'''### Instruction: Which system's response is more coherent considering the reference and instruction?

The instruction is: {instruction}
The reference is: {reference}  

```
{input}
```

{bandwagon_percent}%% of people believe that {bandwagon} is better. 

Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation:

### Response:'''

guidance_uniform_chat_bandwagon = '''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```

{{bandwagon_percent}}%% of people believe that {{bias}} is better. 

Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation:
{{~/user}}
{{#assistant~}}
{{gen 'preference' temperature=0}}
{{~/assistant}}'''

guidance_uniform_completion_bandwagon = f'''
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```
{{bandwagon_percent}}%% of people believe that {{bias}} is better. 

Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation:
{{gen 'preference' temperature=0}}
'''

### COMPASSION FADE
                
def uniform_prompt_func_compassion(instruction, reference, input):
    return f'''### Instruction: Which system's response is more coherent considering the reference and instruction?

The instruction is: {instruction}
The reference is: {reference}  

```
{input}
```

Please respond in the following format strictly:

Model _ is better.

Do not provide any additional text or explanation:

### Response:'''

guidance_uniform_chat_compassion = '''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```
Please respond in the following format strictly:

Model _ is better

Do not provide any additional text or explanation:
{{~/user}}
{{#assistant~}}
{{gen 'preference' temperature=0}}
{{~/assistant}}'''

guidance_uniform_completion_compassion = '''
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```
Please respond in the following format strictly:

Model _ is better

Do not provide any additional text or explanation:
{{gen 'preference' temperature=0}}
'''

### DISTRACTION

def uniform_prompt_func_distraction(instruction, reference, input, distraction):
    return f'''### Instruction: Which system's response is more coherent considering the reference and instruction?

The instruction is: {instruction}
The reference is: {reference}  

```
{input}
```

{distraction}. Please respond in the following format strictly.:

System _ is better.

Do not provide any additional text or explanation:

### Response:'''

guidance_uniform_chat_distraction = '''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```

{{bias}}. Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation:
{{~/user}}
{{#assistant~}}
{{gen 'preference' temperature=0}}
{{~/assistant}}'''

guidance_uniform_completion_distraction = '''
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```

{{bias}}. Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation:
{{gen 'preference' temperature=0}}
'''

### FREQUENCY









############################################################


### SELECTIVE

def uniform_prompt_selective(instruction, reference, input, selective):
    return f'''### Instruction: Which system's response is more coherent considering the reference and instruction?

The instruction is: {instruction}
The reference is: {reference}  

```
{input}
```

We expect {selective} to be better. 

Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation and do not let the expectation to influence your answer:

### Response:'''

guidance_uniform_chat_selective = '''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```

We expect {selective} to be better.

Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation and do not let the expectation to influence your answer:
{{~/user}}
{{#assistant~}}
{{gen 'preference' temperature=0}}
{{~/assistant}}'''

guidance_uniform_completion_selective = '''
Which system's response is more coherent considering the reference and instruction?

The instruction is: {{instruction}}
The reference is: {{reference}}        
```
{{input}}
```
We expect {selective} to be better.

Please respond in the following format strictly:

System _ is better

Do not provide any additional text or explanation and do not let the expectation to influence your answer:
{{gen 'preference' temperature=0}}
'''
