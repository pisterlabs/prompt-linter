#!/usr/bin/env python3

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread
import guidance
import json
import gc
import tempfile

def load_llm():
    model_id = "Trelis/Llama-2-7b-chat-hf-sharded-bf16-5GB" # sharded model by RonanKMcGovern. Change the model here to load something else.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.init_device = 'cuda:0' # Unclear whether this really helps a lot or interacts with device_map.

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, quantization_config=bnb_config, device_map='auto', trust_remote_code=True) # for inference use 'auto', for training us device_map={"":0}
    
    #load LoRA
    #model.load_adapter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model,tokenizer

def get_char_list(script,model,tokenizer):
   # model,tokenizer = load_llm()
    guidance.llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer)
    characters = guidance("""[INST]
<<SYS>>
You are a bot that reads a story and returns a list of the characters in the story.
<</SYS>>
Create list of the names of the characters in the story below.
Story - {{story}}
[/INST]
Here is the requested list of characters present in the story-
[{{gen 'characters' max_tokens = 300 temperature = 0.2 stop = ']' }}]""")
    out = characters(story = script).variables()['characters']
    

    # initializing bad_chars_list
    bad_chars = ["'", '_', '!', "*"]
    out = ''.join((filter(lambda i: i not in bad_chars, 
                              out)))

    charList = out.split(",")


    

    #freeing memory
    model,tokenizer = None,None
    guidance.llm=None
    del characters
    gc.collect()
    torch.cuda.empty_cache()

    #return charList
    return out
    

def generate_with_guidance(script,model,tokenizer):
    guidance.llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer)
    prompt = """
    [INST]
    <<SYS>>\n You are a bot that reads a story and returns a JSON file containing the actions performed throughout the story along with the environmented they take place in divided by scene.
    Divide the scenes based on the change in environment.
    Make sure you stop creating when you conclude the story. Do not make up your own story.
    Always use the names of the characters when you mention them, avoid any ambiguity.<</SYS>>\n
    Create a JSON file containing the description of dividing the story into scenes, and each scene containing the actions being performed throughout the story(which actors are performing what actions, what are their reactions etc.), along with the environment name(City, Location etc.). 
    Divide the scenes based on the change in environment.
    Always use the names of the characters, to avoid ambiguity, describing all the characters that are taking part in any particular action.
    {{#block hidden = True}}
    This is the example format of the JSON-
    {"Scene 1": {
        "Actions": ["John was sitting at the table",
                    "John was having a chat with Smith",
                    "Smith was jumping up and down the table"
                    ],
        "Env" : "Dining Room"
        },
        "Scene 2": {
        "Actions": ["John was driving his car",
                    "John met with an accident",
                    "Smith in the passenger seat flew out of the car"
                    ],
        "Env" : "City"
        },
    }
    {{/block}}
    Create the JSON for this Story = {{story}} 
    [/INST]
    Here is the JSON file in the requested format-
    ```json
    {
    {{#geneach 'scenes' num_iterations = """ + str(3) + """ }}
        "Scene {{@index+1}}": {"Actions":["{{gen 'actions' max_tokens = 300 temperature = 0.2 stop = ']'  }} ], "Env": "{{gen 'location name' max_tokens = 10 temperature = 0.00001 stop='"'}}" } ,
    {{/geneach}}
    }
    ```
    """
    
    program = guidance(prompt)
    out = program(story = script)
    out = str(out).split("```json")[1]
    out = str(out).split("```")[0]
    print(out)
    out = out[0:len(out)-1-5].rstrip()
    out = out[:len(out)-1-1] + "\n}"
    #out = out[0:-1] + "\n}"

    #freeing memory
    model,tokenizer = None,None
    del program 
    guidance.llm=None
    gc.collect()
    torch.cuda.empty_cache()

    return out
    
def generate(script):
    model, tokenizer = load_llm()
#     encoding = tokenizer(purpose, return_tensors="pt").to("cuda:0")
#     output = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask, max_new_tokens=4096, do_sample=True, temperature=0.000001, eos_token_id=tokenizer.eos_token_id, top_k = 0)
#     return tokenizer.decode(output[0], skip_prompt = True,skip_special_tokens=True)

    #USING GUIDANCE
    out=generate_with_guidance(script = script, model = model, tokenizer = tokenizer)
    
    #clearing memory
    model,tokenizer = None,None
    guidance.llm=None
    gc.collect()
    torch.cuda.empty_cache()
    
    return out
def sanityCheck(JSONDict: dict):
    #function to check if JSON is correct, written at end after format is finalized
    return True

def createJSON(script : str):
    jsonString = generate(script)
    print(jsonString)
    JSONDict = json.loads(jsonString)
    
    #Function to Perform Sanity check on JSON
#     try:
#         sanityCheck(JSONDict)
#     except ValueError:
#         print("Unfortunately, JSON Formatted Data isn't accurate")
#     except Exception as e:
#         #generic error print
#         print(str(e))
    
    
    gc.collect()
    torch.cuda.empty_cache()
    return JSONDict

def main():
    model,tokenizer = load_llm()
    script = ""
    with open('./temp/script.txt','r') as file:
        script = file.read()
        file.close()
    charList = get_char_list(script,model,tokenizer)
    jsonDict = generate_with_guidance(script,model,tokenizer)
  
    with open('./temp/charList.txt','w') as file:
        file.write(charList)
        file.close()
    with open('./temp/preppedJSON.json', 'w') as file:
        file.write(str(jsonDict))
        file.close()
    print('Script Analysed!')
if __name__=="__main__":
    main()
