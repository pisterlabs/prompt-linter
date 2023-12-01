import time
import json
import os
import random
import re
import string
from functools import partial
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
from multiprocessing import Pool
import openai
import io
import numpy as np
import tqdm
from rouge_score import rouge_scorer


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = ""
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        # Check if input is a string before calling .lower()
        input = "<noinput>" if isinstance(input, float) or input.strip() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt

def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def post_process_gpt4_response(num_prompt_instructions, response):
    if response is None:
        return []
    print(response)
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords.
        blacklist = [
            "image",
            "images",
            "chart",
            "charts",
            "photo",
            "photos",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]


        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue

        if inst.startswith("Write a program"):
            continue

        if inst[0] in string.punctuation:
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions

def read_prompt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError as e:
        print(f"Error reading 'prompt.txt': {e}. Skipping the problematic characters.")
        with open(file_path, 'rb') as file:
            content = file.read()
        content = content.decode('utf-8', errors='ignore')
        return content

def read_large_context(file_path, num_splits):
    
    context = read_prompt_file(file_path=file_path)
    context_splits = [context[i:i + len(context) // num_splits] for i in range(0, len(context), len(context) // num_splits)]
    return context_splits

def generate_instruction_following_data(
    api_key: str,
    output_dir="./",
    seed_tasks_path="./seed_tasks.json",
    num_instructions_to_generate=100,
    model_name="gpt-4",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=0.1,
    max_tokens=2000,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=0.75,
    num_cpus=16,
    context_split=1800
):
    print("We are at the beginning of the code now!")
    # Load JSON data from a file
    with open(seed_tasks_path, 'r') as f:
        seed_tasks = json.load(f)

    # Transforms the data
        seed_instruction_data = [
            {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
            for t in seed_tasks
        ]

    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 390 #38 # 37 # 9 # 8 # 2 # 0
    # loads the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen3.json")):
        machine_instruction_data = jload(os.path.join(output_dir, "regen3.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # generates new instructions
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # tokenizes all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    context = read_large_context('./output.txt', context_split)
    
    prompt_tmp_txt = read_prompt_file("./prompt.txt") + "\n"

    
    # Initialize the OpenAI model
    model = ChatOpenAI(
        openai_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        request_timeout=180
    )
    print(f'This is the request idx {request_idx}')

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1
        
        # print(f'This is the request idx {request_idx}')
        # break
    
        results = []
        request_start = time.time()
        for _ in range(request_batch_size):
            #  sampling from the seed tasks and the random context
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            selected_context = context[request_idx-1]
                        
            prompt = encode_prompt(prompt_instructions)
            prompt_template = PromptTemplate(template=prompt_tmp_txt, input_variables=["ins_number", "input", "selected_context"])
            try:
                llm_chain = LLMChain(prompt=prompt_template, llm=model)
            except:
                print('Sleeping for 10 seconds...')
                time.sleep(10)
                llm_chain = LLMChain(prompt=prompt_template, llm=model)

            # Sleeps when timeout
            try:
                result = llm_chain.predict(ins_number=num_prompt_instructions, input=prompt, selected_context=selected_context)
            except:
                time.sleep(10)
                print('Sleeping for 10 seconds...')
                print('Skipping')
                continue
                result = llm_chain.predict(ins_number=num_prompt_instructions, input=prompt, selected_context=selected_context)
           
            results.append(result)

        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            try:
                new_instructions = post_process_gpt4_response(num_prompt_instructions, result)
                instruction_data += new_instructions
            except(UnicodeEncodeError):
                instruction_data += new_instructions
                continue
            

        total = len(instruction_data)
        print(f"We have reached here and the total instructions thus far is {total}")
        keep = 0
        for instruction_data_entry in instruction_data:
            print("I have entered the for loop for instruction_data_entry!")
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])            
            print("Added this instruction data entry")
            keep += 1
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
            
        print(f"I have come out of the for loop and I have kept {keep}")
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        jdump(machine_instruction_data, os.path.join(output_dir, "regen3.json"))
        
        
openai.api_key ='INSERT-KEY-HERE'
os.environ['OPENAI_API_KEY'] = openai.api_key

generate_instruction_following_data(
    api_key=openai.api_key,
    output_dir="./new_tasks",
    seed_tasks_path="./seed_tasks.json",
    num_instructions_to_generate=10000, ###
    model_name="gpt-3.5-turbo",
    num_prompt_instructions=3,
    request_batch_size=1, ###
    temperature=0,
    top_p=1.0,
    num_cpus=1
)