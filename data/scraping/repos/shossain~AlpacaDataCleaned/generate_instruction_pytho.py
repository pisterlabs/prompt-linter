"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction_pytho generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="gpt-3.5-turbo" \
  --client="claude" or "openai"  \
"""
import json
import os
import random
import re
import string
import time
from functools import partial
from multiprocessing import Pool

import fire
import numpy as np
import tqdm
import utils
from rouge_score import rouge_scorer
from PyPDF2 import PdfReader
import anthropic


def encode_prompt(prompt_instructions, prompt_path="./prompt_pytho.txt"):
    """Encode multiple prompt instructions into a single string."""
    prompt = open().read(prompt_path) + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = (
            task_dict["instruction"],
            task_dict["input"],
            task_dict["output"],
        )
        
        prompt += f"###\n"
        prompt += f"Instruction: {instruction}\n\n"
        prompt += f"Input: {input}\n\n"
        prompt += f"Output: {output}\n\n"

    prompt += f"###\n\n\n"
    prompt += f"Now, generate a scenario:"
    
    return prompt

def encode_prompt_claude(prompt_instructions, prompt_path):
    """Encode multiple prompt instructions into a single string."""
    prompt = f"Here are sample military training scenarios in <scenario> tags:\n\n"

    for idx, task_dict in enumerate(prompt_instructions):
        prompt += "<scenario>\n\n"

        (instruction, input, output) = (
            task_dict["instruction"],
            task_dict["input"],
            task_dict["output"],
        )
        
        prompt += f"Instruction: {instruction}\n\n"
        prompt += f"Input: {input}\n\n"
        prompt += f"Output: {output}\n\n"

        prompt += f"</scenario>\n\n"

    prompt += open(prompt_path).read()
    
    return prompt

def encode_prompt_claude_question(prompt_context, prompt_path):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_path).read()
    
    prompt += f"""
<context>
{prompt_context}
</context>

Now, generate the questions.
"""
    
    return prompt

def encode_prompt_claude_answer(question, prompt_path):
    """Encode multiple prompt instructions into a single string."""
    context_dict =  utils.get_context(question)
    # print(context_dict["context_log"])
    # prompt = f"""Here is a set of contexts for the essay you are going to write next.
    prompt = f"""Here is a set of contexts for the question I am going to ask next.
{context_dict['context']}

{open(prompt_path).read()}{question}
"""    
    return (prompt, context_dict['context'])

def post_process_gpt3_response(num_prompt_instructions, response):
    splitted_data = re.split(
        f"(Instruction|Input|Output):", response
    )
    
    # print(splitted_data)

    inst = splitted_data[2].strip()
    input = splitted_data[4].strip()
    output = splitted_data[6].strip()

    return [
        {"instruction": inst, "input": input, "output": output}
    ]


def post_process_claude_response(num_prompt_instructions, response):
    splitted_data = re.split(
        f"<scenario>|(Instruction|Input|Output):|</scenario>", response
    )
    
    # print(splitted_data)

    inst = splitted_data[4].strip()
    input = splitted_data[6].strip()
    output = splitted_data[8].strip()

    return [
        {"instruction": inst, "input": input, "output": output}
    ]

def post_process_claude_question(num_prompt_instructions, response):
    matches = re.findall(f"<question>(.*)</question>", response)

    result = []
    for m in matches:
        result.append(m.strip())

    return result

def post_process_claude_answer(question, response, context):
    return [
        {"question": question, "answer": response.strip(), "context": context}
    ]

def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def get_context(file_path, page_index, page_num, file_index):
    context = ""

    with open(file_path, 'rb') as infile:
        try:
            reader = PdfReader(infile)
            print(f'\n{file_path}, pages: {len(reader.pages)} file_index: {file_index}\n') 
            page_processed = 0

            # Exhausted the file
            if (page_index + page_num) >= len(reader.pages):
                return (None, None)

            for page in reader.pages[page_index : (page_index + page_num)]:            
                page_content = page.extract_text().strip()

                if (page_content.startswith("References") 
                    or page_content.startswith("Index")
                    or page_content.startswith("Source Notes")
                    or page_content.startswith("Glossary")):
                    #or page_content.startswith("Notes")):

                    return (None, None)

                # large page and not a menu
                if len(page_content) > 600 and page_content.count('.........................') < 5:   
                    page_processed = page_processed + 1
                    context += f"\n{page_content}"
                                

            print(f"\n\nProcessed: {page_processed}\n\n")
            
        except Exception as e:
            print("File: ", file_path)
            print("The error is: ", e)



    page_index += page_num

    return (context, page_index)

def get_input(input_data_path, file_index=-1, page_index=-1, file_paths=[], page_num=10):

    if file_index == -1:
        for entry in os.listdir(input_data_path):
            file_path = os.path.join(input_data_path, entry)
            if os.path.isfile(file_path):
                file_paths.append(file_path)

        file_paths.sort()
        return {
            "file_index": 0,
            "page_index": 0,
            "file_paths": file_paths,
        }
    
    context = ""

    context, page_index = get_context(file_paths[file_index], page_index, page_num, file_index)

    # move to the next file
    if not context:
        file_index += 1
        if file_index == len(file_paths):
            return None
           
        context, page_index = get_context(file_paths[file_index], 0, page_num, file_index)

    if not context:
        context, page_index = ("", 0)       
    
    return {
        "file_index": file_index,
        "page_index": page_index,
        "file_paths": file_paths,
        "context": context,
    }


def generate_instruction_following_data(
    client,
    prompt_path="./prompt_pytho_claude.txt",
    input_data_path="../data/pdfs_copy",
    output_dir="../alpaca-data",
    seed_tasks_path="./seed_tasks_pytho.jsonl",
    is_question=False,
    is_answer=False,
    question_path_index = -1,
    num_instructions_to_generate=3,
    output_batch_size=100,
    model_name="claude-2.0",
    start_file_index=0,
    num_prompt_instructions=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=8,
):
    run_start = time.time()
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {
            "instruction": t["instruction"],
            "input": t["input"],
            "output": t["output"],
        }
        for t in seed_tasks
    ]
    print(
        f"Loaded {len(seed_instruction_data)} human-written seed instructions"
    )

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    # if os.path.exists(os.path.join(output_dir, "regen.json")):
    #     machine_instruction_data = utils.jload(
    #         os.path.join(output_dir, "regen.json")
    #     )
    #     print(
    #         f"Loaded {len(machine_instruction_data)} machine-generated instructions"
    #     )

    # similarities = {}
    # scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    
    if is_question:
        input_result = get_input(input_data_path)
        input_result["file_index"] = start_file_index

    if is_answer:
        question_paths = os.listdir(input_data_path)
        question_paths.sort()
        questions = []
    
    # for questions, exhaust the input set
    total_generated_results = 0
    batch_total = 0
    batch_id = 0
    machine_instruction_data = []
    

    while total_generated_results < num_instructions_to_generate or is_question or is_answer:
        request_idx += 1
        if is_answer and len(questions) == 0:
            question_path_index += 1
            if question_path_index >= len(question_paths):
                return
            
            question_file = os.path.join(input_data_path, question_paths[question_path_index])
            questions = json.loads(open(question_file, "rb").read())
            print(f"""question_path_index: {question_path_index} question_file: {question_file}""")
            
        
        batch_inputs = []
        # only sampling from the seed tasks
        prompt_instructions = random.sample(
            seed_instruction_data, num_prompt_instructions
        )
        if client == 'openai':
            prompt = encode_prompt(prompt_instructions)
        else:
            if is_question:
                input_result = get_input(
                    input_data_path, input_result["file_index"], input_result["page_index"], input_result["file_paths"])
                if not input_result:
                    return
                prompt = encode_prompt_claude_question(input_result["context"], prompt_path)
            elif is_answer:
                question = questions.pop(0)
                prompt, context = encode_prompt_claude_answer(question, prompt_path)
            else:
                prompt = encode_prompt_claude(prompt_instructions, prompt_path)
        # print(prompt)
        
        # decoding_args = utils.OpenAIDecodingArguments(
        #     temperature=temperature,
        #     n=1,
        #     max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
        #     top_p=top_p,
        #     stop=["\n20", "20.", "20."],
        # )
        request_start = time.time()

        if client == 'openai':
            result = utils.openai_gpt(
                prompt=prompt,
            )
        else:    
            prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
            result = utils.claude_gpt(
                prompt=prompt,
            )
            print("*************************************************************")
            if result is None:
                continue
            print(f"Length: {len(result)}")
            long_result = result
            
            try_count = 1
            while try_count < 1:
                time.sleep(3)
                prompt += f"{result}{anthropic.HUMAN_PROMPT} The <essay> you just wrote is really short. Make your essay a lot longer.{anthropic.AI_PROMPT}"
                result = utils.claude_gpt(
                    prompt=prompt,
                )
                try_count += 1
                if result is None:
                    continue
                print(f"Length: {len(result)}")
                if len(result) > len(long_result):
                    long_result = result
                

            result = long_result

        if result is None:
            continue

        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        try:
            if client == 'openai':
                new_instructions = post_process_gpt3_response(
                    num_prompt_instructions, result
                )
            else:  
                if is_question:
                    new_instructions = post_process_claude_question(
                        num_prompt_instructions, result
                    ) 
                elif is_answer:
                    new_instructions = post_process_claude_answer(
                        question, result, context
                    )
                else:     
                    new_instructions = post_process_claude_response(
                        num_prompt_instructions, result
                    )
            instruction_data += new_instructions

            total = len(instruction_data)
            keep = 0
            for instruction_data_entry in instruction_data:
                keep += 1
                machine_instruction_data.append(instruction_data_entry)
                progress_bar.update(1)
            process_duration = time.time() - process_start
            print(
                f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s"
            )
            print(f"Generated {total} instructions, kept {keep} instructions")
            
            utils.jdump(
                machine_instruction_data, os.path.join(output_dir, f"{client}-run-{run_start}-batch-{batch_id}.json")
            )

            total_generated_results += keep
            batch_total += keep

            if batch_total >= output_batch_size:
                batch_id += 1
                batch_total = 0
                machine_instruction_data = []
            
        except Exception as e:
            print(f"{e}")
        


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
