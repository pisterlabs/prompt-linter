import os
import json
import openai
from dotenv import dotenv_values, load_dotenv
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import multiprocessing
import time
from random import randint
import argparse
from concurrent.futures import ThreadPoolExecutor

config = dotenv_values("../../configs/.env")

openai.organization = config.get('OPENAI_ORGANIZATION')
openai.api_key = config.get('OPENAI_API_KEY')

# system message
def system_message(message_num):
    
    if message_num > 16:
        print("message_num is out of range")
        
    # 총 16개의 메시지
    message = [
        "", # 일부러 비워져있음
        "당신은 챗봇입니다. 사용자가 답을 이해하기 위해 외부에서 검색할 필요가 없도록 자세한 답변을 제공하세요.",
        "당신은 챗봇입니다. 과제가 주어집니다. 상세하고 긴 답변을 생성해야 합니다.",
        "당신은 항상 설명을 제공하는 유용한 챗봇입니다. 5세 아이에게 대답한다고 생각하세요.",
        "귀하는 지시를 매우 잘 따르는 챗봇입니다. 최대한 많이 도와주세요.",
        "귀하는 사람들이 정보를 찾도록 돕는 챗봇입니다. 사용자가 답을 이해하기 위해 외부에서 검색할 필요가 없도록 자세한 답변을 제공하세요.",
        "당신은 챗봇입니다. 사용자가 당신에게 과제를 줍니다. 당신의 목표는 가능한 한 충실하게 작업을 완료하는 것입니다. 작업을 수행하는 동안 단계별로 생각하고 단계를 정당화하세요.",
        "과제를 설명하고 답을 설명해야 합니다. 객관식 문제에 답하는 동안 먼저 정답을 출력합니다. 그런 다음 다른 답이 틀린 이유를 설명합니다. 5세 어린이에게 대답하는 것처럼 생각하세요.",
        "정답을 도출하기 위해 정의를 어떻게 사용했는지 설명하세요.",
        "당신은 챗봇입니다. 과제를 설명하고 답을 설명해야 합니다. 객관식 문제에 답하는 동안 먼저 정답을 출력합니다. 그런 다음 다른 답이 틀린 이유를 설명하세요. 질문에 답하기 위해 추가 지식을 사용해야 할 수도 있습니다.",
        "당신은 사람들이 정보를 찾을 수 있도록 도와주는 챗봇입니다. 사용자가 질문을 합니다. 여러분의 임무는 가능한 한 충실하게 대답하는 것입니다. 답변하는 동안 단계별로 생각하고 답변을 정당화하세요.",
        "사용자가 몇 가지 지침이 포함된 과제를 제공합니다. 여러분의 임무는 최대한 충실하게 지시를 따르는 것입니다. 대답하는 동안 단계별로 생각하고 답을 정당화하세요.",
        "당신은 챗봇입니다. 과제가 주어지면 과제가 무엇을 요구하는지, 과제가 제공하는 지침이 있는지, 그리고 그 지침을 사용하여 답을 찾는 방법을 간단한 단계로 설명합니다.",
        "귀하는 모든 언어를 알고 있으며 한 언어를 다른 언어로 번역하는 방법을 알고 있는 챗봇입니다. 작업이 주어지면 작업에서 무엇을 요구하는지, 작업에서 제공하는 지침을 간단한 단계로 설명합니다. 작업을 해결하고 가이드라인을 사용하여 작업을 해결한 방법을 보여줍니다.",
        "작업의 정의와 샘플 입력이 주어지면 정의를 작은 부분으로 나눕니다. 각 부분에는 몇 가지 지침이 있습니다. 지침의 기준을 충족하는 예시를 보여 주면서 그 의미를 설명합니다. 다음 형식을 사용합니다: 파트 #: 정의의 핵심 부분입니다. 사용법: 핵심 부분의 기준을 충족하는 샘플 응답입니다. 기준을 충족한다고 생각하는 이유를 설명하세요.",
        "귀하는 사람들이 정보를 찾도록 도와주는 챗봇입니다.",
    ]

    return message[message_num]


# define args
def define_args():
    
    p = argparse.ArgumentParser()
    
    p.add_argument("--src_path", type=str)
    p.add_argument("--dst_path", type=str)
    
    config = p.parse_args()
    
    return config


#####################################################
# GPT call
#####################################################
def gpt_call(
    prompt,
    model="gpt-4" # model="gpt-4",
    ):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                    {"role": "user", "content": prompt},
                ]
    )
    output_text = response["choices"][0]["message"]["content"]

    print("output_text", output_text)

    return output_text

def instruction_template(system, question, answer):

    prompt_template = PromptTemplate(
        input_variables=[
            "system",
            "question",
            "answer"
            ],
        template="\n".join([
            "### 시스템:",
            "{system}",
            "### 질문:",
            "{question}",
            "### 정답:",
            "{answer}",
            "### 챗봇:",
        ])
    )
    prompt_template = prompt_template.format(
        system=system,
        question=question,
        answer=answer
    )
    #print("prompt_template", prompt_template)

    return prompt_template

def final_instruction_template(system, question, answer):

    prompt_template = PromptTemplate(
        input_variables=[
            "system",
            "question",
            "answer"
            ],
        template="\n".join([
            "### 시스템:",
            "{system}",
            "### 질문:",
            "{question}",
            "### 챗봇:",
            "{answer}" + "<|endoftext|>"
        ])
    )
    prompt_template = prompt_template.format(
        system=system,
        question=question,
        answer=answer
    )
    #print("prompt_template", prompt_template)

    return prompt_template




# code

call_counter = 0  # This counter is for the gpt_call rate limit

def rate_limit():
    global call_counter
    time_interval = 60.0  # one minute

    if call_counter >= 20000:
        time.sleep(time_interval)
        call_counter = 0  # reset counter after waiting

def append_to_dst(data):
    with open(cfg.dst_path, "a") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write(",\n")

def process_chunk(args):
    chunk, rate_limiter, lock, write_lock = args
    global call_counter

    for line in chunk:
        dict_line = json.loads(line)
        text_in_line = dict_line['text']

        question = text_in_line.split('### 답변:')[0].replace("### 질문: ", "").strip()
        answer = text_in_line.split('### 답변:')[1].replace("", "").replace("\n\n", "\n").strip()

        system_message_text = system_message(randint(0, 15))

        instruction_text = instruction_template(
            system=system_message_text,
            question=question,
            answer=answer
        )

        with lock:
            rate_limit()  # Ensure we do not exceed the rate limit
            gpt_response = gpt_call(instruction_text)
            call_counter += 1

        final_instruction_text = final_instruction_template(
            system=system_message_text,
            question=question,
            answer=gpt_response
        )

        final_instruction_dict = {
            "text": final_instruction_text
        }

        # Write to file immediately after getting the response
        with write_lock:
            append_to_dst(final_instruction_dict)

def main(cfg):
    # Create a Manager object
    manager = multiprocessing.Manager()

    # Use the manager to create locks and rate limiter
    rate_limiter = manager.list([time.time()])
    lock = manager.Lock()
    write_lock = manager.Lock()

    with open(cfg.src_path, 'r') as f:
        data = f.readlines()

    # Check how many lines are already processed
    with open(cfg.dst_path, 'r', encoding='utf-8') as f:
        lines_processed = len(f.readlines())

    data = data[lines_processed:]

    # Create chunks for multiprocessing
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(data) // num_cores
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    with multiprocessing.Pool(num_cores) as pool:
        args = [(chunk, rate_limiter, lock, write_lock) for chunk in chunks]
        list(tqdm(pool.map(process_chunk, args), total=len(chunks)))

    # Close the JSON array in the destination file
    with open(cfg.dst_path, "rb+") as k:
        # Go to the second last character in the file
        k.seek(-2, 2)
        k.truncate()
        k.write(b"\n]")

if __name__ == "__main__":
    cfg = define_args()
    main(cfg)
