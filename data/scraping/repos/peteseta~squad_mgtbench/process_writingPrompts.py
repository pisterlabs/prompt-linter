from dotenv import load_dotenv
import csv
import os
from tqdm import tqdm
import time
import pickle

import tiktoken
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# encoding
enc = tiktoken.get_encoding("cl100k_base")

# https://platform.openai.com/docs/models/model-endpoint-compatibility
GPT_35_MODEL_CODE = "gpt-3.5-turbo-0613"
GPT_4_MODEL_CODE = "gpt-4-0613"

spending = []

def call_openai(question, model, system_prompt="You are a helpful assistant."):
    attempts = 0
    while attempts < 3:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                max_tokens=500,
            )
            # time.sleep(2)
            return response.choices[0].message["content"].replace('\n', ' ')
        except:
            tqdm.write("Failed to get a response from OpenAI. Retrying...")
            attempts += 1
            time.sleep(5)
    raise Exception("Failed to get a response from OpenAI after 3 attempts.")

def call_anthropic(question):
    attempts = 0
    while attempts < 3:
        try:
            anthropic = Anthropic()
            completion = anthropic.completions.create(
                model="claude-2",
                max_tokens_to_sample=500,
                prompt=f"{HUMAN_PROMPT} {question} {AI_PROMPT}",
            )
            return completion.completion.replace('\n', ' ')
        except:
            tqdm.write("Failed to get a response from Anthropic. Retrying...")
            attempts += 1
            time.sleep(5)
    raise Exception("Failed to get a response from Anthropic after 3 attempts.")

def generate_prompt(story_prompt):
    prompt = (
        f"Write a story for this prompt:\n{story_prompt}"
    )
    return prompt

def cost_per_qna(prompt, gpt_35_response, gpt_4_response):
    global total_spending
    
    prompt_tokens = len(enc.encode(prompt))
    gpt_35_response_tokens = len(enc.encode(gpt_35_response))
    gpt_4_response_tokens = len(enc.encode(gpt_4_response))
    
    # GPT-4: $0.03 / 1K tokens input, $0.06 / 1K tokens output
    # GPT-3.5: $0.0015 / 1K tokens	$0.002 / 1K tokens  
    gpt_4_cost = (prompt_tokens * 0.03 + gpt_4_response_tokens * 0.06) / 1000
    gpt_35_cost = 2*((prompt_tokens * 0.0015 + gpt_35_response_tokens * 0.002) / 1000)
    
    total_cost = (gpt_4_cost + gpt_35_cost)
    spending.append(gpt_4_cost + gpt_35_cost)
    
    return total_cost
    
def process(dataset_name):
    output_data = []
    
    for prompt, human_story in zip(tqdm(prompts, desc=f"Prompts for fold {dataset_name}"), human_stories):
        llm_question = generate_prompt(prompt)
        
        tqdm.write(f"Prompt: {prompt}")
        
        # Getting responses from different models/configurations
        gpt35_answer = call_openai(llm_question, GPT_35_MODEL_CODE)
        tqdm.write(f"ChatGPT Response: {gpt35_answer}\n")
        
        gpt35_prompted_answer = call_openai(
                    llm_question,
                    GPT_35_MODEL_CODE,
                    system_prompt="You are a bestselling author known for your exceptional worldbuilding and character development skills. Your writing exhibits a remarkable balance of perplexity and burstiness, creating a captivating and engaging reading experience. Perplexity, or the unpredictability of subsequent words, adds depth and intrigue to your prose, while burstiness, or the variation between sentence lengths and structures, enhances the natural flow and rhythm of your writing. In this task, your goal is to write a passage that showcases your mastery of language, ensuring that your text is fluent, natural, and human-like. Incorporate occasional unconventional or complex sentence structures where appropriate, as this will further enhance the richness of your writing. Your passage should demonstrate your ability to bring worlds to life and develop characters with depth and realism. Embrace the concepts of perplexity and burstiness, enabling you to craft a compelling and authentic piece of writing that captivates the reader's imagination.",
                )
        tqdm.write(f"ChatGPT Prompted Response: {gpt35_prompted_answer}\n")
        
        gpt4_answer = call_openai(llm_question, GPT_4_MODEL_CODE)
        tqdm.write(f"GPT-4 Response: {gpt4_answer}\n")
        
        claude_answer = call_anthropic(llm_question)
        tqdm.write(f"Claude Response: {claude_answer}\n")
        
        tqdm.write(f"Human Response: {human_story}[:1000]\n")

        # Appending processed data
        output_data.append({
            "prompt": prompt.strip(),
            "human_story": human_story.strip(),
            "chatgpt_story": gpt35_answer.strip(),
            "chatgpt_prompted_story": gpt35_prompted_answer.strip(),
            "gpt4_story": gpt4_answer.strip(),
            "claude_story": claude_answer.strip()
        })

        # Saving to CSV
        with open("writingPrompts_responses.csv", 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([prompt.strip(), human_story.strip(), gpt35_answer.strip(), gpt35_prompted_answer.strip(), gpt4_answer.strip(), claude_answer.strip()])
                
        qna_cost = cost_per_qna(llm_question, gpt35_answer, gpt4_answer)
        average_cost_per_prompt = sum(spending) / len(spending)
                
        tqdm.write(f"qna cost: ${qna_cost} | total cost: ${sum(spending)}")
        tqdm.write(f"Estimated total cost for dataset: ${average_cost_per_prompt * len(prompts)}")
        tqdm.write("-------------------------------------------------------------------------")
        
        # Save progress to pickle
        with open("progress.pkl", "wb") as pfile:
            pickle.dump(output_data, pfile)

write_header = False
if write_header:
    with open("writingPrompts_responses.csv", 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["prompt", "human_story", "ChatGPT-0613_answer", "ChatGPT-0613-prompted_answer", "GPT4_answer", "Claude_answer"])

datasets = ["train", "test", "valid"]

for dataset_name in datasets:
    # Check for progress
    try:
        with open(f"progress.pkl", "rb") as pfile:
            processed_data = pickle.load(pfile)
            last_processed = len(processed_data)
            tqdm.write(f"Loaded previous progress for {dataset_name}: {last_processed}")
    except (FileNotFoundError, EOFError):
        last_processed = 0

    if last_processed > 0:
        # Resume processing from where it stopped
        with open(f"writingPrompts_raw/{dataset_name}.wp_source", "r") as f:
            prompts = f.readlines()[last_processed:]
        with open(f"writingPrompts_raw/{dataset_name}.wp_target", "r") as f:
            human_stories = f.readlines()[last_processed:]
    
    process(dataset_name)
    