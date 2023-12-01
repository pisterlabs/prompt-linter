
import openai

from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate

import os
import argparse
import asyncio

from langchain.schema import (
    SystemMessage,
    HumanMessage,
)

async def async_generate(llm, message):    
    resp = await llm.agenerate([message])
    return resp.generations[0][0].text

async def generate_concurrently(segments, apikey, prompt, temperature):
    llm = ChatOpenAI(temperature=temperature, openai_api_key=apikey)
    promptTemplate = PromptTemplate(
    input_variables=["targetText", "targetPrompt"],
    template="""{targetPrompt} 
    {targetText}
    
    
    """,
    )

    #tasks = [async_generate(llm) for _ in range(10)]
    tasks = []
    for segment in segments:       
        content=promptTemplate.format(targetText=segment, targetPrompt=prompt)
        tasks.append(async_generate(llm,[HumanMessage(content=content)]))
    results = await asyncio.gather(*tasks)

    return results

def split_text(text, max_chars):
    lines = text.replace('。', '。\n').replace('.', '.\n').splitlines()  
    segments = []
    current_segment = ""

    for line in lines:
        # If a single line alone exceeds max_chars, force a split
        # 1 行だけで max_chars を超える場合は、強制的に分割します
        while len(line) > max_chars:
            split_index = max_chars - len(current_segment)
            segments.append(current_segment.strip() + line[:split_index])
            line = line[split_index:]
            current_segment = ""

        # If adding the next line doesn't exceed the limit, add it to the current segment
        # 次の行を追加しても制限を超えない場合は、現在のセグメントに追加します
        if len(current_segment) + len(line) + 1 <= max_chars:  # +1 for the line break
            current_segment += line + '\n'
        else:
            # If it does, save the current segment and start a new one
            # 存在する場合は、現在のセグメントを保存し、新しいセグメントを開始します新しいセグメントを開始します。
            segments.append(current_segment.strip())
            current_segment = line + '\n'

    # Don't forget to save the last segment if it's not empty
    # 最後のセグメントが空でない場合は、忘れずに保存してください忘れずに保存してください。
    if current_segment:
        segments.append(current_segment.strip())

    return segments

def delete_formatted_lines(string):
    # Split the string into lines
    lines = string.split("\n")
    
    # Check if the first or last line contains the string "formatted"
    if "整形" in lines[0]:
        del lines[0]
    if "整形" in lines[-1]:
        del lines[-1]
    
    # Join the remaining lines back into a string
    return "\n".join(lines)

async def process_text(text, max_chars, apikey, prompt, temperature):
    segments = split_text(text, max_chars)

    for i, segment in enumerate(segments):
        print("\n入力" + str(i) + "番目:\n" + segment)

    formatted_text = []
    #for segment in segments:
        # Send each part to the GPT API for formatting
        #response = openai.Completion.create(
        #    engine="text-davinci-002",
        #    prompt=segment,
        #    max_tokens=60
        #)

        #response = chain.run(segment)

        # Get the generated text and append to the list
        #formatted_text.append(response)

    responses = await generate_concurrently(segments, apikey, prompt, temperature)
    for i,response in enumerate(responses):
        print("\n出力" + str(i) + "番目:\n" + response)
        formatted_text.append(delete_formatted_lines(response))

    return '\n'.join(formatted_text)

async def main():
    parser = argparse.ArgumentParser(description='Process some text.')
    parser.add_argument('filepath', type=str, help='The path to the text file to be processed')
    parser.add_argument('--output', type=str, default='output.txt', help='The path to the output file')
    parser.add_argument('--config_file', type=str, default='config.txt', help='Path to configuration file')
    args = parser.parse_args()

    # Get the path of the executable file
    exe_path = os.path.abspath(__file__)
    # Get the directory where the executable file is located
    exe_dir = os.path.dirname(exe_path)
    # Get the path of the config file relative to the executable file directory
    config_path = os.path.join(exe_dir, args.config_file)

    # Read the configuration file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = {}
        for line in f:
            key, value = line.strip().split('=')
            config[key] = value

    # Extract the values we're interested in
    apikey = config['api_key']
    prompt = config['prompt']
    max_chars = int(config['max_chars'])
    temperature = float(config['temperature'])

    # Read the content of the file
    with open(args.filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    result = await process_text(text, max_chars, apikey, prompt, temperature)

    # Save the result to the output file
    with open(args.output, 'w', encoding='utf-8') as file:
        file.write(result)

if __name__ == "__main__":
    asyncio.run(main())