import asyncio
import os
import json
import tiktoken
from transcribe import file_to_json_path, get_recordings, get_all_recordings, print_json
import langchain
from langchain.llms import OpenAI
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    HumanMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoTokenizer

# MAX_TRANSCRIPT_LENGTH = 1536
MAX_TRANSCRIPT_LENGTH = 1800
ANSWER_START_LENGTH = 50

NAMESPACE = 'Startup Interviews'

database_path = "data/.langchain.db"
langchain.llm_cache = SQLiteCache(database_path)

training_tokenizer_name = "huggyllama/llama-13b"
trainer_tokenizer = AutoTokenizer.from_pretrained(training_tokenizer_name)

async def main():
    data = get_recordings(f"data/{NAMESPACE}")
    # print(json.dumps(data, indent=4))
    all_recordings = get_all_recordings(data)
    # print_json(all_recordings)
    # print_json(len(all_recordings))

    # limit to only 2 recordings
    # all_recordings = all_recordings[:10]

    chat_items = []

    for i, recording in enumerate(all_recordings):
        # print(f"{i}: {recording['filePath']}")
        # print(f"{i + 1} of {len(all_recordings)}: {recording['title']}")

        json_file_path = file_to_json_path(recording['filePath'])
        if not os.path.exists(json_file_path):
            print(f"\tJSON file does not exist at {json_file_path}")
            continue

        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
            # print(json.dumps(json_data, indent=4))
            """
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "...",
                                "words": [
                                    {
                                        "word": "i",
                                        "start": 0.0,
                                        "end": 0.16,
                                        "confidence": 0.99353653,
                                        "speaker": 0,
                                        "speaker_confidence": 0.8430252,
                                        "punctuated_word": "I"
                                    },
                                ]
            """

            transcript = json_data['results']['channels'][0]['alternatives'][0]
            transcript_text = transcript['transcript']
            words = transcript['words']
            # print(len(words), len(transcript_text.split()))

            # count unique speakers
            num_speakers = get_num_speakers(words)
            # print(len(speakers))
            # print(num_speakers)

            # if num_speakers > 5:
            if num_speakers != 1:
                continue

            if token_length(transcript_text) > MAX_TRANSCRIPT_LENGTH:
                print(f"\tSkipping \"{recording['title']}\" because it's too long: {token_length(transcript_text)}")
                continue

            # chat_item = {
            #     'title': recording['title'],
            #     'speakers': num_speakers,
            #     'text': transcript_text,
            # }
            # duplicate recording
            chat_item = recording.copy()
            # merge in an object with the transcript text
            chat_item.update({
                'speakers': num_speakers,
                'text': transcript_text,
            })
            chat_items.append(chat_item)

    # limit to only 2 chat items
    # chat_items = chat_items[:100]
    # return

    # add start_text and question to each chat item
    print(f"Generating {len(chat_items)} questions")
    count = len(chat_items)
    for i, chat_item in enumerate(chat_items):
        curr = i + 1
        # print(f"{i+1} of {len(chat_items)} ({(perc)}) Generating question for {chat_item['title']}")
        # print(f"{curr} of {count} ({round(curr/count*100, 2)}%) Generating question for {chat_item['title']}")
        perc = round(curr/count*100, 2)
        print(f"{curr} of {count} ({perc}%): Generating question for {chat_item['title']}")
        start_text = get_start_text(chat_item['text'])
        question = get_question(chat_item['title'], start_text)
        print(f"\tQ: {question}")
        chat_item.update({
            'start_text': start_text,
            'question': question,
        })

    # print_json(chat_items)
    print_json(len(chat_items))
    write_jsonl(chat_items, "train")

def get_num_speakers(words):
    speakers = set()
    for word in words:
        speakers.add(word['speaker'])
    num_speakers = len(speakers)
    return num_speakers

enc = tiktoken.get_encoding("cl100k_base")
def get_tokens(contents):
    return enc.encode(contents)
    # return tokenizer(contents)['input_ids']

def decode_tokens(tokens):
    return enc.decode(tokens)
    # return tokenizer.decode(tokens)

def get_start_text(contents):
    tokens = get_tokens(contents)
    # if longer than ANSWER_START_LENGTH tokens, truncate and add ...
    if len(tokens) > ANSWER_START_LENGTH:
        return decode_tokens(tokens[:ANSWER_START_LENGTH]) + '...'
    else:
        return decode_tokens(tokens)

def token_length(contents):
    return len(get_tokens(contents))

def token_length_for_trainer(contents):
    return len(trainer_tokenizer(contents)['input_ids'])

def get_question(title, reply):
    template="You are a helpful, truthful, detailed assistant writing a transcript of an interview."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="""Task: Write the question which is most likely to produce the following reply.

Interview Title: {title}

Reply: {reply}

Question:"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chat = ChatOpenAI(streaming=False, temperature=0)
    resp = chat(chat_prompt.format_prompt(title=title, reply=reply).to_messages())
    return resp.content

# Write chat to .json in format:
# [{ "instruction": "...", "input": "...", "output": "..." }, ...]
def write_jsonl(chat_items, name = 'chat'):
    chat_file_path = f"data/{NAMESPACE}/{name}.jsonl"

    # create rows
    print(f"Creating rows: {len(chat_items)}")
    rows = []
    for chat_item in chat_items:
        row = {
            "instruction": chat_item['question'],
            "input": "",
            "output": chat_item['text'],
            "instruction_length": token_length_for_trainer(chat_item['question']),
            "output_length": token_length_for_trainer(chat_item['text']),
            "title": chat_item['title'],
            "start": chat_item['start_text'],
        }
        rows.append(row)

    # write rows to file
    with open(chat_file_path, 'w') as chat_file:
        # for chat_item in chat_items:
        #     # start_text = get_start_text(chat_item['text'])
        #     # question = get_question(chat_item['title'], start_text)
        #     row = {
        #         # "instruction": question,
        #         "instruction": chat_item['question'],
        #         "input": "",
        #         "output": chat_item['text'],
        #         "len": token_length(chat_item['text']),
        #         "title": chat_item['title'],
        #         # "start": start_text,
        #         "start": chat_item['start_text'],
        #     }
        for row in rows:
            chat_file.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f"Wrote {len(chat_items)} chat items to {chat_file_path}")
    max_instruction_len = max([row['instruction_length'] for row in rows])
    max_output_len = max([row['output_length'] for row in rows])
    print(f"Max instruction length: {max_instruction_len}")
    print(f"Max output length: {max_output_len}")

if __name__ == "__main__":
    asyncio.run(main())
