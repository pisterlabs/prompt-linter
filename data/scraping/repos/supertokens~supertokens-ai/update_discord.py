import requests
import json
import pandas as pd
import os
import openai
from dotenv import load_dotenv
import tiktoken
from split_or_union_data_files import perform_split_or_union
load_dotenv()

########### part 1: fetch threads from discord and save into processed/discord_threads.json ###########

# read existing threads from processed/discord_threads.json if it exists
try:
    with open("processed/discord_threads.json", "r") as f:
        threads = json.loads(f.read())
except:
    threads = []

def fetch_from_channel(channel):
    url = "https://community.supertokens.com/api/threads"
    params = {
        "channelId": channel,
        "accountId": "4e18cd5a-78d6-4be7-b53a-236bf4b40867"
    }
    headers = {
        "Content-Type": "application/json"
    }
    prev_cursor = None
    while True:
        if prev_cursor is not None:
            params["cursor"] = prev_cursor
        print("fetching threads with cursor: " + str(prev_cursor))
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        found_old_threads = False
        for thread in data["threads"]:
            if len(thread["messages"]) <= 1:
                # this is cause previously, we used to not create threads for messages.
                continue
            if thread["id"] in [t["id"] for t in threads]:
                found_old_threads = True
                continue
            threads.append(thread)
        if data["nextCursor"] is None or data["nextCursor"]["prev"] is None or found_old_threads:
            break
        prev_cursor = data["nextCursor"]["prev"]

# general channel
print("fetching general channel")
fetch_from_channel("1c19dd58-394e-412d-a853-f9075fbea34f")
print()

# support-question channel
print("fetching support-question channel")
fetch_from_channel("ae164b95-2204-44d4-9595-69c1cdaf17ad")
print()

# bot-training channel
print("fetching bot-training channel")
fetch_from_channel("8b2adb2b-afbc-4906-849c-f8d0109d4019")
print()

# save threads as a json file
with open("processed/discord_threads.json", "w") as f:
    f.write(json.dumps(threads, indent=4))

test_case_string = "st-bot-test-case"

max_token_limit = 2048

tokenizer = tiktoken.get_encoding("cl100k_base")

openai.api_key = os.environ.get('OPEN_AI_KEY')

########### part 2: create embeddings for new chats ###########

df = pd.DataFrame(columns=['text', 'embeddings', 'id'])
if os.path.exists('processed/discord_threads.csv'):
    df = pd.read_csv('processed/discord_threads.csv')


test_df = pd.DataFrame(columns=['question', 'answer', 'embeddings', 'id'])
if os.path.exists('processed/test_cases.csv'):
    test_df = pd.read_csv('processed/test_cases.csv')

new_df = pd.DataFrame(columns=['text', 'embeddings', 'id'])

def find_df_for_id_from_df(id):
    for i in range(len(df)):
        if df.loc[i, 'id'] == id:
            return df.loc[i]
    return None

count = -1
for curr_thread in threads:
    count += 1
    existing_df = find_df_for_id_from_df(curr_thread['id'])
    if existing_df is not None:
        new_df.loc[count, 'text'] = existing_df['text']
        new_df.loc[count, 'embeddings'] = existing_df['embeddings']
        new_df.loc[count, 'id'] = existing_df['id']
        continue
    
    message = ""
    is_test_case = False
    for curr_message in curr_thread['messages']:
        if test_case_string in curr_message['body']:
            is_test_case = True
            break
        message += curr_message["author"]["username"] + ": " + curr_message['body'] + "~C_END~\n\n"
    

    tokens = tokenizer.encode(message)
    if (len(tokens) > max_token_limit):
        # we skip this thread cause it's too long..
        continue

    print("Fetching embeddings for thread: " + str(count) + " / " + str(len(threads)))
    embeddings = openai.Embedding.create(
        engine='text-embedding-ada-002',
        input=tokens
    )['data'][0]['embedding']

    new_df.loc[count, 'text'] = message
    new_df.loc[count, 'embeddings'] = embeddings
    new_df.loc[count, 'id'] = curr_thread['id']

    if is_test_case:
        question = curr_thread['messages'][0]["body"]
        answer = ""
        for i in range(len(curr_thread['messages'])):
            if i == 0:
                continue
            if test_case_string in curr_thread['messages'][i]['body']:
                break
            answer += curr_thread['messages'][i]['body'] + "\n"
        
        tokens = tokenizer.encode(answer)

        print("Fetching embeddings for test case.")
        embeddings = openai.Embedding.create(
            engine='text-embedding-ada-002',
            input=tokens
        )['data'][0]['embedding']
        curr_index = len(test_df)
        test_df.loc[curr_index, 'question'] = question
        test_df.loc[curr_index, 'answer'] = answer
        test_df.loc[curr_index, 'embeddings'] = embeddings
        test_df.loc[curr_index, 'id'] = curr_thread['id']

new_df.to_csv('processed/discord_threads.csv', index=False)
test_df.to_csv('processed/test_cases.csv', index=False)

########### part 3: update chunks folder ###########
perform_split_or_union()