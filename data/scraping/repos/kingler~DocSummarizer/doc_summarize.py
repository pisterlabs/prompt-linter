import requests
import openai
import asyncio
from pathlib import Path
import tiktoken 
from termcolor import colored
import json
import time 

import backoff


encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

wiki_article_list = [
    "Artificial_intelligence",
    "History_of_artificial_intelligence",
    "Philosophy_of_artificial_intelligence",
    "Ethics_of_artificial_intelligence",
    "Artificial_intelligence_in_healthcare",    
    "Artificial_general_intelligence",
    "Machine_learning",
    "Deep_learning",
    "Reinforcement_learning",
    "Supervised_learning",
    "Unsupervised_learning",
    "data_mining",
    "Convolutional_neural_network",
    "Recurrent_neural_network",
    "Generative_adversarial_network",
    "Natural_language_processing",
    "Computer_vision",   
    "Artificial_intelligence_in_video_games",
    "AlphaGo",
    "evolutionary_algorithm",
]


def download_wikipedia_articles(wiki_titles):
    # Download Wikipedia articles if they don't exist
    for title in wiki_titles:
        data_path = Path('data')
        if not data_path.exists():
            Path.mkdir(data_path)

        if not (data_path / f"{title}.txt").exists():

            print(f"Downloading {title} from Wikipedia")
            response = requests.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'format': 'json',
                    'titles': title,
                    'prop': 'extracts',
                    # 'exintro': True,
                    'explaintext': True,
                }
            ).json()
            page = next(iter(response['query']['pages'].values()))
            wiki_text = page['extract']

            with open(data_path / f"{title}.txt", 'w', encoding="utf-8", errors="ignore") as fp:
                fp.write(wiki_text)

download_wikipedia_articles(wiki_article_list)

docs_to_be_summarized = []
# load all articles into a list
for title in wiki_article_list:
    with open(f"data/{title}.txt", encoding="utf-8", errors="ignore") as fp:
        docs_to_be_summarized.append(fp.read())
        

# count tokens for a single document
# total_tokens = len(encoding.encode(docs_to_be_summarized[0]))

# count the tokens and print for each document for each wiki artricle
total_tokens_for_all_articles = 0
for i in range(len(docs_to_be_summarized)):
    total_tokens = len(encoding.encode(docs_to_be_summarized[i]))
    total_tokens_for_all_articles += total_tokens
    print(colored(f"TOTAL TOKENS FOR {wiki_article_list[i]}: ", "green"), total_tokens)
print(colored(f"TOTAL TOKENS FOR ALL ARTICLES: ", "red"), total_tokens_for_all_articles)


def divide_document_into_500_token_chunks(document):
    chunks = []
    doc_tokens = encoding.encode(document)
    for i in range(0, len(doc_tokens), 500):
        chunk = doc_tokens[i:i+500]
        chunks.append(encoding.decode(chunk))
    return chunks

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
async def call_gpt(document, delay=0):
    if delay:
        print(colored(f"Delaying async call for {delay} seconds...", "red"))
        await asyncio.sleep(delay)
    if type(document) == list:
        print(colored(f"Starting async call for: {document[0][:50]}...", "yellow"))  # Print first 50 characters
    else:
        print(colored(f"Starting async call for: {document[:50]}...", "yellow"))  # Print first 50 characters

    response = await openai.ChatCompletion.acreate(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a diligent and very concise document summarizer."},
            {"role": "user", "content": f"Please summarize the following document: {document}"}
        ]
    )

    if type(document) == list:
        print(colored(f"Finished async call for: {document[0][:50]}...", "blue"))
    else:
        print(colored(f"Finished async call for: {document[:50]}...", "blue"))
    return response["choices"][0]["message"]["content"]

async def summarize_documents(documents):
    # Create a nested list of chunks
    nested_chunks = [divide_document_into_500_token_chunks(doc) for doc in documents]
    
    # Create a flat list of all chunk tasks, but keep the indices of each document's chunks
    tasks = []
    doc_indices = []
    total_tokens_sent = 0
    for doc_chunks in nested_chunks:
        doc_task_indices = []
        for chunk in doc_chunks:
            chunk_token_count = len(encoding.encode(chunk))
            total_tokens_sent += chunk_token_count  # update the total tokens sent

            # calculate how many 60000-token segments are in the total
            segments = total_tokens_sent // 60000

            # set the delay to be 60 seconds for each 60000-token segment
            delay = segments * 60

            task = call_gpt(chunk, delay)
            doc_task_indices.append(len(tasks))
            tasks.append(task)
        doc_indices.append(doc_task_indices)
    
    # Summarize all chunks at once
    all_summaries = await asyncio.gather(*tasks)
    
    # Group the summaries by document
    doc_summaries = [[all_summaries[i] for i in indices] for indices in doc_indices]
    
    # Create a dictionary of summaries
    summaries_dict = {wiki_article_list[i]: summary for i, summary in enumerate(doc_summaries)}
    
    return summaries_dict


# Run the async function
summaries = asyncio.run(summarize_documents(docs_to_be_summarized))


# Save the summaries to a JSON file
with open("summaries.json", "w", encoding="utf-8") as fp:
    json.dump(summaries, fp, ensure_ascii=False, indent=4)

# summarize each summary from the json file again with asyncio and save it to final_summary.json
with open("summaries.json", "r", encoding="utf-8") as fp:
    summaries = json.load(fp)

async def summarize_summary(summaries):
    # Create a list of summary tasks
    tasks = [call_gpt(summary) for summary in summaries.values()]

    # Run all summary tasks at once
    final_summaries = await asyncio.gather(*tasks)
    
    # Create a dictionary of final summaries
    final_summaries_dict = {wiki_article_list[i]: summary for i, summary in enumerate(final_summaries)}
    
    return final_summaries_dict

# Run the async function
final_summaries = asyncio.run(summarize_summary(summaries))

# Save the final summaries to a JSON file
with open("final_summaries.json", "w", encoding="utf-8") as fp:
    json.dump(final_summaries, fp, ensure_ascii=False, indent=4)
