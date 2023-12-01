# Imports
from collections import defaultdict
from time import sleep
from llama_index import (
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from model_context import get_anyscale_context
from templates import custom_template, yn_template
import csv
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(base_url="https://api.endpoints.anyscale.com/v1", api_key="KEY")

# DEBUG LOGS
# import llama_index
# llama_index.set_global_handler("simple")

rag = True
yn = True

if rag:
    # Select Model
    print("Loading model context...")
    service_context = get_anyscale_context()
    set_global_service_context(service_context)

    # Load embedded data for RAG
    print("Loading RAG embeddings...")
    storage_context = StorageContext.from_defaults(persist_dir="vector-db-all")
    index = load_index_from_storage(
        service_context=service_context, storage_context=storage_context
    )
    # Assemble Query Engine
    top_k = 5
    if yn:
        query_engine = index.as_query_engine(
            text_qa_template=yn_template,
            similarity_top_k=top_k,
            # verbose=True,
            # streaming=True,
        )
    else:
        query_engine = index.as_query_engine(
            text_qa_template=custom_template,
            similarity_top_k=top_k,
            # verbose=True,
            # streaming=True,
        )

def query_baseline(text: str, yn: bool) -> str:
    while True:
        if yn:
            content_msg = "Answer with yes/no and an explanation."
        else:
            content_msg = "Express whether the statement is true or false and explain why." #Your job is to
        try:
            chat_completion = client.chat.completions.create(
                model="meta-llama/Llama-2-7b-chat-hf",
                messages=[
                    {
                        "role": "system",
                        "content": content_msg,
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                temperature=0,
            )
            return chat_completion.choices[0].message.content.strip()
        except:
            print("BROKE: ", text)
            sleep(10)

# Load evaluation data
print("Loading evaluation data...")
labeled_data = defaultdict(list)
with open("../all-exemplars-pruned/negative.csv", "r") as full_data:
    data_reader = csv.DictReader(full_data)
    for sample in data_reader:
        labeled_data[sample["generic"]].append(sample["exemplar"])
print(f"{len(labeled_data)} generics loaded!")
generics = list(labeled_data.keys())

# Evaluation Loop
print("Beginning evaluation:")
tie = 0
loss = 0
win = 0
for i in tqdm(range(1000), desc="Generic evaluation process"):
    sample = generics[i].lower()
    for ext in ["Some", "No"]:
        prompt = ext.lower() + " " + sample.lower()
        if yn:
            if ext == "No":
                prompt = "Is it never the case that " + sample[:-1].lower() + "?"
            else:
                prompt = "Do " + prompt[:-1] + "?"
        if rag:
            response = query_engine.query(prompt)
        else:
            response = query_baseline(prompt, yn)

        if yn:
            false_count = str(response).lower().count("no") - str(response).lower().count("not")
            true_count = str(response).lower().count("yes")
        else:
            false_count = str(response).lower().count("false")
            true_count = str(response).lower().count("true")

#        print(false_count)
#        print(true_count)

        if ext == "Some":
            good = true_count
            bad = false_count
        elif ext == "No":
            good = false_count
            bad = true_count

        if good > bad:
            win += 1
        elif bad > good:
            loss += 1
        else:
            tie += 1

print("Wins: ", win)
print("Ties: ", tie)
print("Loss: ", loss)
