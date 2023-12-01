import os
import re
from pathlib import Path


import openai
import pandas
import numpy as np
import tiktoken


EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"
MAX_EMBEDDINGS = 1536
MAX_TOKENS = 1600
GPT_MODEL = "gpt-3.5-turbo"


def get_embedding(text, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]


def list_document_files():
    # return all markdown files under jekyll/_cci2 folder
    folder_path = Path("./docs")
    md_files = folder_path.glob("*.md")
    files = [f for f in md_files]
    return files


def parse(filepath):
    with open(filepath) as f:
        raw = f.read()
        meta = {"filepath": filepath}
        # parse metadata
        if raw.startswith("---"):
            raw_header, body = raw.split("---", 2)[1:]
            for raw_line in raw_header.split("\n"):
                line = raw_line.strip()
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip(" \"'")
        else:
            body = raw

        title = meta["title"] if "title" in meta else meta["filename"]

        body = f"# {title}\n{body}"
        sections = re.findall("[#]{1,4} .*\n", body)
        split_txt = "-=-=-=-=-="
        # TODO: ignore "Next Step"
        for section in sections:
            body = body.replace(section, split_txt)

        # TODO: strip `{: header }`
        contents = [x.strip() for x in body.split(split_txt)]
        headers = [x.strip("# \n") for x in sections]
        sections_tuple = zip(headers, contents)

        # skip short sections
        sections_tuple = [(x, y) for x, y in sections_tuple if len(y.strip()) > 30]

        return meta, sections_tuple


def get_document_embeddings(files):
    embeddings = []
    for f in files:
        _, section_tuple = parse(f)
        for header, section in section_tuple:
            print("calculating embeddings:", str(f), header)
            embeddings.append(
                {
                    "title": str(f),
                    "header": header,
                    "section": section,
                    "emb": get_embedding(section),
                }
            )

    return embeddings


def save_embeddings_to_csv(embeddings):
    cols = ("title", "header", "section") + tuple(range(MAX_EMBEDDINGS))
    rows = []
    for emb in embeddings:
        # print("processing csv:", emb["title"], emb["header"])
        new_row = [emb["title"], emb["header"], emb["section"]]
        for i in range(MAX_EMBEDDINGS):
            new_row.append(emb["emb"][i])
        rows.append(new_row)
    export_df = pandas.DataFrame(rows, columns=cols)
    export_df.to_csv("embeddings.csv", index=False)


def cal_embeddings():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY is not set")

    files = list_document_files()
    embeddings = get_document_embeddings(files)
    save_embeddings_to_csv(embeddings)


def vector_projection(a, b):
    # calculate similarity of two vectors
    return np.dot(np.array(a), np.array(b))


def convert_embeddings_from_str(emb):
    embeddings = []
    for i in range(MAX_EMBEDDINGS):
        embeddings.append(float(emb[str(i)]))
    return embeddings


def get_relevant_sections(input_emb, document_emb):
    distance = []
    for index, row in document_emb.iterrows():
        distance.append(
            (vector_projection(input_emb, convert_embeddings_from_str(row)), index)
        )

    # return the top 10 most relevant sections
    rows_index = [i[1] for i in sorted(distance, reverse=True)[:10]]
    relevant_sections = document_emb.loc[rows_index]

    return [s["section"] for _, s in relevant_sections.iterrows()]


def get_all_embeddings_from_csv():
    embeddings = pandas.read_csv("embeddings.csv")
    return embeddings


def num_tokens(text):
    encoding = tiktoken.encoding_for_model(GPT_MODEL)
    return len(encoding.encode(text))


def construct_context(sections):
    # Ensure context token length < max tokens
    context = sections[0]
    length = num_tokens(context)
    for section in sections[1:]:
        section_len = num_tokens(section)
        if length + section_len > MAX_TOKENS:
            break
        context += section
        length += section_len

    return context


def request(prompt, context=""):
    # Send request to OpenAI API
    print("Asking ChatGPT...")
    messages = [
        {
            "role": "system",
            "content": "You're a CircleCI doc assistant. \
                Answer the question based on the context provided.",
        },
        {"role": "assistant", "content": context},
        {"role": "user", "content": prompt},
    ]
    chat_completion = openai.ChatCompletion.create(model=GPT_MODEL, messages=messages)
    print(chat_completion.choices[0].message.content)
    print("\n")


def start_chatting():
    while True:
        user_input = input("Please enter your prompt: ")
        if user_input == "exit":
            break

        if not user_input:
            continue

        input_emb = get_embedding(user_input)
        # TODO: cache document embeddings
        document_embeddings = get_all_embeddings_from_csv()
        relevant_sections = get_relevant_sections(input_emb, document_embeddings)
        context = construct_context(relevant_sections)
        request(user_input, context)


if __name__ == "__main__":
    # cal_embeddings()
    start_chatting()
