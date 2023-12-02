import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") 
openai.api_key = API_KEY
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

df = pd.read_csv('icici_with_token_new.csv')

# df["token"] = None
# for idx, r in df.iterrows():
# #     print(len(r.content))
# #     df["token"] = df[len(r.content)]
#     df.loc[idx,'token'] = len(r.content)

# "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" -- MLAI
# "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" -- MLAI
# "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" -- LP
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

# import time
# def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
#     embeddings = {}
#     for idx, r in df.iterrows():
#         print(idx)
# #         print(r)
#         embeddings[idx] = get_embedding(r.title)
#         time.sleep(5)  # Add a delay of 10 second between requests
#     return embeddings


def load_embeddings(fname: "str") -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)

    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
        (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()

    }

document_embeddings = load_embeddings("icici_embed.csv")


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

# a = order_document_sections_by_query_similarity("Give me a list of icici credit crads  interest rate", document_embeddings)[:5]

MAX_SECTION_LEN = 7552
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

print(f"Context separator contains {separator_len} tokens")


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)[:4]
    #     print(most_relevant_document_sections)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        #         print(section_index)
        document_section = df.loc[section_index]
        #         print(document_section)
        chosen_sections_len += document_section.token + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    header = """Answer the question as truthfully as possible using the provided Given Below Information only, and if the answer is not contained within the text below, say "I don't know.For more details call on our CUSTOMER CARE NO.1800 1080 or visit our website https://www.icicibank.com/ and Question is not provided please ask for it"\n\nGiven Below Information:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"





COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 1.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}


# def generate_response(input_text):
#     output_text = f"<a href={input_text}>{input_text}</a>"
#     return output_text


import re
import pandas as pd
import openai


def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: document_embeddings,
        show_prompt: bool = True,
) -> str:
    # Construct prompt to send to OpenAI API
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    # Print prompt if show_prompt is True
    if show_prompt:
        print(prompt)

    # Send prompt to OpenAI API and get response
    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )
    text = response["choices"][0]["text"]

    # Add relevant URLs to response based on query
    regex = r"(credit card)"
    match = re.search(regex, text.lower())
    print(match)
    if match:
        text += " To know more, please visit https://www.icicibank.com/card/credit-cards/credit-card"

    regex = r"(personal loan)"
    match = re.search(regex, text.lower())
    if match:
        text += " To know more, please visit https://www.icicibank.com/personal-banking/loans/personal-loan"

    regex = r"(home loan)"
    match = re.search(regex, text.lower())
    if match:
        text += " To know more, please visit https://www.icicibank.com/personal-banking/loans/home-loan"

    regex = r"(fixed deposit)"
    match = re.search(regex, text.lower())
    if match:
        text += " To know more, please visit https://www.icicibank.com/personal-banking/deposits/fixed-deposit"

    regex = r"(demat account)|(trading account)"
    match = re.search(regex, text.lower())
    if match:
        text += " To know more, please visit https://www.icicibank.com/personal-banking/accounts/three-in-one-trading-account"
    # Check if response contains a URL and generate response if so
    regex = r"(?P<url>https?://[^\s]+)"
    #     regex = r'https?://(?:www\.)?icicibank\.com/\S+(?:\?\S*)?(?:#\S*)?'
    #     regex = r"https?://[^\s<>]+(?:\w/)?(?:[^\s()]*)"
    match = re.search(regex, text)
    print(match)
    if match:
        url = match.group("url")
        text = text.replace(url, "")
        # link = generate_response(url)
        link = url
        #         print(link)
        return f"{text}, {link}"

    # Return response text
    return text

def inputdata(inpval: str) :
    response = answer_query_with_context(inpval, df, document_embeddings)
    if isinstance(response, tuple):
        text, url = response
        return text, url
    else:
        return response


# print(answer_query_with_context("Tell me about home loan", df, document_embeddings)[0].strip())
# def add_text(history, text):
#     history = history + [(text, None)]
#     return history, ""

# def inputdata(history) :
#     history2 = history[-1][0]


#     response = answer_query_with_context(history2, df, document_embeddings)
#     # if isinstance(response, tuple):
#     #     text, url = response
#     #     return text, url
#     # else:
#     history[-1][1] = response
#     print("History: ",history)
#     return history

# import gradio as gr
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()


#     txt = gr.Textbox()
#     clear = gr.Button("Clear")
#         # with gr.Column(scale=0.15, min_width=0):
#         #     btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

#     txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(inputdata, chatbot, chatbot)
#     clear.click(lambda: None, None, chatbot, queue=False)

#     if __name__ == "__main__":
#         demo.launch()