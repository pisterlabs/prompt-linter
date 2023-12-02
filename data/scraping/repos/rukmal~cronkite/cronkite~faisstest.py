# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import faiss
from io import StringIO
import pandas as pd
import os
from langchain.embeddings import OpenAIEmbeddings

# articles is a list of article dictionaries, hall_prompt is a string, k is the top answers
def get_most_similar(articles, hall_prompt, k):

    raw_texts = map(lambda x: x["content"], articles)
    embeddings = OpenAIEmbeddings()
    query_embedding = np.asarray([embeddings.embed_query(hall_prompt)])
    doc_embeddings = np.asarray(embeddings.embed_documents(raw_texts))
    d = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(doc_embeddings)
    D, I = index.search(query_embedding, k)  # search
    answers = []
    print(f"LENGTH: {len(I[0])}")
    for k in I[0]:
        answers.append(articles[int(k)])

    print(f"ANSWERS: {answers}")

    return answers


def main():
    # remove duplicates and NaN
    # sentences = [word for word in list(set(data)) if type(word) is str]
    article1 = {"title": "", "url": "", "date": "", "content": "someone with a football", "summary": ""}
    article2 = {"title": "", "url": "", "date": "", "content": "someone not with a football", "summary": ""}
    article3 = {"title": "", "url": "", "date": "", "content": "someone not with a soccer", "summary": ""}
    articles = [article1, article2, article3]
    hall_prompt = "Someone sprints with a football"
    k = 2
    text_results = get_most_similar(articles, hall_prompt, k)
    print(text_results)


if __name__ == "__main__":
    main()
