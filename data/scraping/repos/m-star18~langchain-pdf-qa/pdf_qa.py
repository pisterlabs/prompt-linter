# import
import os
import argparse

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate


def main(args):
    # Vectorize the question sentences. Then, 5 closely related sentences were extracted from the FAISS database
    os.environ["OPENAI_API_KEY"] = args.OPEN_API_KEY
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(args.faiss_save_path, embeddings)

    query = args.query
    embedding_vector = embeddings.embed_query(query)

    docs_and_scores = db.similarity_search_with_score_by_vector(embedding_vector, k=5)
    if len(docs_and_scores) != 5:
        raise AssertionError("The number of documents returned by the similarity search is not 5.")

    text = "「"
    for doc in docs_and_scores:
        text += doc[0].page_content
    text += "」"

    # Using the PromptTemplate class, create a prompt that refers to the 5 sentences above and returns the answer to the question.
    template = query
    prompt = PromptTemplate(
        input_variables=[],
        template=text + "Please refer to the text above and answer the following question in English. " + template,
    )
    # Throw the created prompt to text-davinci-003, get the answer, and display it.
    llm = OpenAI(model_name="text-davinci-003")
    print(llm(prompt.format()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OPEN_API_KEY", type=str, default="")
    parser.add_argument("--faiss_save_path", type=str, default="faiss_pdf_index", help="Please specify the name of the created Faiss object.")
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()

    main(args)
