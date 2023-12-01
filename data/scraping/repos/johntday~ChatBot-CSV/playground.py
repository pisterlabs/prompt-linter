import os

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

DB_NAME = 'notion_hybris_faiss_index'

if __name__ == '__main__':
    do_something()


def fetch_vector_store(name=DB_NAME) -> FAISS:
    vector_db = FAISS.load_local(f"embeddings/{name}", OpenAIEmbeddings())
    print(f"\nLoaded '{name}'")
    return vector_db


def extract_keywords():
    load_dotenv(verbose=True)

    NOTION_TOKEN = os.getenv("NOTION_TOKEN")
    NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

    db = fetch_vector_store()

    db_dict = db.docstore._dict
    documents = list(db_dict.values())
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    f = open("keywords.txt", "a")
    # f.write("Now the file has more content!")

    llm = OpenAI(
        model="text-davinci-003",
        # prompt="Extract keywords from this text:\n\nBlack-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors.",
        temperature=0.5,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
    )

    for doc in documents:
        keywords = llm(f"Extract keywords from this text:\n\n{doc}")
        f.write(keywords)
        # print(keywords)
        # break

    f.close()
