import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_extras.app_logo import add_logo
from redisvl.index import SearchIndex
from redisvl.vectorize.text import OpenAITextVectorizer
from streamlit_extras.colored_header import colored_header

st.set_page_config(
    layout="wide",
)

def upload_file(vectorizer: OpenAITextVectorizer, output_file_path):
    form = st.form(key="file_upload")
    uploaded_file = form.file_uploader("Choose a file")
    submit = form.form_submit_button(label="Generate Embeddings")

    if submit:
        df = pd.read_csv(uploaded_file)
        df["Narration_Embedding"] = df.Narration.apply(
            lambda x: np.array(vectorizer.embed(x), dtype=np.float32).tobytes()
        )
        records = df.to_dict("records")
        create_and_load_index(records)
        st.dataframe(df.drop(columns=["Narration_Embedding"], axis=1))


def create_and_load_index(records):
    index = SearchIndex.from_yaml("index.yaml")
    index.connect("redis://localhost:6379")
    index.create(overwrite=True)
    index.load(records)


def main():
    load_dotenv()
    colored_header(
        label="Load Pre-Labelled Data",
        description="Every transaction description will be vectorized and stored in the redis database",
        color_name="violet-60",
    )
    add_logo("assets/redis-favicon-144x144.png")
    oai = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": os.getenv("OPENAI_API_KEY")},
    )
    upload_file(vectorizer=oai, output_file_path="data/embeddings.csv")


if __name__ == "__main__":
    main()
