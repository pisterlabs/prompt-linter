import os
import ast
import streamlit as st
from redisvl.query import RangeQuery
from redisvl.index import SearchIndex
from redisvl.vectorize.text import OpenAITextVectorizer
from streamlit_extras.app_logo import add_logo
from dotenv import load_dotenv
import pandas as pd
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.row import row

INDEX_NAME = "idx:txn"

def find_category(index: SearchIndex, narration_embedding):
    query = RangeQuery(
        vector=narration_embedding,
        vector_field_name="Narration_Embedding",
        return_fields=["Category", "Narration", "vector_distance"],
        num_results=1,
        distance_threshold=0.15,
    )

    result = index.query(query)
    if result:
        return result[0]
    else:
        return {'Category' : False}


def upload_file(vectorizer: OpenAITextVectorizer):
    form = st.form(key="file_upload")
    uploaded_file = form.file_uploader("Choose a file")
    submit = form.form_submit_button(label="Predict Categories")

    if submit:
        df = pd.read_csv(uploaded_file)
        df["Narration_Embedding"] = df.Narration.apply(lambda x: vectorizer.embed(x))
        records = df.to_dict("records")
        st.dataframe(df)
        return records


def process_records(index: SearchIndex, records):
    metrics = {}
    for record in records:
        if not (
            category := find_category(index=index, narration_embedding=record["Narration_Embedding"])[
                "Category"
            ]
        ):
            category = "Uncategorized"
        metrics.setdefault(category, 0)
        metrics[category] += record["Amount"]
    return dict(sorted(metrics.items(), key=lambda item: item[1]))


def render_metric_cards(metrics):
    n_cols_per_row = 3
    i = 0
    r = row(n_cols_per_row, vertical_align="center")
    for category, value in metrics.items():
        i+=1
        r.metric(label=category, value=f"$ {round(value,2)}")
        if i%n_cols_per_row == 0:
            r = row(n_cols_per_row, vertical_align="center")
    
    style_metric_cards()


def main():
    load_dotenv()
    colored_header(
        label="Upload Transaction Data",
        description="Every transaction description will be vectorized and categorized based on vector similarity search",
        color_name="violet-60",
    )
    index = SearchIndex.from_existing(INDEX_NAME, "redis://localhost:6379")
    add_logo("assets/redis-favicon-144x144.png")
    oai = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": os.getenv("OPENAI_API_KEY")},
    )
    if records := upload_file(vectorizer=oai):
        metrics = process_records(index=index, records=records)
        render_metric_cards(metrics=metrics)


if __name__ == "__main__":
    main()
