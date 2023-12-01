import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
from rich import print
from streamlit_extras.app_logo import add_logo
from redisvl.query import RangeQuery
from redisvl.index import SearchIndex
from redisvl.vectorize.text import OpenAITextVectorizer

load_dotenv()
EXPLANATION = []
INDEX_NAME = "idx:txn"
# Common Functions


def find_category(index: SearchIndex, narration):
    oai = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": os.getenv("OPENAI_API_KEY")},
    )
    vector = oai.embed(narration)
    EXPLANATION.append(
        f"A vector embedding is created for {narration} by Open AI Model *text-embedding-ada-002*"
    )

    query = RangeQuery(
        vector=vector,
        vector_field_name="Narration_Embedding",
        return_fields=["Category", "Narration", "vector_distance"],
        num_results=1,
        distance_threshold=0.15,
    )

    result = index.query(query)
    EXPLANATION.append(
        f"A KNN1 search is performed between embedding for '{narration}' and embeddings stored for pre-labelled data"
    )
    if result:
        EXPLANATION.append(
            f"The app returned category **'{result[0]['Category']}'** based on similar labelled transaction **{result[0]['Narration']}**"
        )
        return result[0]
    else:
        EXPLANATION.append(f"No similar categories found for **'{narration}'**")
        return False


def get_explanation():
    expl_doc = ""
    for i, txt in enumerate(EXPLANATION):
        expl_doc += f"{i+1} : {txt}<br><br>"
    return expl_doc


def main():
    index = SearchIndex.from_existing(INDEX_NAME, "redis://localhost:6379")

    add_logo("assets/redis-favicon-144x144.png")
    colored_header(
        label="Find Category",
        description="Use Vector Similarity search categorize transactions based on pre-labelled transactions",
        color_name="violet-60",
    )
    form = st.form(key="search_form")
    prompt = form.text_input(label="Enter some text")
    submit = form.form_submit_button(label="Submit")

    if submit:
        result = find_category(index=index, narration=prompt)
        if result:
            st.header(result["Category"])
            st.markdown(
                f"""
                Matched with existing : {result['Narration']}
                <br>
                Vector score : {round(1 - float(result['vector_distance']), 2)}
                """,
                unsafe_allow_html=True,
            )
        else:
            st.header("Uncategorized")

    with form.expander(label="Execution Log"):
        st.markdown(get_explanation(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
