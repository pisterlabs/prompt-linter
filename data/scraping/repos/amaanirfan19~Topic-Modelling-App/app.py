from io import StringIO
import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

st.set_page_config(
    page_title="Topic Modelling using BERTopic",
    page_icon="🎈",
    layout="wide"
)


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns(3)

with c30:
    # st.image("logo.png", width=400)
    st.title("🔑 Topic Modelling using BERTopic")
    st.header("")


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
        """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Fine Tune Your Model! **")

dataframe = None

ce, c1, ce, c2, c3 = st.columns([0.07, 1.5, 0.07, 1.9, 0.01])
with c1:
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
    if uploaded_file is None:
        sample_data = st.checkbox(
            "Try with sample reviews data"
        )
        if sample_data:
            dataframe = pd.read_csv("./vividseats_reviews.csv")

    ModelType = st.radio(
        "Choose your model",
        ["BERTopic (Default)"],
    )

    min_topic_size = st.number_input(
        "Minimum Topic Size",
        min_value=1,
        value=20,
    )

    top_n_words = st.number_input(
        "Top N Words",
        min_value=1,
        value=10
    )

    nr_topics = st.number_input(
        "Total Number of Topics",
        min_value=None,
        step=1,
    )

    embedding_model = st.selectbox(
        'Embedding Model',
        ('',
         'paraphrase-MiniLM-L3-v2',
         'all-mpnet-base-v2',
         'multi-qa-mpnet-base-dot-v1',
         'all-distilroberta-v1',
         'all-MiniLM-L12-v2',
         'multi-qa-distilbert-cos-v1',
         'all-MiniLM-L6-v2',
         'paraphrase-multilingual-mpnet-base-v2',
         'paraphrase-albert-small-v2',
         'paraphrase-multilingual-MiniLM-L12-v2',
         'distiluse-base-multilingual-cased-v1',
         'distiluse-base-multilingual-cased-v2'))

    diversity_checkbox = st.checkbox("Click to set custom diversity")

    Diversity = st.slider(
        "Diversity",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        help="""The higher the setting, the more diverse the keywords.

Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.

""", disabled=not diversity_checkbox
    )

    n_gram_range = st.slider(
        'N Gram Range',
        1, 5, (1, 2))

    low_memory = st.checkbox(
        "Use Low Memory",
        help="UMAP low memory to True to make sure less memory is used",
    )

    with c2:
        if dataframe is not None:
            st.dataframe(dataframe)
            option = st.selectbox(
                'Select the column with the data',
                dataframe.columns.tolist())

            if st.button("Generate Topics"):
                vectorizer_model = CountVectorizer(
                    ngram_range=n_gram_range, stop_words="english")
                docs = dataframe[option]

                ################################################################
                if embedding_model == '':
                    embedding_model = None

                ################################################################
                st.write("Fine Tuning your model...")
                model = BERTopic(
                    top_n_words=top_n_words,
                    min_topic_size=min_topic_size,
                    nr_topics=nr_topics,
                    vectorizer_model=vectorizer_model,
                    embedding_model=embedding_model,
                    low_memory=low_memory
                )

                topics, _ = model.fit_transform(docs)

                barchart = model.visualize_barchart()
                hierarchical = model.visualize_hierarchy()
                bar_plot = st.plotly_chart(barchart)
                hierarchical_plot = st.plotly_chart(hierarchical)
                st.write("Calculating coherence score...")

                documents = pd.DataFrame({"Document": docs,
                                          "ID": range(len(docs)),
                                         "Topic": topics})
                documents_per_topic = documents.groupby(
                    ['Topic'], as_index=False).agg({'Document': ' '.join})
                cleaned_docs = model._preprocess_text(
                    documents_per_topic.Document.values)

                # Extract vectorizer and analyzer from BERTopic
                vectorizer = model.vectorizer_model
                analyzer = vectorizer.build_analyzer()

                # Extract features for Topic Coherence evaluation
                words = vectorizer.get_feature_names()
                tokens = [analyzer(doc) for doc in cleaned_docs]
                dictionary = corpora.Dictionary(tokens)
                corpus = [dictionary.doc2bow(token) for token in tokens]
                topic_words = [[words for words, _ in model.get_topic(topic)]
                               for topic in range(len(set(topics))-1)]

                # Evaluate
                coherence_model = CoherenceModel(topics=topic_words,
                                                 texts=tokens,
                                                 corpus=corpus,
                                                 dictionary=dictionary,
                                                 coherence='c_v')
                coherence = coherence_model.get_coherence()
                st.metric(label="Coherence Score", value=coherence)
