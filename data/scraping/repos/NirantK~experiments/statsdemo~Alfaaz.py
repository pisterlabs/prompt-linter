import collections
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import textacy
import textacy.similarity
import textacy.tm
import textacy.vsm
from fastcore.utils import Path
from textacy.spacier.doc_extensions import get_preview, get_tokens

st.title("Alfaaz: Text Exploration Tools")
st.image(
    image="https://verloop.io/wp-content/uploads/2020/08/cropped-VP.io-Website-Grey@2x.png"
)

preview_count = 9
warning_count = 10000
# TODO: Figure out hash function which can be used with corpus objects
def make_corpus(
    df: pd.DataFrame, col_name: str, min_token_count: int
) -> textacy.Corpus:
    spacy_records = df[col_name].apply(lambda x: textacy.make_spacy_doc(x, lang="en"))
    long_records = [
        record for record in spacy_records if len(record) >= min_token_count
    ]
    corpus = textacy.Corpus("en", data=list(long_records))
    return corpus


# @st.cache
# def read_data_to_df(url: str):
#     df = pd.read_json(url)
#     return df

# if st.checkbox("Load Abhibus Demo"):
#     st.markdown("Demo Client: **Abhibus**")
#     api_url = st.text_input(
#         "Please share your Sheet *API* link here",
#         value="https://api.steinhq.com/v1/storages/5f619d225d3cdc44fcd7d4b1",
#     )
#     sheet = st.text_input("Please input your sheet name here:", value="Queries150")
#     col_name = st.text_input(
#         "What is the name of the column which has user queries?", value="QueryText"
#     )
#     data_url = f"{api_url}/{sheet}"
#     df = read_data_to_df(data_url)

import io

st.set_option("deprecation.showfileUploaderEncoding", False)

uploaded_file = st.sidebar.file_uploader(
    "Upload a Tab Separated File", type=["tsv"], accept_multiple_files=False
)
st.sidebar.markdown(
    "If you see an error, please upload a Tab Separated Value (tsv) file."
)
with st.sidebar.beta_expander("See Explanation on TSV files"):
    st.markdown(
        "TSV files are basically same as csv files, but use a tab character to separate columns."
    )
    st.markdown(
        "Since lot of our input might have ',' in text, it's useful to use a different character for separating columns."
    )
    st.image(
        "https://i.ibb.co/KFzdyTv/Screen-Shot-2020-10-20-at-11-42-48-AM.png",
        caption="Download as TSV is the last option in Google Sheets",
    )


@st.cache
def file_io(uploaded_file):
    df = pd.read_csv(uploaded_file, sep="\t")
    return df


if uploaded_file is not None:
    st.markdown("## Data Preview")
    df = file_io(uploaded_file)
    st.markdown(
        f"Here are {preview_count} random samples from the input of {len(df)} rows"
    )
    preview_df = df.sample(preview_count, random_state=37)
    st.write(preview_df)

    st.markdown("### Preparing for Analysis")
    col_name = st.radio("Select Text Column", options=df.columns)

    with st.beta_expander("Change Minimum Sentence Length"):
        st.info(
            "Use a large number for quick analysis in the beginning (>5) and reduce when you are going for depth"
        )
        min_token_count = st.number_input(
            "Analyze sentences which have atleast how many tokens?",
            value=4,
            min_value=2,
            max_value=10,
        )
    print("Making Corpus Now!")
    corpus = make_corpus(df, col_name=col_name, min_token_count=min_token_count)
    st.write(
        "Records for Analysis:",
        corpus.n_docs,
        "Total Sentences:",
        corpus.n_sents,
        "Total Tokens:",
        corpus.n_tokens,
    )

mode = st.sidebar.radio(
    "Analytics Mode:",
    options=[
        "Word Frequencies",
        "Topics",
        "Similar Sentences",
        "Sentence Explorer",
        "Generate Sentences",
    ],
)

if mode == "Word Frequencies" and uploaded_file is not None:
    st.markdown("## Word Frequencies Exploration")

    if st.sidebar.checkbox("Ignore Case (Recommended)", value=True):
        freq_dict: Dict = corpus.word_counts(
            as_strings=True, normalize="lower", filter_nums=True, filter_punct=True
        )
    else:
        freq_dict: Dict = corpus.word_counts(
            as_strings=True, filter_nums=True, filter_punct=True
        )

    freq_dict = {
        k: v
        for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)
        if v >= 2
    }

    index_key = "Frequency"
    freq_df = pd.DataFrame(freq_dict, index=[index_key]).transpose()

    min_freq_count = st.sidebar.slider(
        "What is the minimum frequency of words you want to see?",
        value=int(freq_df[index_key].quantile(0.81)),
        min_value=2,
        max_value=int(freq_df[index_key].quantile(0.999)),
    )
    freq_df = freq_df[freq_df[index_key] > min_freq_count]
    left, right = st.beta_columns([2, 1])
    left.bar_chart(freq_df)
    word_display_count = min(20, len(freq_df))
    right.markdown(f"Top {word_display_count} words by frequency")
    right.table(freq_df[:word_display_count])

    with st.beta_expander("Filter By Category Values"):
        col_list = list(df.columns)
        col_list.remove(col_name)
        filter_col_name = st.selectbox(
            "Select Column which you want to filter by", options=col_list
        )
        if filter_col_name is not None:
            filter_val = st.selectbox(
                "Select Values to Filter by", options=df[filter_col_name].unique()
            )
            filter_df = pd.DataFrame(df[df[filter_col_name] == filter_val][col_name])
            filter_corpus = make_corpus(
                filter_df, col_name=col_name, min_token_count=min_token_count
            )
            st.info(
                f"You are exploring {filter_corpus.n_docs} text queries. \n This is a subset of total -- where {filter_col_name} value is: {filter_val}"
            )
            # st.dataframe(filter_df.sample(preview_count, random_state=37))

            filter_freq_dict: Dict = filter_corpus.word_counts(
                as_strings=True, normalize="lower", filter_nums=True, filter_punct=True
            )
            filter_freq_dict = {
                k: v
                for k, v in sorted(
                    filter_freq_dict.items(), key=lambda item: item[1], reverse=True
                )
                if v >= 2
            }
            filter_freq_df = pd.DataFrame(
                filter_freq_dict, index=[index_key]
            ).transpose()
            left_filter, right_filter = st.beta_columns([2, 1])
            left_filter.bar_chart(filter_freq_df)
            word_display_count = min(10, len(filter_freq_df))
            right_filter.markdown(f"Top {word_display_count} words by frequency")
            right_filter.table(filter_freq_df[:word_display_count])


if mode == "Topics" and uploaded_file is not None:
    st.markdown("## Topics Exploration")

    def make_doc_term_matrix(
        corpus: textacy.Corpus,
        min_freq_count: int,
        ngrams: int = 1,
        entities: bool = True,
    ):
        vectorizer = textacy.vsm.Vectorizer(
            tf_type="linear",
            apply_idf=True,
            idf_type="smooth",
            norm="l2",
            min_df=min_freq_count,
            max_df=0.95,
        )
        doc_term_matrix = vectorizer.fit_transform(
            (
                doc._.to_terms_list(ngrams=ngrams, entities=entities, as_strings=True)
                for doc in corpus
            )
        )
        return vectorizer, doc_term_matrix

    ngrams = st.sidebar.number_input("ngrams:", value=2, min_value=1, max_value=3)
    n_topics = st.sidebar.number_input(
        "How many topics do you want to explore?", value=5, min_value=2, max_value=10
    )

    if corpus.n_docs > warning_count:
        st.warning(
            f"Your corpus has more than {warning_count} records. This can be painfully slow. Consider increasing the Minimum Sentence Length"
        )

    with st.spinner(text="Building a Topic Model"):
        vectorizer, doc_term_matrix = make_doc_term_matrix(
            corpus, min_freq_count=min_token_count, ngrams=ngrams
        )
        model = textacy.tm.TopicModel("nmf", n_topics=n_topics)
        model.fit(doc_term_matrix)
        doc_topic_matrix = model.transform(doc_term_matrix)

    doc_topic_matrix.shape
    for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=5):
        st.markdown(f"Topics {topic_idx+1} : {'; '.join(top_terms)}")

if mode == "Similar Sentences" and uploaded_file is not None:
    "## Similar Sentences"
    if corpus.n_docs > 1000:
        st.warning(
            f"Your search has more than {warning_count} records. This can be slow"
        )

    input_sentence = st.text_input(
        "Enter your sentence to which you want to find similar examples",
        value="train ticket cancel",
        max_chars=300,
    )
    input_doc = textacy.make_spacy_doc(input_sentence, lang="en")

    sent_count = st.slider(
        "Maximum number of Similar sentences that you want",
        min_value=1,
        max_value=10,
        value=3,
    )
    similarity_cutoff = (
        st.slider("Similarity Cutoff", min_value=1, max_value=100, value=70) / 100.0
    )
    for i, doc in enumerate(corpus):
        score = textacy.similarity.word2vec(input_doc, doc)
        if score >= similarity_cutoff and sent_count > 0:
            st.write(doc, f"{score:.2f}")
            sent_count -= 1
        elif sent_count <= 0:
            st.write("---*Finished*---")
            break

if mode == "Sentence Explorer" and uploaded_file is not None:
    "## Find Sentences with Specific Words"
    input_sentence = st.text_input(
        "Enter the words (separated by space)", value="ticket", max_chars=300
    )
    input_doc = textacy.make_spacy_doc(input_sentence, lang="en")
    input_tokens: set = set([str(x) for x in get_tokens(input_doc)])
    sent_count = st.slider(
        "Maximum number of sentences that you want", min_value=1, max_value=10, value=3
    )

    for i, doc in enumerate(corpus):
        doc_tokens = set([str(x) for x in get_tokens(doc)])
        if len(input_tokens.intersection(doc_tokens)) >= len(input_tokens):
            st.write(doc)
            sent_count -= 1
        if sent_count == 0:
            break

if mode == "Generate Sentences":

    "### This works for BANKING ONLY"

    input_sentence = st.text_input(
        "Enter sentence which you want to augment?",
        value="When is my credit card bill due?",
    )
    min_freq_count = st.slider(
        "How many tries do you want to make?", value=1, min_value=1, max_value=5
    )

    intent = st.text_input("Enter a 2 word intent please", value="Due Date")
    append_phrase = f"Sentence: {input_sentence}\nIntent: {intent}\n Paraphrase:"

    prompt = (
        """
    Sentence :  Hello, I want to know about my card status so can you please tell me?
    Intent : Card Status
    Paraphrase : Hello, Can you tell me about the card status? It would be really nice if I can get this information
    Sentence :  Hello, I wanted to know my payment due
    Intent : Payment Amount Due
    Paraphrase : I want information about the payment amount due, who do I contact about this?
    Sentence :  HELLO, I want to register for a security PIN because I want to use it for online transactions.
    Intent : e-PIN Registration
    Paraphrase : I want to use e-pin for online transactions so I would like to know about how to register for security pin
    Sentence :  I want to know the status of my covered card/debit card because i am going to travel.
    Intent : Card Status
    Paraphrase : I am going to travel. Can you tell me about eh covered card status
    Sentence :  I would like to donate to Zakat because I want to be good.
    Intent : Donation
    Paraphrase : I want to donate money out of my goodness
    Sentence :  I want to know what is my covered card outstanding
    Intent : Payment Amount Due
    Paraphrase : What is the outstanding balance for the covered card? I would like to know that but I am unable to find these details anywhere?
    Sentence :  Hello, I want to know what was the last debit transaction on my Debit card because I want to know if somebody is using it.
    Intent : Debit Transaction
    Paraphrase : Hello, what was the last debit transaction on my card because I suspect someone else is using it.
    Sentence :  HELLO, I FORGOT MY E-PIN SO CAN YOU RESET IT FOR ME PLEASE?
    Intent : e-PIN Reset
    Paraphrase : I want to reset my e-pin because I forgot.
    Sentence :  Send me a covered card statement because I want to check my account.
    Intent : Covered Card e-statement
    Sentence:
    """
        + append_phrase
    )

    input_api_key = st.text_input("Enter your GPT3 key")
    engine_id = st.selectbox(
        "Engine Id", options=["davinci", "davinci-beta", "curie", "curie-beta"]
    )

    import openai

    openai.api_key = input_api_key

    def calling_our_simulation_overlords(prompt: str, min_freq_count: int):
        response = openai.Completion.create(
            engine="davinci-beta",
            prompt=prompt,
            max_tokens=50,
            temperature=0.9,
            frequency_penalty=0.8,
            n=min_freq_count,
            stop="Sentence",
        )
        return response

    try:
        response = calling_our_simulation_overlords(
            prompt=prompt, min_freq_count=min_freq_count
        )
    except Exception as e:
        raise (
            f"You will need a GPT3 key. Please contact Deepak Singh for the same. Do NOT share the key with your teammates. This key might be reset every 5-7 days.\nDeveloper Info: {e}"
        )
    generated_sentences = [
        choice["text"].strip() for choice in response.to_dict()["choices"]
    ]

    "### Output Sentence"
    for idx, sentence in enumerate(list(set(generated_sentences))):
        st.markdown(f"{idx+1}) {sentence}")
        st.markdown("\n")
