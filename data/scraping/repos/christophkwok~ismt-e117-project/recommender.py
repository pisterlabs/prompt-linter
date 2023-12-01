import os
import re
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import contractions
import spacy
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import tensorflow_hub as hub

from utils import get_checkpoint


def parse_args():
    model_choices = [
        "description_count",
        "description_tfidf",
        "authors_categories_count",
        "authors_categories_tfidf",
        "authors_categories_lda_count",
        "authors_categories_lda_tfidf",
        "title_description_universal_sentence_encoder",
    ]
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--title",
        type=str,
        default="The Four Loves",
        help="Specify the title that you wish to use to get content-based recommendations for.",
    )
    argparser.add_argument(
        "--recommender_model",
        type=str,
        default="description_count",
        choices=model_choices,
        help="Specify the name of the recommender you wish to use.",
    )
    argparser.add_argument(
        "--num_recommendations",
        type=int,
        default=5,
        help="Number of recommendations to output.",
    )
    argparser.add_argument(
        "--run_sample",
        action="store_true",
        help="If this flag is included, the script will run the run_sample method.",
    )
    argparser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Filename of output file for results.",
    )
    argparser.add_argument("--models_dir", type=str, default="models")
    return argparser.parse_args()


### 0 ###
### general preprocessing
def general_preprocessing(df):
    # check uniqu values
    # print(df.nunique())

    # check missing values
    # print(len(df) - df.count())  # 3x faster than df.isna().sum()
    # only keep title, description, author, categories for use in recommender methods and saving in checkpoints
    df = df[["title", "description", "authors", "categories"]]
    df = df[df.title.notnull()]
    df = df[df.description.notnull()]
    # This helps drop foreign languages and strange characters
    df["description"] = df.description.str.replace(r'[^\x00-\x7F]+', "")
    df["title"] = df.title.str.replace(r'[^\x00-\x7F]+', "")
    df = df[df.description != ""]
    df = df[df.title != ""]
    df = df.reset_index(drop=True)
    return df


### 1 ###
### recommendations based on description

# clean data (need further check!)
#
def generate_indices_mapping(df):
    # NOTE: duplicate titles in the index still exist - must drop duplicates in titles
    # Construct a reverse map of indices and book titles. Convert the index into series
    indices = pd.concat(
        [
            pd.Series(df.index, name="index"),
            df["description"],
            df["authors"],
            df["categories"],
        ],
        axis=1,
    )
    indices.index = df["title"]
    return indices


def description_analysis(df, tfidf=True):

    # Define a TF-IDF or Count Vectorizer Object. Remove all english stop words such as 'the', 'a'
    vectorizer = TfidfVectorizer() if tfidf else CountVectorizer()

    # Construct the required TF-IDF matrix by fitting and transforming the data
    matrix = vectorizer.fit_transform(df["description"])
    matrix.shape

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(matrix, matrix)

    indices = generate_indices_mapping(df).drop_duplicates()

    return {"cosine_sim": cosine_sim, "indices": indices, "df": df}


### Function that takes in book title as input and outputs most similar books
def get_recommendations(title, cosine_sim, df, indices, num_recommendations=5):
    # Get the index of the book that matches the title
    idx = indices.loc[title]["index"]

    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1 : 1 + num_recommendations]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return df[["title", "description"]].iloc[book_indices].drop_duplicates()


### 2 ###
### recommendations based on authors and categories (NER removed as it took too long time)


def preprocess_authors_categories(df):
    # covert authors and categories from str to lists
    df["authors"] = df["authors"].str.split(";")
    df["categories"] = df["categories"].str.split(";")

    # select features
    features = ["authors", "categories"]

    # clean data (need further check!)
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # Check if value exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ""

    for feature in features:
        df[feature] = df[feature].apply(clean_data)

    # everything should be in place, but like to still return the df just in case
    return df


"""
###### NER tagging (tried it but took very too long)
import nltk
from nltk.tag import StanfordNERTagger as snt
from nltk.parse.corenlp import CoreNLPParser

stg = snt('C:/Users/moda1/AppData/Roaming/nltk_data/stanfordNLP/stanford-ner-4.0.0/classifiers/english.conll.4class.distsim.crf.ser.gz','C:/Users/moda1/AppData/Roaming/nltk_data/stanfordNLP/stanford-ner-4.0.0/stanford-ner.jar', encoding='utf-8')
parser = CoreNLPParser(url='http://localhost:64000')

df2 = df.sample(100)

def nertagging(description):
    description = str(description)
    nertagged = stg.tag(description.split())
    
    nertag = []
    for x, y in nertagged:
        if y != "O":
            nertag.append(x)
    
    nertag = list(dict.fromkeys(nertag))
    return nertag

df2['nertag'] = df2['description'].apply(lambda x : nertagging(x))
"""


def authors_categories_analysis(df, tfidf=True):
    # we make a copy to avoid mutating the original dataframe
    df = df.copy()

    df = preprocess_authors_categories(df)

    # create a "metadata soup"
    def create_soup(x):
        return " ".join(x["authors"]) + " " + " ".join(x["categories"])

    df["soup"] = df.apply(create_soup, axis=1)

    # Define a vectorizer depending on tfidf parameter
    vectorizer = (
        TfidfVectorizer(stop_words="english")
        if tfidf
        else CountVectorizer(stop_words="english")
    )
    matrix = vectorizer.fit_transform(df["soup"])
    matrix.shape

    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim = cosine_similarity(matrix, matrix)

    # Reset index of our main DataFrame and construct reverse mapping as before
    df = df.reset_index()
    indices = generate_indices_mapping(df)

    return {"cosine_sim": cosine_sim, "indices": indices, "df": df}


### 3 ###
### recommendations based on topics (using LDA) + authors and categories


def generate_lda_dependencies(df):
    df = df.copy()

    # prepare NLTK Stop words
    stop_words = stopwords.words("english")
    stop_words.extend(["from", "subject", "re", "edu", "use", "book"])  # add as needed

    # clean data (need further check!)
    data = df.description.values.tolist()

    def clean_data(sentences):
        for text in sentences:
            text = contractions.fix(text)
            text = re.sub("\S*@\S*\s?", "", text)  # remove emails
            text = re.sub("\s+", " ", text)  # remove newline chars
            text = re.sub("'", "", text)  # remove single quotes
            text = gensim.utils.simple_preprocess(
                str(text), deacc=True
            )  # lowercase & tokenize + deacc=True removes punctuations
            yield (text)

    data_words = list(clean_data(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(
        data_words, min_count=5, threshold=100
    )  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stopwords, Form Bigrams, Trigrams and Lemmatization
    def process_words(
        texts, stop_words=stop_words, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
    ):
        texts = [
            [word for word in simple_preprocess(str(doc)) if word not in stop_words]
            for doc in texts
        ]  # remove stop words
        texts = [bigram_mod[doc] for doc in texts]  # make bigrams
        texts = [trigram_mod[bigram_mod[doc]] for doc in texts]  # make trigrams
        texts_out = []  # lemmatize
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(
                [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            )
        # remove stopwords once more after lemmatization
        texts_out = [
            [word for word in simple_preprocess(str(doc)) if word not in stop_words]
            for doc in texts_out
        ]
        return texts_out

    data_ready = process_words(data_words)  # processed Text Data!

    ## Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    return id2word, corpus, data_ready


def authors_categories_lda_analysis(df, tfidf=True):
    df = df.copy()
    df = preprocess_authors_categories(df)

    id2word, corpus, data_ready = generate_lda_dependencies(df)

    # Build LDA model (need further check of parameters!)
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=30,
        random_state=100,
        update_every=1,
        chunksize=50,
        passes=10,
        alpha="symmetric",
        iterations=100,
        per_word_topics=True,
    )

    doc_lda = lda_model[corpus]

    # Compute Perplexity (the lower, the better)
    print("\nPerplexity: ", lda_model.log_perplexity(corpus))

    # Compute Coherence Score (the higher, the better)
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=data_ready, dictionary=id2word, coherence="c_v"
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print("\nCoherence Score: ", coherence_lda)

    # Finding the dominant topic in each sentence
    topics_df1 = pd.DataFrame()
    topics_df2 = pd.DataFrame()
    topics_df3 = pd.DataFrame()

    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if len(row) >= 3:
                if j == 0:
                    topics_df1 = topics_df1.append(
                        pd.Series(int(topic_num)), ignore_index=True
                    )
                elif j == 1:
                    topics_df2 = topics_df2.append(
                        pd.Series(int(topic_num)), ignore_index=True
                    )
                elif j == 2:
                    topics_df3 = topics_df3.append(
                        pd.Series(int(topic_num)), ignore_index=True
                    )
                else:
                    break
            elif len(row) == 2:
                if j == 0:
                    topics_df1 = topics_df1.append(
                        pd.Series(int(topic_num)), ignore_index=True
                    )
                elif j == 1:
                    topics_df2 = topics_df2.append(
                        pd.Series(int(topic_num)), ignore_index=True
                    )
                    topics_df3 = topics_df3.append(
                        pd.Series(
                            "",
                        ),
                        ignore_index=True,
                    )
            elif len(row) == 1:
                topics_df1 = topics_df1.append(
                    pd.Series(int(topic_num)), ignore_index=True
                )
                topics_df2 = topics_df2.append(pd.Series(""), ignore_index=True)
                topics_df3 = topics_df3.append(pd.Series(""), ignore_index=True)

    topics_df1.rename(columns={0: "1st_Topic"}, inplace=True)
    topics_df2.rename(columns={0: "2nd_Topic"}, inplace=True)
    topics_df3.rename(columns={0: "3rd_Topic"}, inplace=True)
    topics_df1 = topics_df1.astype(int).astype(str)

    # combine topics dataframe to original df
    df = pd.concat([df, topics_df1, topics_df2, topics_df3], axis=1, sort=False)

    # create a "metadata soup 2"
    def create_soup2(x):
        return (
            "".join(str(x["1st_Topic"]))
            + " "
            + "".join(str(x["2nd_Topic"]))
            + " "
            + "".join(str(x["3rd_Topic"]))
            + " "
            + " ".join(x["authors"])
            + " "
            + " ".join(x["categories"])
        )

    df["soup2"] = df.apply(create_soup2, axis=1)

    # Define a vectorizer
    vectorizer = (
        TfidfVectorizer(stop_words="english")
        if tfidf
        else CountVectorizer(stop_words="english")
    )
    matrix = vectorizer.fit_transform(df["soup2"])
    matrix.shape

    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim = cosine_similarity(matrix, matrix)

    # Reset index of our main DataFrame and construct reverse mapping as before
    df = df.reset_index()
    indices = generate_indices_mapping(df)

    return {
        "cosine_sim": cosine_sim,
        "indices": indices,
        "df": df,
    }


def title_description_universal_sentence_encoder(
    df, weight_title=0.5, weight_description=1
):
    df = df.copy()

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    title_embeddings, description_embeddings = embed(df["title"]), embed(
        df["description"]
    )

    # generate two similarity matrices - one for title, one for description
    # they should be the same shape
    title_similarity = linear_kernel(title_embeddings, title_embeddings)
    description_similarity = linear_kernel(
        description_embeddings, description_embeddings
    )

    # combine into one matrix by averaging values
    combined_similarity = (
        weight_title * title_similarity + weight_description * description_similarity
    ) / 2

    return {
        "cosine_sim": combined_similarity,
        "indices": generate_indices_mapping(df),
        "df": df,
    }


def run_sample():
    df = general_preprocessing(pd.read_csv("data/books.csv"))

    # Collect the matrices and other parameters that will be used by get_recommendations (besides title)
    # this will be a nested dictionary
    trained = {
        "description_count": get_checkpoint(
            "models/description_count.chkpt", description_analysis, df, tfidf=False
        ),
        "description_tfidf": get_checkpoint(
            "models/description_tfidf.chkpt", description_analysis, df, tfidf=True
        ),
        "authors_categories_count": get_checkpoint(
            "models/authors_categories_count.chkpt",
            authors_categories_analysis,
            df,
            tfidf=False,
        ),
        "authors_categories_tfidf": get_checkpoint(
            "models/authors_categories_tfidf.chkpt",
            authors_categories_analysis,
            df,
            tfidf=True,
        ),
        "authors_categories_lda_count": get_checkpoint(
            "models/authors_categories_lda_count.chkpt",
            authors_categories_lda_analysis,
            df,
            tfidf=False,
        ),
        "authors_categories_lda_tfidf": get_checkpoint(
            "models/authors_categories_lda_tfidf.chkpt",
            authors_categories_lda_analysis,
            df,
            tfidf=True,
        ),
        "title_description_universal_sentence_encoder": get_checkpoint(
            "models/title_description_universal_sentence_encoder.chkpt",
            title_description_universal_sentence_encoder,
            df,
        ),
    }

    samples = ["The Four Loves", "Blink-182", "The Rule of Four", "Cypress Gardens"]

    # iterate through all our trained matrices and print out recommendations
    for recommender_name, checkpoint in trained.items():

        # quick patch for now
        if checkpoint is None:
            continue

        cosine_sim = checkpoint["cosine_sim"]
        indices = checkpoint["indices"]
        df = checkpoint["df"]

        if args.output_csv:
            df_to_write = pd.DataFrame()

        for book_title in samples:
            print("\nmethod: {}, title - {}:\n".format(recommender_name, book_title))

            description = indices.loc[book_title]["description"]
            print("description: {}\n".format(description))

            recommendations = get_recommendations(
                book_title, cosine_sim=cosine_sim, df=df, indices=indices
            )
            print(recommendations)

            if args.output_csv:
                # add input title and description to dataframe
                recommendations["input_title"] = book_title
                recommendations["input_description"] = description
                recommendations["method"] = recommender_name

                df_to_write = df_to_write.append(recommendations)

        if args.output_csv:
            # reorder columns
            df_to_write = df_to_write[
                ["method", "input_title", "input_description", "title", "description"]
            ]
            print(df_to_write)
            df_to_write.to_csv(args.output_csv)
        print("\n\n\n")


def main(args=None):
    if args is None:
        raise ValueError("Must supply arguments to main")

    model_to_function_map = {
        "description_count": (description_analysis, {"tfidf": False}),
        "description_tfidf": (description_analysis, {"tfidf": True}),
        "authors_categories_count": (authors_categories_analysis, {"tfidf": False}),
        "authors_categories_tfidf": (authors_categories_analysis, {"tfidf": True}),
        "authors_categories_lda_count": (
            authors_categories_lda_analysis,
            {"tfidf": False},
        ),
        "authors_categories_lda_tfidf": (
            authors_categories_lda_analysis,
            {"tfidf": True},
        ),
        "title_description_universal_sentence_encoder": (
            title_description_universal_sentence_encoder,
            {},
        ),
    }

    df = general_preprocessing(pd.read_csv("data/books.csv"))

    function, kwargs = model_to_function_map[args.recommender_model]
    checkpoint = get_checkpoint(
        os.path.join(args.models_dir, args.recommender_model + ".chkpt"),
        function,
        df,
        **kwargs
    )

    cosine_sim = checkpoint["cosine_sim"]
    indices = checkpoint["indices"]
    df = checkpoint["df"]

    print("\nmethod: {}, title - {}:\n".format(args.recommender_model, args.title))
    print("description: {}\n".format(indices.loc[args.title]["description"]))
    series_recommendations = get_recommendations(
        args.title,
        cosine_sim=cosine_sim,
        df=df,
        indices=indices,
        num_recommendations=args.num_recommendations,
    )

    print(series_recommendations)
    return series_recommendations


if __name__ == "__main__":
    # this portion will only run if we execute the script directly
    # it will not run if we import something from here in another file

    args = parse_args()
    if args.run_sample:
        run_sample()
    else:
        main(args=args)
