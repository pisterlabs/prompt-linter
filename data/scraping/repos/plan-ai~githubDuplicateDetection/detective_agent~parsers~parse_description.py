from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.similarities import MatrixSimilarity
from detective_agent.parsers.utils import prepare_corpus, preprocess_data, split_text


def create_gensim_lsa_model(dictionary, doc_term_matrix, number_of_topics):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    # generate LSA model
    lsamodel = LsiModel(
        doc_term_matrix, num_topics=number_of_topics, id2word=dictionary
    )  # train model
    return lsamodel, dictionary


def build_optimized_lsi_model(docs, stop=3, start=1, step=1):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    coherence_models = []
    doc_clean = preprocess_data(split_text(docs))
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    for num_topics in range(start, stop, step):
        # generate LSA model
        model, dictionary = create_gensim_lsa_model(
            dictionary, doc_term_matrix, num_topics
        )
        # train model
        coherencemodel = CoherenceModel(
            model=model, texts=doc_clean, dictionary=dictionary, coherence="c_v"
        )
        coherence_models.append(model)
        coherence_values.append(coherencemodel.get_coherence())
    return coherence_models, coherence_values


def get_similarity(model, initial_doc, doc):
    initial_doc_clean = preprocess_data(split_text(initial_doc))
    dictionary, initial_doc = prepare_corpus(initial_doc_clean)
    corpus_lsi = model[initial_doc]
    doc_clean = preprocess_data(split_text(doc))
    dictionary, new_corpus = prepare_corpus(doc_clean)
    new_corpus_lsi = model[new_corpus]
    similarity_index = MatrixSimilarity(corpus_lsi)
    similarity_scores = similarity_index[new_corpus_lsi]
    return similarity_scores
