from openai.embeddings_utils import cosine_similarity, get_embedding

from util import EMBEDDING_MODEL


def embedding(sentence):
    return get_embedding(sentence, engine=EMBEDDING_MODEL)


label_embeddings = [embedding(label) for label in ["Relevent with foodborne illness.", "Not Relevent with foodborne illness."]]


def label_score(review_embedding):
    return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])


def TRC(sentence: str, model=None):
    if model is None:
        probas = label_score(get_embedding(sentence, engine=EMBEDDING_MODEL), label_embeddings)
        return 1 if probas > 0 else 0
    else:
        return model.inference(sentence, embedding(sentence))
