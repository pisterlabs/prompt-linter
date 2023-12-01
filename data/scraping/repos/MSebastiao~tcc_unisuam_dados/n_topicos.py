import gensim
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def main():
    with open('sentencas.txt', 'r', encoding='UTF-8') as f:
            texto_completo = f.read()
            textos = re.split('--------------------------', texto_completo)

    stop_words = set(stopwords.words('portuguese'))
    stop_words_juridico = stopwords_juridico = ['art', 'nº', 'artigo']
    texts = []

    for document in textos:
        tokens = word_tokenize(document.lower())  # Tokenização
        tokens = [token for token in tokens if token.isalpha()]  # Remoção de pontuações e números
        tokens = [token for token in tokens if token not in stop_words]  # Remoção de stopwords
        tokens = [token for token in tokens if token not in stop_words_juridico]
        texts.append(tokens)

    # Criar o dicionário e o corpus
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(doc) for doc in texts]

    # Configurações para as múltiplas execuções
    num_runs = 10
    num_topics_list = range(9, 11)

    best_num_topics = None
    best_coherence_score = -1

    # Executar múltiplas vezes e calcular a média dos scores de coerência
    for num_topics in num_topics_list:
        coherence_scores = []

        for run in range(num_runs):
            lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
            coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)
            print("Número de tópicos:", num_topics, "- Execução", run+1, "- Score de coerência:", coherence_score)

        avg_coherence_score = sum(coherence_scores) / num_runs
        print("Número de tópicos:", num_topics, "- Média do score de coerência:", avg_coherence_score)

        if avg_coherence_score > best_coherence_score:
            best_coherence_score = avg_coherence_score
            best_num_topics = num_topics

    # Número ideal de tópicos com base na maior média de coerência
    print("Número ideal de tópicos:", best_num_topics)

if __name__ == '__main__':
     main()