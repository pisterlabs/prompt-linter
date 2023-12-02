import os
import sys
import csv
from gensim import corpora
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Crossovers

mallet_path = sys.argv[1]
workers = int(sys.argv[2])
model_folder = sys.argv[3]
data_folder = sys.argv[4]
prefix = sys.argv[5]

id2word_path = os.path.join(data_folder, prefix + '_id2word.dict')
corpus_path = os.path.join(data_folder, prefix + '_corpus.mm')
raw_path = os.path.join(data_folder, prefix + '_preprocessed')
log_path = os.path.join(data_folder, prefix + '_log.csv')
ga_seed = int(sys.argv[6])
lda_seed = int(sys.argv[7])
iteration_min = int(sys.argv[8])
iteration_max = int(sys.argv[9])
topic_min = int(sys.argv[10])
topic_max = int(sys.argv[11])
PopulationSize = int(sys.argv[12])
Generations = int(sys.argv[13])

def calculate_topic(iteration):
    return int((topic_max - topic_min) / (iteration_max - iteration_min) * (iteration - iteration_min) + topic_min)


id2word = corpora.Dictionary.load(id2word_path)
corpus = corpora.MmCorpus(corpus_path)
texts = []
with open(raw_path) as f:
    for line in f:
        texts.append(line.strip('\n').split(',')[1].split(' '))
print(f'Total number of words: {len(id2word)}')
print(f'Total number of documents in corpus: {len(corpus)}')
print(f'Total number of posts: {len(texts)}')

coherence_scores = {}
logfile = open(log_path, 'w')
writer = csv.writer(logfile)
writer.writerow(['num_topics', 'iterations', 'coherence_score'])

def eval_coherence(paras):
    num_topics = calculate_topic(paras[0])
    iterations = paras[1]
    print(num_topics, iterations, end=' ')

    global coherence_scores
    config = str(num_topics) + '_' + str(iterations)
    if not config in coherence_scores.keys():

        ldamallet = LdaMallet(mallet_path=mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word, workers=workers, 
                              prefix=os.path.join(model_folder, config), optimize_interval=10, iterations=iterations, random_seed=lda_seed)
        cm = CoherenceModel(ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
        score = cm.get_coherence()
        coherence_scores[config] = score
        writer.writerow([num_topics, iterations, score])
    print(coherence_scores[config])
    return coherence_scores[config]  

genome = G1DList.G1DList(2)
genome.evaluator.set(eval_coherence)
genome.setParams(rangemin=iteration_min, rangemax=iteration_max)
genome.crossover.set(Crossovers.G1DListCrossoverUniform)
ga = GSimpleGA.GSimpleGA(genome, seed=ga_seed)
ga.setPopulationSize(PopulationSize)
ga.setGenerations(Generations)
ga.evolve(freq_stats=1)
print(ga.bestIndividual())
params = ga.bestIndividual().genomeList
best_config = [calculate_topic(params[0]), params[1]]
print(f'Best num_topic: {best_config[0]}, best iterations: {best_config[1]}')
logfile.close()