# coding: utf-8

from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity


dataset = Dataset()
dataset.load_custom_dataset_from_folder("octis_data_20ng")

model = LDA(num_topics=20)

# finding out the names of the hyperparameteres
# import inspect
# print(inspect.getsource(LDA))
"""
    num_topics=100, distributed=False, chunksize=2000,
    passes=1, update_every=1, alpha="symmetric", eta=None, decay=0.5,
    offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001,
"""

trained_model = model.train_model(dataset)

for topic in trained_model["topics"]:
    print(" ".join(topic))

cv = Coherence(texts=dataset.get_corpus(), topk=10, measure="c_v")
print("Coherence: " + str(cv.score(trained_model)))

diversity = TopicDiversity(topk=10)
print("Diversity score: " + str(diversity.score(trained_model)))
