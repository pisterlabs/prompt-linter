# coding: utf-8

from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence

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

search_space = {"alpha": Real(low=0.001, high=5.0), "eta": Real(low=0.001, high=5.0), "num_topics": Integer(10, 25)}

# Initialize an optimizer object and start the optimization.
optimizer = Optimizer()
coherence = Coherence(texts=dataset.get_corpus(), topk=10, measure="c_npmi")
opt_result = optimizer.optimize(model,
                                dataset,
                                metric=coherence,
                                search_space=search_space,
                                save_path="octis_results",
                                number_of_call=30,
                                model_runs=5)

# save the results of the optimization in a CSV file
opt_result.save_to_csv("results.csv")
