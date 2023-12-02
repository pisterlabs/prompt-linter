
import datetime
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time
import regex as re

print("CTM is running ------------------------")


DATE = datetime.datetime.today().strftime("%b_%d_%Y")
print(DATE)

FOLDER_PATH = "---"

# converting the cleaned text column to a tsv
FILE_NAME = 'corpus.tsv'
data = pd.read_csv(FOLDER_PATH + "corpus.tsv", header=None)

from bertopic import BERTopic
from octis.dataset.dataset import Dataset


dataset = Dataset()
dataset.load_custom_dataset_from_folder(FOLDER_PATH)

from octis.models.CTM import CTM
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


def ctm_model_output(i):

    model = CTM(num_topics=i, num_epochs=30, inference_type='zeroshot', bert_model="all-mpnet-base-v2")
    model_output_ctm = model.train_model(dataset)

    topic_diversity = TopicDiversity(topk=10) # Initialize metric
    topic_diversity_score = topic_diversity.score(model_output_ctm)
    print("Topic diversity: "+str(topic_diversity_score))


    # Initialize metric
    npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')
    npmi_score = npmi.score(model_output_ctm)
    print("Coherence: "+str(npmi_score))



for i in [10, 20, 30, 50]:
    ctm_model_output(i)