import pickle
import torch
import sys
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.evaluation.measures import CoherenceNPMI
from contextualized_topic_models.utils.data_preparation import TextHandler

def show_topics(topic_list):
    for idx, topic_tokens in enumerate(topic_list):
        print(idx)
        print(' '.join(topic_tokens))

if len(sys.argv) < 2:
    raise Exception("Usage: python {} {}".format(sys.argv[0], "<model_file>"))

handler = TextHandler("contextualized_topic_models/data/wiki/wiki_train_en_prep.txt")
handler.prepare()
ctm = CTM(input_size=len(handler.vocab), inference_type="contextual", bert_input_size=768, \
    num_data_loader_workers=1)
ctm.load(sys.argv[1], sys.argv[2])
with open("contextualized_topic_models/data/wiki/wiki_train_en_prep.txt", "r") as en:
    texts = [doc.split() for doc in en.read().splitlines()]

# obtain NPMI coherences on the topic modeled documents
show_topics(ctm.get_topic_lists(10))
npmi = CoherenceNPMI(texts=texts, topics=ctm.get_topic_lists(10))
print(npmi.score())
