import argparse
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from utils import load_data

from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.evaluation.measures import CoherenceCV, InvertedRBO, TopicDiversity
from contextualized_topic_models.datasets.dataset import CTMDataset
import nltk

import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, default="train.pkl", required=False)
parser.add_argument("--n_topics", type=int, default=20, required=False)
parser.add_argument("--model_type", type=str, default="LDA", choices=["LDA", "prodLDA"], required=False)
parser.add_argument("--tries", type=int, default=5, required=False)
parser.add_argument("--num_epochs", type=int, default=100, required=False)
parser.add_argument("--beta", type=float, default=1.0, required=False)
args = parser.parse_args()

nltk.download('stopwords')

texts, images = load_data(args.train)

sp = WhiteSpacePreprocessing(texts, stopwords_language='german')

preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()
tp = TopicModelDataPreparation("clip-ViT-B-32-multilingual-v1")


print("Text preprocessing")
text_training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
preprocessed_split = [t.split() for t in preprocessed_documents]

img_model = SentenceTransformer('clip-ViT-B-32')
print("Encoding images")
img_emb = img_model.encode(images, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
img_emb = np.array(img_emb.cpu())
image_training_dataset = CTMDataset(X_contextual = img_emb, X_bow = text_training_dataset.X_bow, idx2token = text_training_dataset.idx2token)

print("Joining")
joint_embs = np.concatenate((text_training_dataset.X_contextual, img_emb), axis=1)
print("JOINT_EMBS: ", joint_embs.shape)
training_dataset = CTMDataset(X_contextual = joint_embs, X_bow = text_training_dataset.X_bow, idx2token = text_training_dataset.idx2token)

for t in range(args.tries):
    model_dir = "models/joint/%s_%s_%s_%s_%s/" %(args.model_type, args.n_topics, args.num_epochs, args.beta, t)
    os.makedirs(model_dir)
    log = open(os.path.join(model_dir,"log.txt"), "w")
    print("Train file: %s" %os.path.abspath(args.train), file=log)
    pickle.dump(training_dataset, open(os.path.join(model_dir, "training_dataset.pkl"), "wb"))
    pickle.dump(text_training_dataset, open(os.path.join(model_dir, "text_training_dataset.pkl"), "wb"))
    pickle.dump(image_training_dataset, open(os.path.join(model_dir, "image_training_dataset.pkl"), "wb"))
    pickle.dump(tp, open(os.path.join(model_dir, "tp.pkl"), "wb"))
    pickle.dump(preprocessed_split, open(os.path.join(model_dir, "preprocessed_split.pkl"), "wb"))

    loss_weights = {"beta":args.beta}
    ctm = ZeroShotTM(bow_size=len(tp.vocab),
                     contextual_size=joint_embs.shape[1], 
                     n_components=args.n_topics, 
                     model_type=args.model_type, 
                     num_epochs = args.num_epochs, 
                     loss_weights=loss_weights)
    ctm.fit(training_dataset, save_dir=model_dir, verbose=True)
    
    topics = ctm.get_topic_lists(25)
    irbo = InvertedRBO(topics=topics)
    td = TopicDiversity(topics=topics)   
    cv = CoherenceCV(texts=preprocessed_split, topics=topics)

    print("IRBO: %2.4f" %irbo.score(), file=log)
    print("Diversity: %2.4f" %td.score(), file=log)
    print("Coherence: %2.4f" %cv.score(), file=log)
    log.close()

