"""
openai==0.11.4
"""
import argparse
import logging
import os
import pathlib
import pickle

import openai
from transformers import GPT2TokenizerFast

logging.basicConfig(level=logging.INFO)

os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

API_KEY = "YOUR_KEY"

from mteb import MTEB


TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS
class OpenAIEmbedder:
    """
    Benchmark OpenAIs embeddings endpoint on USEB.
    """
    def __init__(self, engine, task_name=None, batch_size=32, save_emb=False, **kwargs):
        self.engine = engine
        self.max_token_len = 2046 # 2048 - 2 special tokens
        self.batch_size = batch_size
        self.save_emb = False # Problematic as the filenames end up being the same
        self.base_path = f"embeddings/{engine.split('/')[-1]}/"
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.task_name = task_name

        if save_emb:
            assert self.task_name is not None

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
        
    def encode(self, 
            sentences,
            decode=True,
            idx=None,
            **kwargs
        ):

        openai.api_key = API_KEY

        fin_embeddings = []

        embedding_path = f"{self.base_path}/{self.task_name}_{sentences[0][:5]}_{sentences[-1][-5:]}.pickle"
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]

                all_tokens = []
                used_indices = []
                for j, txt in enumerate(batch):
                    tokens = self.tokenizer.encode(txt, add_special_tokens=False)
                    token_len = len(tokens)
                    if token_len == 0:
                        raise ValueError("Empty items should be cleaned prior to running")
                    if token_len > self.max_token_len:
                        tokens = tokens[:self.max_token_len]
                    # For some characters the API raises weird errors, e.g. input=[[126]]
                    if decode:
                        tokens = self.tokenizer.decode(tokens)
                    all_tokens.append(tokens)
                    used_indices.append(j)

                out = [[]] * len(batch)
                if all_tokens:
                    response = openai.Engine(id=self.engine).embeddings(input=all_tokens)
                    assert len(response["data"]) == len(
                        all_tokens
                    ), f"Sent {len(all_tokens)}, got {len(response['data'])}"

                    for data in response["data"]:
                        idx = data["index"]
                        # OpenAI seems to return them ordered, but to be save use the index and insert
                        idx = used_indices[idx]
                        embedding = data["embedding"]
                        out[idx] = embedding

                fin_embeddings.extend(out)
        # Save embeddings
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--engine", type=str, default="text-similarity-ada-001")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    args = parser.parse_args()
    return args

def main(args):

    # Different batch size than the arg
    # The below is used to send X embeddings to the API
    # The CLI arg is how much will be saved / pickle file

    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)
        model = OpenAIEmbedder(args.engine, task_name=task, batchsize=256, save_emb=True)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        model_name = args.engine.split("/")[-1].split("_")[-1]
        evaluation = MTEB(tasks=[task], task_langs=[args.lang])
        evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize, eval_splits=eval_splits)

if __name__ == "__main__":
    args = parse_args()
    main(args)
