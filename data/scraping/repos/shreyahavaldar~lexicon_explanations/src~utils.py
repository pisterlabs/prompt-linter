import torch
import os
import torch
import numpy as np
import shap
import transformers
from datasets import load_dataset, Dataset, Sequence, Value, Features
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from src.pair_data import LIWCWordData
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import nltk
from sklearn.model_selection import train_test_split
import string
from sklearn.preprocessing import MultiLabelBinarizer

from octis.preprocessing.preprocessing import Preprocessing
from octis.dataset.dataset import Dataset as octDataset
from octis.models.LDA import LDA
from octis.models.NeuralLDA import NeuralLDA
from octis.models.ETM import ETM
from octis.models.CTM import CTM
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence
import csv
import random


def save(obj, name):
    base_path = Path(__file__).parent
    torch.save(obj, base_path / "../output" /  (name + ".pt"))

def load(name):
    base_path = Path(__file__).parent
    if os.path.isfile(base_path / "../output" / (name + ".pt")):
        return torch.load(base_path / "../output" / (name + ".pt"))
    else:
        return None

def word_shap(token_strs, shap_values):
    def normalize(word):
        return word.strip().lower()

    # print(len(token_strs), len(shap_values))
    vals = [np.zeros(shap_values.shape[1])]
    words = [""]
    for tok, val in zip(token_strs, shap_values):
        # print("words:", words)
        # For GPT2 lexer, if the token starts with space, then this is the start
        # of a new word
        if tok.startswith(" "):
            words[-1] = normalize(words[-1])
            words.append("")
            vals.append(np.zeros(shap_values.shape[1]))

        # Punctuation should not be added to existing words
        if any(char in set(string.punctuation) for char in tok):
            words[-1] = normalize(words[-1])
            words.append("")
            vals.append(np.zeros(shap_values.shape[1]))

        words[-1] = words[-1] + tok
        vals[-1] = vals[-1] + val

        # For Roberta lexer, if the token ends with space, then this is the end
        # of an existing word
        if tok.endswith(" "):
            words[-1] = normalize(words[-1])
            words.append("")
            vals.append(np.zeros(shap_values.shape[1]))

        # Punctuation is counted as the end of a word
        if any(char in set(string.punctuation) for char in tok):
            words[-1] = normalize(words[-1])
            words.append("")
            vals.append(np.zeros(shap_values.shape[1]))
    # print("final words:", words)
    # print(np.sum(shap_values), np.sum(np.stack(vals)))
    words[-1] = normalize(words[-1])
    return vals, words


def topic_shap(tokens, word2idx, topics, shap_values):
    topic_values = np.zeros((topics.shape[0], shap_values[0].shape[0]))
    # topics_z = np.concatenate([topics, np.zeros((topics.shape[0], 1))], axis=1)
    for tok, val in zip(tokens, shap_values):
        topic_values += np.array([val * topics[i, word2idx.get(tok, -1)]
                                 for i in range(topics.shape[0])])
    # no_topic = 0.0
    # for tok, val in zip(tokens, shap_values):
    #     no_topic += val * (word2idx.get(tok, -1) == -1)
    return topic_values #np.concatenate([topic_values, np.array([no_topic])])


def sort_shap(shap_values, feature_names):
    sort_idx = np.argsort(shap_values)
    return shap_values[sort_idx], feature_names[sort_idx]

def get_topics(config, data):
    if config["topics"] == "liwc":
        return get_liwc_topics()
    elif config["topics"] == "ctm" or config["topics"] == "neurallda":
        base_path = Path(__file__).parent
        data_name = f"{config['dataset']}_raw.txt"
        
        if not os.path.isfile(base_path / "../data" / data_name):
            with open(base_path / ("../data/" + data_name), "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter="\t", escapechar="\\")
                for line in data:
                    writer.writerow([line])

        if not os.path.exists(base_path / "../data" / (config["dataset"] + "_preprocessed")):
            preprocessor = Preprocessing(
                num_processes=None,
                vocabulary=None,
                max_features=None,
                remove_punctuation=True,
                punctuation=string.punctuation,
                lemmatize=True,
                stopword_list="english",
                min_chars=1,
                min_words_docs=0,
                max_df=0.95,
            )
            dataset = preprocessor.preprocess_dataset(documents_path=(base_path / ("../data/" + data_name)))
            dataset.save("data/" + config["dataset"] + "_preprocessed")

        dataset = octDataset()
        dataset.load_custom_dataset_from_folder("data/" + config["dataset"] + "_preprocessed")
        # npmi = Coherence(texts=dataset.get_corpus())

        if config["topics"] == "ctm":
            model = CTM(num_topics=30, num_epochs=35, inference_type='zeroshot', bert_model="bert-base-nli-mean-tokens")
        elif config["topics"] == "neurallda":
            model = NeuralLDA(num_topics=30, lr=0.0001)

        # search_space = {"num_layers": Categorical({1, 2, 3}), 
        #         "num_neurons": Categorical({100, 200, 300}),
        #         "activation": Categorical({'sigmoid', 'relu', 'softplus'}), 
        #         "dropout": Real(0.0, 0.95)
        # }
        model_output = model.train_model(dataset)
        print(model_output)
        word2idx = {word: i for i, word in enumerate(dataset.get_vocabulary())}
        topics = model_output["topic-word-matrix"]
        print(model_output["topics"][0][:5])
        print([dataset.get_vocabulary()[i] for i in np.argsort(-topics[0, :])[:5]])
        print([model.model.train_data.idx2token[i] for i in np.argsort(-topics[0, :])[:5]])
        topics = topics / np.sum(topics, axis=0, keepdims=True)
        return topics, word2idx
    
    elif config["topics"] == "lda":
        base_path = Path(__file__).parent
        topics_matrix_df = pd.read_csv(base_path / ("../data/processed_LDA_files/" + config["dataset"] + ".csv"), dtype={"words": str}, keep_default_na=False)
        topics_matrix_df["words"] = topics_matrix_df["words"].astype(str)
        word2idx = dict(zip(topics_matrix_df["words"], range(len(topics_matrix_df["words"]))))
        topics_matrix_df.drop(columns=["words"], inplace=True)
        topics_matrix_df = topics_matrix_df.T
        topics = topics_matrix_df.to_numpy().astype(np.float64)
        topics = topics / np.sum(topics, axis=0, keepdims=True)
        assert (topics.shape[1] == len(word2idx))
        return topics, word2idx

    else:
        raise NotImplementedError

def get_liwc_topics():
    base_path = Path(__file__).parent
    feature_groups = LIWCWordData(
        base_path / "../data/LIWC2015_processed_full.csv",
        split=None)

    print(feature_groups.groups)
    topic_names = feature_groups.groups
    print(topic_names)
    topics = {name: set() for name in topic_names}
    print(feature_groups[0])
    for word, groups in feature_groups:
        for group in groups:
            topics[group].add(word)
    all_tokens = set().union(*topics.values())
    word2idx = {word: i for i, word in enumerate(all_tokens)}
    topics = np.array([[1.0 if tok in topics[topic_names[i]]
                        else 0.0 for tok in all_tokens] for i in range(len(topic_names))])
    topics = topics / np.sum(topics, axis=0, keepdims=True)
    return topics, word2idx


def get_topic_shap(model, alt_model, data, topics, word2idx, shap_values=None):
    explainer = shap.Explainer(model, padding="max_length", truncation=True, max_length=512)
    explainer_other = shap.Explainer(alt_model, padding="max_length", truncation=True, max_length=512)
    if shap_values is None:
        # print("First model output:", model(data[0]))
        shap_values = explainer(data).values

    # Finding all the stop words to turn into topics
    stop_words = set()
    punctuation = set()
    for i in tqdm(range(len(data))):
        tok_sample = explainer.masker.data_transform(data[i])[0]
        values, words = word_shap(tok_sample, np.zeros((len(tok_sample), 1)))

        tok_sample_other = explainer_other.masker.data_transform(data[i])[0]
        _, words_other = word_shap(tok_sample_other, np.zeros((len(tok_sample_other), 1)))
        for word in (words + words_other):
            if word != "" and word not in word2idx:
                if any(char in set(string.punctuation) for char in word):
                    punctuation.add(word)
                else:
                    stop_words.add(word)
    stop_words = sorted(list(stop_words))
    punctuation = sorted(list(punctuation))
    print(stop_words)
    print(punctuation)
    print(len(stop_words))

    # Update the word2idx and topics
    assert (np.allclose(np.sum(topics, axis=0), np.ones(topics.shape[1])))
    if len(stop_words) + len(punctuation) > 0:
        last_topic_idx = len(topics)
        topics = np.concatenate([topics, np.zeros((topics.shape[0], len(stop_words) + 1))], axis=1)
        topics = np.concatenate([topics, np.zeros((len(stop_words) + 1, topics.shape[1]))], axis=0)
        last_word_idx = len(word2idx)
        new_words = {stop_word: (last_word_idx + i) for i, stop_word in enumerate(stop_words)}
        word2idx.update(new_words)
        punctuation_idx = len(word2idx)
        new_punctuation = {symbol: punctuation_idx for i, symbol in enumerate(punctuation)}
        word2idx.update(new_punctuation)
        for i, word in enumerate(stop_words):
            topics[last_topic_idx + i, word2idx[word]] = 1
        topics[len(topics) - 1, -1] = 1

    idx2stop_word_topic = {last_topic_idx + i: word for i, word in enumerate(stop_words)}

    print("Topics shape:", topics.shape)

    stop_words = set()
    word_shap_v = np.zeros((topics.shape[1], shap_values[0].shape[1]))
    print("word shap shape:", word_shap_v.shape)
    word_shap_cnt = np.ones(topics.shape[1])
    for i in tqdm(range(shap_values.shape[0])):
        tok_sample = explainer.masker.data_transform(data[i])[0]
        assert (len(tok_sample) == len(shap_values[i]))
        values, words = word_shap(tok_sample, shap_values[i])
        for word, val in zip(words, values):
            word_shap_v[word2idx.get(word, -1), :] += np.abs(val)
            word_shap_cnt[word2idx.get(word, -1)] += 1
    word_shap_v = word_shap_v / word_shap_cnt[:, np.newaxis]

    topic_values = topics.astype(np.float64) @ word_shap_v.astype(np.float64)
    print(np.sum(topics, axis=0))
    assert (np.allclose(np.sum(topics, axis=0), np.ones(topics.shape[1])))
    print(np.sum(word_shap_v, axis=0), np.sum(topic_values, axis=0))
    assert (np.allclose(np.sum(word_shap_v, axis=0), np.sum(topic_values, axis=0)))
    # topic_values = np.zeros((topics.shape[0], shap_values[0].shape[1]))
    # for word, i in tqdm(word2idx.items()):
    #     idx = word2idx.get(word, -1)
    #     topic_values += np.array([word_shap_v[idx] * topics[j, idx]
    #                              for j in range(topics.shape[0])])

    # for i in tqdm(range(shap_values.shape[0])):
    #     tok_sample = explainer.masker.data_transform(data[i])[0]
    #     values, words = word_shap(tok_sample, shap_values[i])
    #     for word in words:
    #         if word != "" and word not in word2idx:
    #             stop_words.add(word)
    #     topic_values = topic_shap(words, word2idx, topics, values)
    #     topic_vals.append(topic_values)
    #     word_vals.append(values)
    # topic_vals = np.stack(topic_vals, axis=0)
    assert(len(stop_words) == 0)
    return shap_values, topic_values, word_shap_v, topics, idx2stop_word_topic


def load_models(config):
    dataset_name = config["dataset"]
    if dataset_name == "goemotions":
        num_labels = 6
        problem_type = "multi_label_classification"
    elif dataset_name == "blog":
        num_labels = 5
        problem_type = "multi_label_classification"
    elif dataset_name == "polite":
        num_labels = 1
        problem_type = "regression"
    elif dataset_name == "yelp":
        num_labels = 5
        problem_type = "single_label_classification"
    else:
        num_labels = 2
        problem_type = "single_label_classification"


    tokenizer1 = AutoTokenizer.from_pretrained(
        "distilroberta-base")
    model1 = AutoModelForSequenceClassification.from_pretrained(
        "distilroberta-base", num_labels=num_labels, problem_type=problem_type).cuda()
    pred1 = transformers.TextClassificationPipeline(
        model=model1, tokenizer=tokenizer1, device=0, top_k=None, padding="max_length", truncation=True, max_length=512)

    tokenizer2 = AutoTokenizer.from_pretrained(
        "gpt2")
    model2 = AutoModelForSequenceClassification.from_pretrained(
        "gpt2", num_labels=num_labels, problem_type=problem_type).cuda()
    tokenizer2.pad_token = tokenizer2.eos_token
    model2.config.pad_token_id = tokenizer2.pad_token_id
    pred2 = transformers.TextClassificationPipeline(
        model=model2, tokenizer=tokenizer2, device=0, top_k=None, padding="max_length", truncation=True, max_length=512)

    return pred1, pred2

def load_data(config):
    dataset_name = config["dataset"]
    base_path = Path(__file__).parent
    if dataset_name == "sst2":
        sst2_train = load_dataset("glue", "sst2", split="train")
        sst2_val = load_dataset("glue", "sst2", split="validation")
        sst2_test = load_dataset("glue", "sst2", split="test")
        return sst2_train, sst2_val, sst2_test
    elif dataset_name == "cola":
        cola_train = load_dataset("glue", "cola", split="train")
        cola_val = load_dataset("glue", "cola", split="validation")
        return cola_train, cola_val
    elif dataset_name == "tweet":
        cola_train = load_dataset("ought/raft", "tweet_eval_hate", split="train")
        cola_train = cola_train.rename_column("Tweet", "sentence")
        cola_train = cola_train.rename_column("Label", "labels")
        cola_val = load_dataset("ought/raft", "tweet_eval_hate", split="test")
        cola_val = cola_val.rename_column("Tweet", "sentence")
        cola_val = cola_val.rename_column("Label", "labels")
        return cola_train, cola_val
    elif dataset_name == "goemotions":
        goemotions_train = load_dataset("go_emotions", "simplified", split="train")
        goemotions_train = goemotions_train.rename_column("text", "sentence")
        goemotions_val = load_dataset("go_emotions", "simplified", split="validation")
        goemotions_val = goemotions_val.rename_column("text", "sentence")
        goemotions_test = load_dataset("go_emotions", "simplified", split="test")
        goemotions_test = goemotions_test.rename_column("text", "sentence")
        def to_tensor(x):
            y = torch.zeros((6)).float()
            anger = 2
            surprise = 26
            disgust = 11
            enjoyment = 17
            fear = 14
            sadness = 25
            for idx in x["labels"]:

                if idx == anger:
                    y[0] = 1.0
                elif idx == surprise:
                    y[1] = 1.0
                elif idx == disgust:
                    y[2] = 1.0
                elif idx == enjoyment:
                    y[3] = 1.0
                elif idx == fear:
                    y[4] = 1.0
                elif idx == sadness:
                    y[5] = 1.0

            x["labels"] = y
            return x
        goemotions_train = goemotions_train.map(to_tensor)
        goemotions_val = goemotions_val.map(to_tensor)
        goemotions_test = goemotions_test.map(to_tensor)
        new_features = goemotions_train.features.copy()
        new_features["labels"] = Sequence(Value("float32"))
        goemotions_train = goemotions_train.cast(new_features)
        goemotions_val = goemotions_val.cast(new_features)
        goemotions_test = goemotions_test.cast(new_features)
        
        return goemotions_train, goemotions_val, goemotions_test
    elif dataset_name == "blog":
        blog_train = load_dataset("blog_authorship_corpus", split="train")
        blog_train = blog_train.rename_column("text", "sentence")
        blog_val = load_dataset("blog_authorship_corpus", split="validation")
        blog_val = blog_val.rename_column("text", "sentence")

        def get_raw_labels(x):
            age_group = "10s"
            if x["age"] >= 33:
                age_group = "30s"
            elif x["age"] >= 23:
                age_group = "20s"
            x["raw_labels"] = [age_group, x["gender"]]
            return x

        blog_train = blog_train.map(get_raw_labels)
        blog_val = blog_val.map(get_raw_labels)
        mlb = MultiLabelBinarizer()
        train_labels = list(mlb.fit_transform([blog["raw_labels"] for blog in blog_train]).astype(float))
        val_labels = list(mlb.transform([blog["raw_labels"] for blog in blog_val]).astype(float))

        blog_train = blog_train.add_column("labels", train_labels)
        blog_val = blog_val.add_column("labels", val_labels)
        indices = list(range(len(blog_val)))
        random.seed(316)
        random.shuffle(indices)
        test_size = int(len(blog_val) // 2)
        blog_test = blog_val.select(indices[:test_size])
        blog_val = blog_val.select(indices[test_size:])

        # print(blog_train[0])
        return blog_train, blog_val, blog_test
    elif dataset_name == "emobank":
        emobank = pd.read_csv(base_path / '../data/emobank.csv')
        emobank['labels'] = emobank[['V', 'A', 'D']].sum(axis=1) / 3
        emobank = emobank.rename({'text': 'sentence'}, axis=1)
        emobank_train = Dataset.from_pandas(emobank[emobank["split"] == "train"][["sentence", "labels"]])
        emobank_dev = Dataset.from_pandas(emobank[emobank["split"] == "dev"][["sentence", "labels"]])
        return emobank_train, emobank_dev
    elif dataset_name == "polite":
        polite = pd.read_csv(base_path / '../data/wikipedia.annotated.csv')
        polite = polite.rename({"Request": "sentence", "Normalized Score": "labels"}, axis=1)
        polite_train, polite_test = train_test_split(polite, test_size=0.2)
        polite_val, polite_test = train_test_split(polite_test, test_size=0.5)
        polite_train = Dataset.from_pandas(polite_train[["sentence", "labels"]])
        polite_val = Dataset.from_pandas(polite_val[["sentence", "labels"]])
        polite_test = Dataset.from_pandas(polite_test[["sentence", "labels"]])
        return polite_train, polite_val, polite_test
    elif dataset_name == "yelp":
        yelp_train = load_dataset("yelp_review_full", split="train")
        yelp_train = yelp_train.rename_column("text", "sentence").rename_column("label", "labels")
        yelp_test = load_dataset("yelp_review_full", split="test")
        yelp_test = yelp_test.rename_column("text", "sentence").rename_column("label", "labels")

        indices = list(range(len(yelp_test)))
        random.seed(316)
        random.shuffle(indices)
        test_size = int(len(yelp_test) // 2)
        yelp_val = yelp_test.select(indices[:test_size])
        yelp_test = yelp_test.select(indices[test_size:])
        return yelp_train, yelp_val, yelp_test
    else:
        raise NotImplementedError

def write_to_db(dataset_name, table_name, splits):
    train= []
    val = []
    test = []
    if("train" in splits):
        data_train = load_dataset(dataset_name, split="train")  
        train = [data_train[i]['text'] for i in tqdm(range(len(data_train)))]
    if("val" in splits):
        data_val = load_dataset(dataset_name, split="validation")
        val = [data_val[i]['text'] for i in tqdm(range(len(data_val)))]
    if("test" in splits):
        data_test = load_dataset(dataset_name, split="test")
        test = [data_test[i]['text'] for i in tqdm(range(len(data_test)))]

    msgs = np.concatenate([train, test, val])

    df = pd.DataFrame(columns=["message_id", "message"])
    message_ids = range(len(msgs))
    df["message"] = msgs
    df["message_id"] = message_ids
    
    db = sqlalchemy.engine.url.URL(drivername='mysql', host='127.0.0.1', database='shreyah', query={'read_default_file': '~/.my.cnf', 'charset':'utf8mb4'})
    engine = sqlalchemy.create_engine(db)
    df.to_sql(table_name, con=engine, index=False, if_exists='replace', chunksize=50000)

def process_mallet_topics(filepath, numtopics, dataset):
    topics = {}
    words = set(())
    for i in range(numtopics):
        topics[i] = {}

    with open(filepath) as f:
        lines = csv.reader(f, delimiter=',', quotechar='\"')
        for parts in lines:
            if(len(parts) == 6):
                continue
            topic_id = int(parts[0])
            parts = parts[1:]
            for i in range(len(parts)):      
                if(i%2 == 0): #word
                    try:
                        word = parts[i]
                        score = parts[i+1]
                        words.add(word)
                    except:
                        print("error")
                (topics[topic_id])[word] = score
    
    words = list(words)
    word_scores = []        
    for w in words:
        scores = []
        for t in topics:
            try:
                score = topics[t][w]
            except(KeyError):
                score = 0
            scores.append(score)
        word_scores.append(scores)
    
    lda_scores = pd.DataFrame(data=word_scores, columns=range(numtopics))
    lda_scores["words"] = words
    outfile = "data/processed_LDA_files/" + dataset + ".csv"
    lda_scores.to_csv(outfile, index=False)

    #ensure all columns add to 1
    for i in range(numtopics):
        print(np.sum(np.array(lda_scores[i].astype(float))))

def process_lda():
    process_mallet_topics("LDA/yelp_lda_50_30/lda.wordGivenTopic.csv", 30, "yelp_50")
    process_mallet_topics("LDA/blog_lda_50_30/lda.wordGivenTopic.csv", 30, "blog_50")
    process_mallet_topics("LDA/emotions_lda_50_30/lda.wordGivenTopic.csv", 30, "goemotions_50")