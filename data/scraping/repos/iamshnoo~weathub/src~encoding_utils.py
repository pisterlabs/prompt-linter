import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import fasttext.util
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import itertools
import random
import openai

# read api key from secret file
# uncomment the following line to use openai ada
# openai.api_key = open("secret.txt", "r").readline().strip()

logging.getLogger("transformers").setLevel(logging.ERROR)

fasttext_models = {}
fasttext_executor = ThreadPoolExecutor(max_workers=4)


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed()

## MODELS USED ##
# ON LINE 62, SET THE MODEL NAME TO ONE OF THE FOLLOWING
# BASED ON WHAT YOU SET IN ```run_weat.py````

# urduhack/roberta-urdu-small (Urdu monobert)
# DeepPavlov/rubert-base-cased (Russian monobert)
# dbmdz/bert-base-turkish-uncased (Turkish monobert)
# dbmdz/bert-base-italian-uncased (Italian monobert)
# monsoon-nlp/bert-base-thai (Thai monobert)
# asafaya/bert-base-arabic (Arabic monobert)
# nlpaueb/bert-base-greek-uncased-v1 (Greek monobert)
# bert-base-chinese (Chinese monobert)
# jcblaise/roberta-tagalog-base (Tagalog monobert)
# trituenhantaoio/bert-base-vietnamese-uncased (Vietnamese monobert)
# l3cube-pune/bengali-bert (Bengali monobert)
# l3cube-pune/hindi-bert-v2 (Hindi monobert)
# l3cube-pune/telugu-bert (Telugu monobert)
# l3cube-pune/marathi-bert-v2 (Marathi monobert)
# l3cube-pune/punjabi-bert (Punjabi monobert)
# ai4bharat/indic-bert (Indic multilingual bert)
# distilbert-base-multilingual-cased (Multilingual distilbert)
# xlm-roberta-base (Multilingual xlm-roberta)

## MODELS USED ##

@lru_cache(maxsize=128)
def load_hf_tokenizer_model(name="distilbert-base-multilingual-cased"):
    # Load tokenizer and model once
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_hidden_states=True)
    model.eval()

    # Make sure to use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, device, model


tokenizer, device, model = load_hf_tokenizer_model()


@lru_cache(maxsize=128)
def load_fasttext_model(lang="en"):
    directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ft_embeddings"
    )

    model_filename = f"cc.{lang}.300.bin"
    model_path = os.path.join(directory, model_filename)

    if not os.path.exists(model_path):
        fasttext.util.download_model(lang, if_exists="ignore")
        os.rename(model_filename, model_path)

    return fasttext.load_model(model_path)


def load_fasttext_model_parallel(lang="en"):
    return fasttext_executor.submit(load_fasttext_model, lang).result()


# Static embeddings : FastText
def encode_words_a(texts, lang="en", phrase_strategy="average"):
    if lang not in fasttext_models:
        fasttext_models[lang] = load_fasttext_model_parallel(lang)
    fasttext_emb = fasttext_models[lang]

    phrase_embeddings = []

    for text in texts:
        words = text.split(" ")
        word_embeddings = np.array(
            [fasttext_emb.get_word_vector(word) for word in words]
        )

        if phrase_strategy == "average":
            phrase_embedding = np.mean(word_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown phrase strategy: {phrase_strategy}")

        phrase_embeddings.append(phrase_embedding)

    return np.array(phrase_embeddings)


# Static embeddings : BERT
def encode_words_b(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        hidden_states = outputs.hidden_states
        selected_hidden_state = hidden_states[0]
        embeddings.append(selected_hidden_state[0, 1].cpu().numpy())
    return embeddings


# Method 1 : Uses the last hidden state for contextualized word embeddings
def encode_words_1(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        embeddings.append(outputs.last_hidden_state[0, 1].cpu().numpy())
    return embeddings


# Method 2 : Uses the CLS token for contextualized word embeddings
def encode_words_2(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        embeddings.append(outputs.last_hidden_state[0, 0].cpu().numpy())
    return embeddings


# Method 3 : Hidden state from (last-1)-th layer for contextualized word embeddings
def encode_words_3(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        hidden_states = outputs.hidden_states
        selected_hidden_state = hidden_states[-2]
        embeddings.append(selected_hidden_state[0, 1].cpu().numpy())
    return embeddings


# Method 4 : Average of all hidden states for contextualized word embeddings
def encode_words_4(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        hidden_states = outputs.hidden_states

        stacked_hidden_states = torch.stack(hidden_states)

        avg_embedding = torch.mean(stacked_hidden_states[:, 0, 1], dim=0).cpu().numpy()

        embeddings.append(avg_embedding)
    return embeddings


# Method 5: OpenAI ADA
def encode_openai_ada(texts, subword_strategy="average", phrase_strategy="average"):
    def handle_phrase(embeddings, strategy):
        embeddings_tensors = [
            torch.from_numpy(np.array(embedding)) for embedding in embeddings
        ]
        embeddings_stacked = torch.stack(embeddings_tensors)

        if strategy == "average":
            return torch.mean(embeddings_stacked, dim=0).numpy()
        else:
            raise ValueError(f"Unknown phrase strategy: {strategy}")

    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)["data"][0][
            "embedding"
        ]

    all_embeddings = []

    for text in texts:
        words = text.split(" ")
        word_embeddings = []

        for word in words:
            embedding = get_embedding(word)
            word_embeddings.append(embedding)

        if len(words) > 1:
            phrase_embedding = handle_phrase(word_embeddings, phrase_strategy)
            all_embeddings.append(phrase_embedding)
        else:
            all_embeddings.append(word_embeddings[0])

    return all_embeddings


def handle_subwords(tokens, strategy, encoding_method, tokenizer, model):
    embeddings = encoding_method(tokens, tokenizer, model)
    embeddings_tensors = [torch.from_numpy(embedding) for embedding in embeddings]
    embeddings_stacked = torch.stack(embeddings_tensors)

    if strategy == "average":
        return [torch.mean(embeddings_stacked, dim=0).numpy()]
    elif strategy == "max":
        return [torch.max(embeddings_stacked, dim=0).values.numpy()]
    elif strategy == "first":
        return [embeddings[0]]
    else:
        raise ValueError(f"Unknown subword strategy: {strategy}")


def handle_phrase(word_embeddings, strategy):
    embeddings_tensors = [torch.from_numpy(embedding) for embedding in word_embeddings]
    embeddings_stacked = torch.stack(embeddings_tensors)

    if strategy == "average":
        return torch.mean(embeddings_stacked, dim=0).numpy()
    else:
        raise ValueError(f"Unknown phrase strategy: {strategy}")


def encode(
    texts,
    subword_strategy="average",
    phrase_strategy="average",
    encoding_method="1",
    tokenizer=tokenizer,
    model=model,
):
    all_embeddings = []

    encoding_methods = {
        "0": encode_words_b,
        "1": encode_words_1,
        "2": encode_words_2,
        "3": encode_words_3,
        "4": encode_words_4,
    }

    if phrase_strategy is None:
        phrase_strategy = "average"

    for text in texts:
        words = text.split(" ")
        word_embeddings = []

        for word in words:
            tokens = tokenizer(word).tokens()[1:-1]

            if len(tokens) == 1:
                embeddings = encoding_methods[encoding_method](tokens, tokenizer, model)
            else:
                embeddings = handle_subwords(
                    tokens,
                    subword_strategy,
                    encoding_methods[encoding_method],
                    tokenizer,
                    model,
                )

            word_embeddings.append(embeddings[0])

        if len(words) > 1:
            phrase_embedding = handle_phrase(word_embeddings, phrase_strategy)
            all_embeddings.append(phrase_embedding)
        else:
            all_embeddings.append(word_embeddings[0])

    return all_embeddings


def encode_words(texts, args, tokenizer=tokenizer, model=model):
    if args["embedding_type"] == "static_fasttext":
        return encode_words_a(texts, args["lang"], args["phrase_strategy"])
    elif args["embedding_type"] == "static_bert":
        return encode(
            texts,
            args["subword_strategy"],
            args["phrase_strategy"],
            encoding_method="0",
            tokenizer=tokenizer,
            model=model,
        )
    elif args["embedding_type"] == "contextual":
        # If method 2 (CLS token) is selected, skip subword and phrase strategies
        if args["encoding_method"] == "2":
            return encode_words_2(texts, tokenizer, model)
        else:
            return encode(
                texts,
                args["subword_strategy"],
                args["phrase_strategy"],
                encoding_method=args["encoding_method"],
                tokenizer=tokenizer,
                model=model,
            )
    elif args["embedding_type"] == "openai_ada":
        return encode_openai_ada(
            texts, args["subword_strategy"], args["phrase_strategy"]
        )
    else:
        raise ValueError(f"Unknown embedding type: {args['embedding_type']}")


def encode_new(texts, args, model_name="distilbert-base-multilingual-cased"):
    if args["embedding_type"] == "static_fasttext":
        return encode_words_a(texts, args["lang"], args["phrase_strategy"])
    elif args["embedding_type"] == "static_bert":
        return encode(
            texts,
            args["subword_strategy"],
            args["phrase_strategy"],
            encoding_method="0",
            tokenizer=tokenizer,
            model=model,
        )
    elif args["embedding_type"] == "contextual":
        # If method 2 (CLS token) is selected, skip subword and phrase strategies
        if args["encoding_method"] == "2":
            return encode_words_2(texts, tokenizer, model)
        else:
            return encode(
                texts,
                args["subword_strategy"],
                args["phrase_strategy"],
                encoding_method=args["encoding_method"],
                tokenizer=tokenizer,
                model=model,
            )
    else:
        raise ValueError(f"Unknown embedding type: {args['embedding_type']}")


# DEMO usage
# NOTE: For [CLS] token method, subword or phrase strategies are not relevant
if __name__ == "__main__":

    texts = ["Hugging Face" ,"this is a test"]

    args = {
        "embedding_type": "contextual",
        "subword_strategy": "average",
        "phrase_strategy": "average",
        "encoding_method": "2",
        "lang": "en",
    }
    embeddings = encode_words(texts, args)
    print(len(embeddings), len(embeddings[0]))

    # args2 = {
    #     "embedding_type": "contextual",
    #     "subword_strategy": "first",
    #     "phrase_strategy": "average",
    #     "encoding_method": "2",
    #     "lang": "en",
    # }

    # embeddings2 = encode_words(texts, args2)
    # print(len(embeddings2), len(embeddings2[0]))
