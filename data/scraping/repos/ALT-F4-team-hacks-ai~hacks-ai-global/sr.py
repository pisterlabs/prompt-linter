'''import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import torch
import stable_whisper
import math
import whisper
import joblib
import os
import openai
import time

from scipy.signal import argrelextrema
from tqdm import tqdm
from transformers import AutoTokenizer
from glob import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = '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'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Функция для разделения текста на предложени else я
def parse_text(full_text, sentence_transformer):
    text = full_text
    sentences = text.split('. ')
    # Эмбеддинг предложений
    embeddings = sentence_transformer.encode(sentences)
    # Get the length of each sentence
    sentece_length = [len(each) for each in sentences]
    # Determine longest outlier
    long = np.mean(sentece_length) + np.std(sentece_length) * 1.5
    # Determine shortest outlier
    short = np.mean(sentece_length) - np.std(sentece_length) * 1.5
    # Shorten long sentences
    text = ''
    for each in sentences:
        if len(each) > long:
            # let's replace all the commas with dots
            comma_splitted = each.replace(',', '.')
        else:
            text += f'{each}. '
    sentences = text.split('. ')
    # Now let's concatenate short ones
    text = ''
    for each in sentences:
        if len(each) < short:
            text += f'{each} '
        else:
            text += f'{each}. '
    # Split text into sentences
    sentences = text.split('. ')
    # Embed sentences
    embeddings = sentence_transformer.encode(sentences)
    print(embeddings.shape)

    # Normalize the embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Create similarities matrix
    similarities = cosine_similarity(embeddings)

    def rev_sigmoid(x: float) -> float:
        return (1 / (1 + math.exp(0.5 * x)))

    def activate_similarities(similarities: np.array, p_size=10) -> np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10, 10, p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid)
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1, 1)
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities

    # Let's apply our function. For long sentences i reccomend to use 10 or more sentences
    activated_similarities = activate_similarities(similarities, p_size=10)

    ### 6. Find relative minima of our vector. For all local minimas and save them to variable with argrelextrema function
    minmimas = argrelextrema(activated_similarities, np.less,
                             order=2)  # order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
    # Create empty string
    split_points = [each for each in minmimas[0]]
    text = ''
    count_symbols_in_paragraph = 0  # Сколько символов в абзаце
    for num, each in enumerate(sentences):
        if num in split_points and count_symbols_in_paragraph >= 500:
            text += f'\n\n {each}. '
            count_symbols_in_paragraph = 0
        else:
            sentence_to_add = f'{each}. '
            text += sentence_to_add
            count_symbols_in_paragraph += len(sentence_to_add)

    if count_symbols_in_paragraph != 0:
        beg_of_last_paragraph = text.rfind('\n\n')

        if len(text) - beg_of_last_paragraph - 2 <= 500 and beg_of_last_paragraph != -1:
            text = text[:beg_of_last_paragraph] + text[beg_of_last_paragraph + 2:]

    return text.split('\n\n')


# фунцкия для извлечения терминов
def get_termins(texts):
    answers = []
    for text in texts:
        start = time.time()
        print(text)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": f'Найди в тексте термин с его определением, которое дано в самом тексте и выпиши мне его слово в слово из текста, не называя другие определения: {text}'}
            ]
        )
        ans = completion.choices[0].message.content
        answers.append(ans)
        elapsed_time = start - time.time()
        time.sleep(max(21 - elapsed_time, 0))
    return answers


# функция для суммаризаци абзацов
def summarize(paragraphs, model, tokenizer):
    results = []
    for paragraph in paragraphs:
        with torch.no_grad():
            summary_ids = model.generate(
                torch.tensor(tokenizer(paragraph)['input_ids']).reshape(1, -1).to(device),
                num_beams=int(2),
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                min_length=2,
                max_length=100,
                early_stopping=True
            )
            result = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            results.append(result)

    return results


class TextProcessor():
    def __init__(self, filename):
        self.filename = filename

    def process_text(self):
        print("starting")
        model = whisper.load_model("base")
        # 1. транскрибируем текст

        print("transcribating")
        result = model.transcribe(self.filename)
        result_text = result["text"]
        print("transcribating finished")

        # 2. парсим на абзацы
        print("loading_sentence_transformer")
        sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
        print("parsing started")
        parsed_text = parse_text(result_text, sentence_transformer)
        print("parsing done")

        # 3 выделяем термины + определения
        print("start termins")
        termins = get_termins(parsed_text)
        print("termins done")

        # 5 сумаризируем абзац

        SUMM_MODEL_NAME = "UrukHan/t5-russian-summarization"
        tokenizer = AutoTokenizer.from_pretrained(SUMM_MODEL_NAME)

        summ_model_path = "models/sum_model.pkl"
        summ_model = joblib.load(summ_model_path)
        parsed_summed_text = summarize(parsed_text, summ_model, tokenizer)
        print("summurization done")

        # 6 обьединяем по названиям
        # grouped_text = group_text(parsed_summed_text)

        parsed_summed_text = json.dumps(parsed_summed_text, ensure_ascii=False).encode('utf8')
        termins = json.dumps(termins, ensure_ascii=False).encode('utf8')

        return parsed_summed_text, termins


audi = "data/audio6.mp3"
text_proccessor_object = TextProcessor(audi)
print(text_proccessor_object.process_text())'''
import json

class TextProcesso():
    def process_text(name):
        with open('static/terms.json', 'r', encoding='utf-8') as f:
            terms = json.load(f)
            terms = json.dumps(terms, sort_keys=False, indent=4, separators=(',', ': '))
        f.close()
        text = 'Появился, значит, в Зоне Чёрный сталкер. К лагерю ночью повадился ходить и там сует руку в палатку и говорит: «Водички попить!» А если не дашь хлебнуть из фляжки или наружу полезешь — пришибет! А раз мужик один решил пошутить: вылез тихо из палатки, надел кожаную перчатку и полез к соседям в палатку. Полез, значит, и попрошайничает жалостно: «Водички, водички попить…» А тут из палатки навстречу высовывается рука и за горло его — цап! И сиплый голосок отзывается тихонько: «А тебе моя водичка зачем нужна?!»'
        return text, terms
