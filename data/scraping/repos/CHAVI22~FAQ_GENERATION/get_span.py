
# get_answer.py
from coherence_span import get_coherent_sentences
from pipelines import span_pipeline
import re
from textblob import TextBlob
import pickle
import nltk.tokenize as nt
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')


'''
python -m textblob.download_corpora
'''


def get_entities(sent):
    namedEnt = []
    b = TextBlob(sent)
    for n in b.noun_phrases:
        if len(n) > 4 and n not in namedEnt:
            if n[0] not in ["'"]:
                namedEnt.append(n)
    return namedEnt


x = span_pipeline("span-extraction", span_model="valhalla/t5-small-qa-qg-hl",
                  span_tokenizer="valhalla/t5-small-qa-qg-hl")


'''
Given a passage of text this function will return a list of dictionaries.
Each dictionary has the following keys:
    'sentence' -> Holds an individual sentence from passage
    'spans' -> list of spans extarcted from the given passage
    'questions' -> an empty list initialized. Will be filled with questions from question generation module
    'answers' -> an empty list initialized. Will be filled with answers from answer generation module
'''


def check_input(passage):

    if (bool(re.match('^[ !@#$%^&*\-()_+:;=|\\>.,]*$', passage)) == True):
        print("invalid")
        return False
    else:
        print("valid")
        return True


def extract_spans(context):
    if check_input(context):
        # print(context)
        res = get_spans(context)
        return res
    else:
        return []


def get_spans(context):
    ext_spans = []
    gen_res = []
    c = sent_tokenize(context)
    # print(c)
    spanDict = dict()
    for i in list(x(context)):
        # print("inside the for loop for x of context")
        # print("\n")

        # print("\n")
        ext_spans = []
        sent = i[0]
        # print(sent)
        span = i[1]
        # print(span)
        if sent not in spanDict:
            spanDict[sent] = []
        for a in span:
            new_a = (a.strip("<pad>")).strip()
            if new_a not in spanDict[sent]:
                spanDict[sent].append(new_a)
                ext_spans.append(new_a)
        for j in x(sent):
            for span in j[1]:
                a = (span.strip("<pad>")).strip()
                if a not in spanDict[sent]:
                    spanDict[sent].append(a)
                    ext_spans.append(a)
        ent = get_entities(sent)
        temp_ext_spans = [i.lower() for i in ext_spans]
        for n in ent:
            if n not in temp_ext_spans:
                temp_ext_spans.append(n)
                ext_spans.append(n)
                spanDict[sent].append(n)
    # print("Getting coherent sentences")
    combs = get_coherent_sentences(c)
    # print(combs)
    for i in combs:
        for j in x(i):
            ext_spans = []
            sent = j[0]
            span = j[1]
            # print("current sentence : ",sent)
            # print("current answers : ",ans)
            if sent not in spanDict:
                spanDict[sent] = []
            for a in span:
                new_a = (a.strip("<pad>")).strip()
                if new_a not in spanDict[sent] and abs(len(new_a)-len(sent)) > 5:
                    spanDict[sent].append(new_a)
                    ext_spans.append(new_a)
            ent = get_entities(sent)
            # print("additional answers : ",ent)
            temp_ext_spans = [i.lower() for i in ext_spans]
            for n in ent:
                if n not in temp_ext_spans:
                    temp_ext_spans.append(n)
                    ext_spans.append(n)
                    spanDict[sent].append(n)

    for i in spanDict:
        ind_sent = {}
        ind_sent["sentence"] = i
        ind_sent["spans"] = spanDict[i]
        ind_sent["questions"] = []
        ind_sent["answers"] = []
        ind_sent["context"] = context
        gen_res.append(ind_sent)
        # print(ind_sent)
    return gen_res
