import openai
import json
from PllUnderstandability.PllWordFileParser import PllWordFileParser
from PllUnderstandability.S3WordFileParser import S3WordFileParser
from PllUnderstandability.BrWordFileParser import BrWordFileParser
from readcalc import readcalc
from PllUnderstandability.TextParser import IndexCalculator
from PllUnderstandability.IgnorableChapterConstructor import IgnorableChapterConstructor
from PllUnderstandability.TextParser import ReadabilityMetrics
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('averaged_perceptron_tagger')

#openai.api_key = "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"
#model_engine = "text-davinci-003"

'''
prompt = "Until further notice, switch to German language. Verwende niemals Nebensätze, niemals Relativsätze und umschreibe alle Fachwörter."
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=4096-len(prompt),
    n=1,
    stop=None,
    temperature=0.5,
)

response = completion.choices[0].text
print(response)

pass
'''

with open("C:/Users/User/PycharmProjects/OlCmsTools/source_text_paths.txt", "r", encoding="utf-8") as f:
    source_file_list = json.load(f)

def analyze_paragraphs(guideline, type, paragraph_dict, translate_type=None):
    result = ""
    index_calculator = IndexCalculator()
    translated_text = ""
    for key, text_list in paragraph_dict.items():
        for text in text_list:
            text = text.strip().replace("\r", " ").replace("  ", " ").strip()
            metrics = index_calculator.Handle(text)
            if metrics is not None:
                result += add_to_csv(guideline, type, key, metrics, text)
            if translate_type is not None:
                translated_text = translate_text_to_easy_langage(text).replace("\n", " ").replace("\r", " ").replace("  ", " ").strip()
                #print(translated_text)
                metrics = index_calculator.Handle(translated_text)
                #if metrics is not None:
                result += add_to_csv(guideline, translate_type, key, metrics, translated_text)
    return result

out_csv = "Leitlinie|Typ|Kapitel|Text|Chars|Words|Types|Sentences|Syllables|Polysyllable Words|Difficult Words|Words > 4|Words > 6|Words > 10|Words > 13|ASL|ASW|Flesh-Kincaid|Coleman-Liau|Gunning-Fog|Smog|ARI|LIX|Dale-Chall\n"

def add_to_csv(guideline, type, chapter, metric, text):
    if metric is not None and metric.number_words > 0:
        result = "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n" % (
            guideline.replace("\r\n"," ").replace("\n", " ").replace("\r", " ").replace("  ", " ").strip(),
            type.replace("\r\n"," ").replace("\n", " ").replace("\r", " ").replace("  ", " ").strip(),
            chapter.replace("\r\n"," ").replace("\n", " ").replace("\r", " ").replace("  ", " ").strip(),
            text.replace("\r\n"," ").replace("\n", " ").replace("\r", " ").replace("  ", " ").strip(),
            metric.number_chars,
            metric.number_words,
            metric.number_types,
            metric.number_sentences,
            metric.number_syllables,
            metric.number_polysyllable_words,
            metric.difficult_words,
            metric.number_words_longer_4,
            metric.number_words_longer_6,
            metric.number_words_longer_10,
            metric.number_words_longer_13,
            metric.ASL,
            metric.ASW,
            metric.flesch_reading_ease,
            metric.coleman_liau_index,
            metric.gunning_fog_index,
            metric.smog_index,
            metric.ari_index,
            metric.lix_index,
            metric.dale_chall_score
        )
    else:
        result = "%s|%s|%s|%s||||||||||||||||||||\n" % (
            guideline,
            type,
            chapter,
            text
        )
    return result


def translate_text_to_easy_langage(text):
    prompt = "Folgenden Text in leichte Sprache umwandeln: %s" % text
    try:
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=4096-len(prompt),
            n=1,
            stop=None,
            temperature=0.5,
        )

        response = completion.choices[0].text
    except:
        response = "ERROR"
    return response


for guideline in source_file_list['files']:

    if len(guideline['br']) > 0:
        print(guideline['br'])
        parser = BrWordFileParser()
        paragraph_dict = parser.Parse(guideline['br'])
        out_csv += analyze_paragraphs(guideline['title'], "BR", paragraph_dict)

    if len(guideline['pll']) > 0:
        print(guideline['pll'])
        parser = PllWordFileParser()
        paragraph_dict = parser.Parse(guideline['pll'])
        out_csv += analyze_paragraphs(guideline['title'], "PLL", paragraph_dict)

    if len(guideline['s3']) > 0:
        print(guideline['s3'])
        parser = S3WordFileParser()
        paragraph_dict = parser.Parse(guideline['s3'])
        out_csv += analyze_paragraphs(guideline['title'], "S3L", paragraph_dict)

    with open("count_results.csv", "w", encoding="utf-8") as f:
        f.write(out_csv)



