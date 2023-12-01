import os
import sys

import fasttext
import openai
import torch
from pyrate_limiter import Duration, Limiter, RequestRate
# import intel_extension_for_pytorch as ipex
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel,
                          MarianTokenizer)

rate_limits = (RequestRate(50, Duration.MINUTE),)
limiter = Limiter(*rate_limits)

# device = "xpu" if torch.xpu.is_available() else "cpu"
device = "cpu"
bart_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
bart_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
translation_model_name = "Helsinki-NLP/opus-mt-{src_lang}-en"
translation_model = MarianMTModel.from_pretrained(
    translation_model_name.format(src_lang="fr")
)
translation_tokenizer = MarianTokenizer.from_pretrained(
    translation_model_name.format(src_lang="fr")
)
translation_model = translation_model.to(device)
bart_model = bart_model.to(device)
language_model = fasttext.load_model("lid.176.ftz")


def _truncate_tokens(text, max_chars=10000):
    """truncate text to 4096 tokens - 4096 * 4 == 16384 (10000 with buffer) chars"""
    return text[:max_chars]


def _clean_summary(summary):
    last_fullstop = summary.rfind(".")
    if last_fullstop != -1:
        summary = summary[:last_fullstop]
    return summary.capitalize()


def detect_language(text):
    predictions = language_model.predict(text, k=1)
    lang_code = predictions[0][0].replace("__label__", "")
    print(f"language: {lang_code}")
    return lang_code


def translate_to_english(text, src_lang):
    global translation_model, translation_tokenizer
    translation_model = MarianMTModel.from_pretrained(
        translation_model_name.format(src_lang=src_lang)
    )
    translation_tokenizer = MarianTokenizer.from_pretrained(
        translation_model_name.format(src_lang=src_lang)
    )
    translation_model = translation_model.to(device)
    translated = translation_model.generate(
        **translation_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
    )
    return translation_tokenizer.decode(translated[0], skip_special_tokens=True)


def generate_bart_summary(
    text, max_length=150, min_input_length=50, keyword_threshold=5
):
    lang_code = detect_language(text)
    if lang_code != "en":
        text = translate_to_english(text, lang_code)
    inputs = bart_tokenizer.encode(text, return_tensors="pt", truncation=True)
    inputs = inputs.to(device)
    summary_ids = bart_model.generate(
        inputs,
        max_length=max_length,
        min_length=50,
        num_beams=4,
    )
    return _clean_summary(
        bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    )


def generate_openai_summary(text, max_length=150):
    limiter.try_acquire("summarize_text")
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except KeyError:
        print("OPENAI_API_KEY not set, exiting...")
        sys.exit(0)
    text = _truncate_tokens(text)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Please provide a summary of the following text without starting the summary with the phrase 'The text describes':\n{text}\n\nSummary:",
        max_tokens=max_length,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return _clean_summary(response.choices[0].text.strip())


def generate_summary(text, max_length=150, use_openai=False):
    if use_openai and os.getenv("OPENAI_API_KEY"):
        return generate_openai_summary(text, max_length)
    else:
        return generate_bart_summary(text, max_length)
