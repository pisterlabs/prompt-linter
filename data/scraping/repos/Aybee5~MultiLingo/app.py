import os

import six
# from google.cloud import translate_v2 as translate
import streamlit as st
from dotenv import load_dotenv
from googletrans import Translator

import cohere

load_dotenv()

translator = Translator()
co = cohere.Client(os.environ['COHERE_API_KEY'])

langs = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "ny": "Chichewa",
    "zh-cn": "Chinese Simplified",
    "zh-tw": "Chinese Traditional",
    "co": "Corsican",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "tl": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Frisian",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "ht": "Haitian Creole",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "iw": "Hebrew",
    "hi": "Hindi",
    "hmn": "Hmong",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ig": "Igbo",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "ku": "Kurdish (Kurmanji)",
    "ky": "Kyrgyz",
    "lo": "Lao",
    "la": "Latin",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "lb": "Luxembourgish",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mi": "Maori",
    "mr": "Marathi",
    "mn": "Mongolian",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "no": "Norwegian",
    "ps": "Pashto",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ma": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sm": "Samoan",
    "gd": "Scots Gaelic",
    "sr": "Serbian",
    "st": "Sesotho",
    "sn": "Shona",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "es": "Spanish",
    "su": "Sudanese",
    "sw": "Swahili",
    "sv": "Swedish",
    "tg": "Tajik",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "cy": "Welsh",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zu": "Zulu"
}

options = list(langs.values())

st.title("MultiLingo:  Multilanguage Text Summarization for Everyone")

uploaded_file = st.file_uploader(
    "Upload the txt file to summarize", type="txt")

selectedLanguage = st.multiselect(
    "Select a language", options, default=None, max_selections=1)


def translate_text(target, text):

    # translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    result = translator.translate(text, dest=target)
    st.download_button('Download summarized text', result.text, file_name='summarized.txt', mime='text/plain')


def summarize():
    if uploaded_file is not None and selectedLanguage.__len__() > 0:
        selectedLanguageKey = list(langs.keys())[list(
            langs.values()).index(selectedLanguage[0])]
        summarizeText(selectedLanguageKey)


def summarizeText(selectedLanguageKey):
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)
    converted_data = bytes_data.decode("utf-8")
    response = co.summarize(
        text=converted_data,
        length='long',
        format='paragraph',
        model='summarize-xlarge',
        additional_command='',
        temperature=0.3,
    )
    translate_text(selectedLanguageKey, response.summary)


submit_btn = st.button("Summarize", on_click=summarize)
