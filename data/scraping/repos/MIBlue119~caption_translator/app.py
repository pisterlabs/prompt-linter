import os
import argparse
from pathlib import Path

import openai
from tqdm import tqdm

from caption_translator.fileloader import WebVttLoader, SrtLoader
from caption_translator.prompter import TranslatorPrompter
from caption_translator.config import AppConfig
from caption_translator.translator import Translator
from caption_translator.utils import LANGUAGES, TO_LANGUAGE_CODE

def main(args):
    # Set the OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")         
    file_path = args.file_path
    # Check the file is .vtt or .srt file
    file_extension = Path(file_path).suffix
    if file_extension not in [".vtt", ".srt"]:
        raise ValueError("File must be a .vtt /.srt file")
    # Initialize the loader class
    if file_extension == ".vtt":
        data_loader = WebVttLoader()
    elif file_extension == ".srt":
        data_loader = SrtLoader()
    # Initialize the config class
    config = AppConfig()
    config.set_text_engine(args.text_engine)
    config.LANGUAGE = args.language
    if args.test:
        config.IS_TEST = True
        config.TEST_NUM = args.test_num
    else:
        config.IS_TEST = False
    # Initialize the Translator class
    translator = Translator(config,TranslatorPrompter, data_loader)
    translator.make_translation(file_path)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", dest="file_path", type=str, help="Path to the file to be translated.")
    parser.add_argument(
        "--text_engine", 
        dest="text_engine",
        type=str,default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo","text-davinci-003"],
        help="Text engine to be used for summarization.")
    parser.add_argument(
        "--language",
        type=str,
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        default="japanese",
        help="language to translate to",
    )    
    parser.add_argument(
        "--test",
        dest="test",
        action="store_true",
        help="If test we only translate 10 contents you can easily check",
    )
    parser.add_argument(
        "--test_num",
        dest="test_num",
        type=int,
        default=24,
        help="test num for the test",
    )        
    args = parser.parse_args()
    main(args)