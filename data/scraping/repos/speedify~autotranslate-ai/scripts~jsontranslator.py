import os
import sys
import openai
import time
import json
import tiktoken
from translator import (
    translate_text,
    define_source_file,
    get_languages,
    store_translated_text_to_file,
)
from map_translations import map_translations

source = "json"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORG_ID"]
max_retries = 3
retry_delay = 60
source_dir = "en"
app_table = define_source_file()
translation_table = {}
translated_app_table = {}
dest_dir = sys.argv[2]
os.makedirs(dest_dir, exist_ok=True)
languages = get_languages()
enc = tiktoken.encoding_for_model("gpt-4")

# Loop through languages and translate JSON files
for language_code, country_code in languages.items():
    linebyline = False
    badError = False
    print("Translating to {}  with format {}".format(language_code, source))
    dest_file = "{}-{}.json".format(language_code, country_code)
    dest_filepath = os.path.join(dest_dir, dest_file)
    for retry in range(max_retries):
        if language_code == "en":
            continue
        if len(app_table) == 0:
            print("Skipping blank translation")
        else:
            try:
                for key, value in app_table.items():
                    if len(enc.encode(json.dumps(value, ensure_ascii=False))) > 5000:
                        linebyline = True

                    if not linebyline:
                        if isinstance(value, str):
                            translated_app_table[key] = translate_text(
                                app_table[key], language_code, source
                            )
                        else:
                            temp_text = ""
                            temp_text = json.dumps(app_table[key], ensure_ascii=False)
                            temp_text = translate_text(temp_text, language_code, source)
                            try:
                                translated_app_table[key] = json.loads(temp_text)
                            except json.JSONDecodeError:
                                print("Error decoding translated JSON for key:", key)
                                translated_app_table[key] = {}
                    else:
                        print(
                            "Retrying translation using line-by-line for key {} in {}".format(
                                key, language_code
                            )
                        )
                        linebyline = False
                        if isinstance(value, str):
                            translated_app_table[key] = translate_text(
                                app_table[key], language_code, source
                            )
                        else:
                            temp_text = ""
                            temp_text = map_translations(
                                translate_text,
                                app_table[key],
                                language_code,
                                source,
                            )
                            try:
                                translated_app_table[key] = temp_text
                            except json.JSONDecodeError:
                                badError = True
                                print("Error decoding translated JSON for key:", key)
                                translated_app_table[key] = {}

            except openai.error.OpenAIError as e:
                error_code = e.__dict__.get("error", {}).get("code", "Unknown")
                print(
                    "OpenAI error encountered: {}. Error code: {}. Retrying in {} seconds... (Retry {}/{})".format(
                        e, error_code, retry_delay, retry + 1, max_retries
                    )
                )
                time.sleep(retry_delay)
                if retry == max_retries:
                    badError = True

        store_translated_text_to_file(dest_filepath, translated_app_table, source)

        if badError:
            print("JSON could not be translated", file=sys.stderr)
            sys.exit(1)
