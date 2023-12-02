import os
import sys
import openai
import time
import shutil
from translator import store_translated_text_to_file, translate_text, get_languages

# set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORG_ID"]
source = "txt"
max_retries = 3
retry_delay = 60
translation_table = {}
source_dir = sys.argv[1]
languages = get_languages()

# define files to exclude
exclude_files = []

# loop through languages
for language_code, country_code in languages.items():
    # create destination directory
    if language_code == "en":
        continue

    dest_dir = "{}-{}-{}".format(language_code, country_code, source)
    output_directory = sys.argv[2]
    dest_dir_full = os.path.join(output_directory, dest_dir)
    os.makedirs(dest_dir_full, exist_ok=True)
    # copy and translate text files
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt") and filename not in exclude_files:
            print("Processing filename:", filename)
            source_filepath = os.path.join(source_dir, filename)
            dest_filepath = os.path.join(dest_dir_full, filename)

            # Check if the source file is blank
            if os.path.getsize(source_filepath) == 0:
                if language_code != "en" or country_code != "US":
                    print("Copying blank file: {}".format(filename))
                    shutil.copy(source_filepath, dest_filepath)
                continue

            print(
                "Translating file: {} to language: {}".format(filename, language_code)
            )

            # read text from source file
            with open(source_filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # If the text in the source file is blank, copy the file without translating
            if text.strip() == "":
                print("Skipping blank translation for file: {}".format(filename))
                shutil.copy(source_filepath, dest_filepath)
            else:
                for retry in range(max_retries):
                    try:
                        translated_text = translate_text(text, language_code, source)
                        break
                    except openai.error.OpenAIError as e:
                        error_code = e.__dict__.get("error", {}).get("code", "Unknown")
                        print(
                            "OpenAI error encountered: {}. Error code: {}. Retrying in {} seconds... (Retry {}/{})".format(
                                e, error_code, retry_delay, retry + 1, max_retries
                            )
                        )
                        time.sleep(retry_delay)

                if translated_text.strip() == "":
                    print("Skipping blank translation for file: {}".format(filename))
                    shutil.copy(source_filepath, dest_filepath)
                else:
                    store_translated_text_to_file(
                        dest_filepath, translated_text, source
                    )

# Copy excluded files to destination directory
for language_code, country_code in languages.items():
    dest_dir = "{}-{}-{}".format(source_dir, language_code, country_code)
    output_directory = sys.argv[2]
    dest_dir_full = os.path.join(output_directory, dest_dir)
    for filename in exclude_files:
        if os.path.exists(os.path.join(source_dir, filename)):
            if language_code != "en" or country_code != "US":
                source_filepath = os.path.join(source_dir, filename)
                dest_filepath = os.path.join(dest_dir_full, filename)
                shutil.copy(source_filepath, dest_filepath)
