import openai
import json
import sys
import argparse
import os

# Top 10 languages by number of speakers
languages = [
    ("zh", "Chinese"),
    ("zh_Hant", "Traditional Chinese"),
    ("vi", "Vietnam"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("it", "Italian"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("en", "English"),
    ("hi", "Hindi"),
    ("ar", "Arabic"),
    ("pt", "Portuguese"),
    ("bn", "Bengali"),
    ("ru", "Russian"),
    ("jv", "Javanese"),
]

def translate_data(data, target_language, full_translation, existing_translations):
    data_without_app_name = {k: v for k, v in data.items() if k != "app_name"}

    if not full_translation:
        data_without_app_name = {k: v for k, v in data_without_app_name.items() if k not in existing_translations}

    if len(data_without_app_name) == 0:
        print("no new strings need translations")
        return data_without_app_name

    json_string = json.dumps(data_without_app_name, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": f"Translate the following JSON content(respect the Flutter l10n arb language coding grammer) from English to {target_language}: {json_string}"
        }
    ]

    print(f"{messages}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
    )

    translated_text = response.choices[0].message["content"].strip()
    print(f"{translated_text}")
    return json.loads(translated_text)

def translate_file(file_path, full_translation):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for lang_code, lang_name in languages:
        if lang_code == "en":  # Skip English
            continue

        existing_translations = {}
        output_file = f"app_{lang_code}.arb"

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing_translations = json.load(f)

        translated_data = translate_data(data, lang_name, full_translation, existing_translations)
        if existing_translations:
#             translated_data.update(existing_translations)
            existing_translations.update(translated_data)
            translated_data = existing_translations

        translated_data["app_name"] = data["app_name"]  # Add the untranslated app_name field

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        print(f"Translated and saved {file_path} to {output_file} in {lang_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=False, help="openai api key")
    parser.add_argument("--file", required=True, help="Path to the input .arb file")
    parser.add_argument("--full_translation", action="store_true", help="Translate all strings, even if already translated")

    args = parser.parse_args()

    if args.key is not None:
        openai.api_key = args.key
    else:
        openai.api_key = os.environ["OPENAI_API_KEY"]

    translate_file(args.file, args.full_translation)