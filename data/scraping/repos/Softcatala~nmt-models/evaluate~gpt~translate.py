import openai
import time

#SOURCE_LANG="English"
#TARGET_LANG="Catalan"
#SOURCE_FILE="flores200.eng"
#PREDICT_FILE="flores200-predict.cat"

SOURCE_LANG="Catalan"
TARGET_LANG="English"
SOURCE_FILE="flores200.cat"
PREDICT_FILE="flores200-predict.eng"

with open(SOURCE_FILE) as in_file, open (PREDICT_FILE, "w") as out_file:

    id = 0
    for text in in_file:
        completion = ""
        while True:
            try:
                completion = openai.ChatCompletion.create(
                  model = "gpt-3.5-turbo",
                  temperature = 0,
                  max_tokens = 2000,
                  messages = [
                      {"role": "system", "content": f"You are a helpful assistant that translates {SOURCE_LANG} to {TARGET_LANG}."},
                      {"role": "user", "content": f'Translate the following {SOURCE_LANG} text to {TARGET_LANG}: "{text}"'}
                  ]
                )
                break
            except Exception as error:
                print(str(error))
                time.sleep(10)

        translated = completion.choices[0].message["content"]
        print(f"{id} - {translated}")
        out_file.write(f"{translated}\n")
        id += 1
