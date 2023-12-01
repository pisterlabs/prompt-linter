import openai
import os
import csv
import regex
import api_key

def log_api_interaction(message, response):
    with open("api_log.txt", "a") as log_file:
        log_file.write("API Request Message:\n")
        log_file.write(json.dumps(message, ensure_ascii=False, indent=4))
        log_file.write("\n\nAPI Response:\n")
        log_file.write(json.dumps(response, ensure_ascii=False, indent=4))
        log_file.write("\n\n---\n\n")

def generate_word_info(word, output_folder='word_data'):
    output_path = os.path.join(output_folder, f"{word}.json")

    if os.path.exists(output_path):
        print(f"Information for {word} already exists. Skipping request.")
        return

    message = [
        {'role': 'system', 'content': "You need to check 4 times the word before you do the request. For a given word, gender (m, f, neuter), article (der, die, das ), word type (if conjugated verb, replace with infinitive), translation, 2-3 example sentences (de->en), usage frequency, IPA code, verb auxiliary for past, European difficulty level, and if the verb is irregular, present 3rd form singular, past perfect, and preterite as well as frequency over a scale of 10."},
        {"role": "user", "content": f"For the German word '{word}', give me the JSON answer inside a JSON response, if error remove last e or n."}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=0,
    )

    log_api_interaction(message, response)

    text = response["choices"][0]["message"]["content"]
    json_string = regex.search(r'\{(?:[^{}]|(?R))*\}', text).group()
    json_string = regex.sub(r',\s*\}', '}', json_string)
    print(json_string)
    info = json.loads(json_string)

    os.makedirs(output_folder, exist_ok=True)

    with open(output_path, "w") as outfile:
        json.dump(info, outfile, ensure_ascii=False, indent=4)

    return info

def process_words_from_csv(input_file):
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            word = row[0]
            print(f"Processing word: {word}")
            generate_word_info(word)

if __name__ == '__main__':
    process_words_from_csv('resulting_csv/golden_words_german.csv')
