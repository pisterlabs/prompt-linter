from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from nltk.tokenize import sent_tokenize
import openai
import os
import copy
import json
import uuid


LANGUAGE = "english"
summary_length = 100 #words
summaries_per_text = 5

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
temp_dir = data_dir = os.path.join(data_dir, 'temp')

# Provide a list of links to scrape:

link_list = ["https://opensource.com/article/23/4/run-distributed-database-cloud",
            "https://opensource.com/article/23/4/edit-photos-open-source-ai",
            "https://opensource.com/article/23/4/raspberry-pi-projects-influxdb",
            "https://opensource.com/article/23/4/cluster-open-source-python-api-wrapper",
            "https://opensource.com/article/23/4/search-engine-creative-commons-openverse",
            "https://opensource.com/article/23/4/resolve-git-merge-conflicts",
            "https://opensource.com/article/23/3/open-source-accounting-run-business",
            "https://opensource.com/article/23/3/open-source-security-scorecard",
            "https://opensource.com/article/23/3/create-accessible-websites-drupal",
            "https://opensource.com/article/23/3/community-documentation"
            ]

def web_scrape(input_value):
    try:
        parser = HtmlParser.from_url(input_value, Tokenizer(LANGUAGE))
        input_value = parser.document
        sentences = []
        for paragraph in parser.document.paragraphs:
            sentences.extend(paragraph.sentences)
        text_block = ' '.join([str(sentence) for sentence in sentences])
    except Exception as error:
        text_block = str(error)
    return text_block

def init_dict(link_list):
    result_dict = {}
    for i in range(0, len(link_list)):
        link = link_list[i]
        result_dict[i] = {}
        result_dict[i]["link"] = link
        result_dict[i]["text"] = web_scrape(link)
    return result_dict


def golden_summary(input, length):
    worked = True
    len2 = len(input.split())
    length = min(length, len2)
    # 100 tokens roughly 75 words. Add extra for special cases (a factor of 80)

    m_tokens = int((100 * int(length)) / 80) 
    prompt = f'''Create a perfect summary of a given text but strictly follow these rules:
       1. Dont add extra content to the input text.
       The perfect summary should only contain information in the original text.
       2. The summary should be at most {length} words long - it should preferably be less.
       The text to summarize is: 
       '''

    model = "gpt4all-falcon-q4"
    prompt += input

    print("starting the golden summary creation... might take a while")
    
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=m_tokens,
            temperature=0.20,
            top_p=0.99,
            n=1,
            echo=False,
            stream=False
        )
    except Exception as error:
        response = str(error)
        worked = False
    return response, worked

def filter_sentences_by_word_limit(text, max_word_limit): 
    sentences = sent_tokenize(text)
    summarized_sentences = []
    word_count = 0  
    for sentence in sentences:
        current_words = str(sentence).split()
        if word_count + len(current_words) > max_word_limit:
            break
        summarized_sentences.append(str(sentence))
        word_count += len(current_words)
    return summarized_sentences

def response_to_sentences(response, word_limit):
    generated_text = response["choices"][0].text.strip()
    filtered = filter_sentences_by_word_limit(generated_text, word_limit)
    return filtered

def save_dict_to_json(data_dict, directory_path):
    unique_filename = str(uuid.uuid4()) + '.json'
    file_path = os.path.join(directory_path, unique_filename)
    with open(file_path, 'w') as json_file:
        json.dump(data_dict, json_file)
    print(unique_filename)
    return unique_filename

def main_golden_summary(input, length):
    openai.api_base = "http://localhost:4891/v1"
    openai.api_key = "not needed for a local LLM"
    summary, worked = golden_summary(input, length)
    if worked:
        return response_to_sentences(summary, length)
    return str(summary)

def create_golen_summaries(story_dict, length, ammount, temp_dir):
    new_story_dict = copy.deepcopy(story_dict)
    for dict_entry in new_story_dict.keys():
        text = new_story_dict[dict_entry]["text"]
        new_story_dict[dict_entry]["golden_summaries"] = {}
        for i in range(0, ammount):
            new_story_dict[dict_entry]["golden_summaries"][i] = main_golden_summary(text, length)
            save_dict_to_json(new_story_dict, temp_dir)
    return new_story_dict

def main():
    story_dict = init_dict(link_list)
    story_dict = create_golen_summaries(story_dict, summary_length, summaries_per_text, temp_dir)
    print(story_dict)
    final_filename = save_dict_to_json(story_dict, data_dir)
    print(final_filename)

if __name__ == "__main__":
    main()