import openai, configparser, time, json, requests, spacy, os, numpy as np
import time, json, requests
from langdetect import detect
from googletrans import Translator
from bs4 import BeautifulSoup

huggingfacemodels = {
    'sc':"https://api-inference.huggingface.co/models/haining/scientific_abstract_simplification",
    'bart-sc': "https://api-inference.huggingface.co/models/sambydlo/bart-large-scientific-lay-summarisation",
    'kis': "https://api-inference.huggingface.co/models/philippelaban/keep_it_simple"
}

max_length = 2000
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

languages = {
    'nl':'nl_core_news_md',
    'en':'en_core_web_md'
}

class HuggingFaceModels:
    def __init__(self, key=None):
        global huggingface_api_key
        try:
            huggingface_api_key = key
        except:
            huggingface_api_key = 'not_submitted'

    """"""
    def query(self, payload, API_URL):
        headers = {"Authorization": f"Bearer {huggingface_api_key}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    """"""
    def scientific_simplify(self, text, lm_key):
        length = len(text)
        API_URL = huggingfacemodels.get(lm_key)
        gt = Translator()
        translated_text = gt.translate(text=text,src='nl',dest='en').text
        result = self.query({"inputs": str(translated_text),"parameters": {"max_length": length},"options":{"wait_for_model":True}}, API_URL)[0]['generated_text']
        result = gt.translate(text=result,src='en',dest='nl').text
        return result

    def summarize(self, text, lm_key):
        gt = Translator()        
        soup = BeautifulSoup(text, 'html.parser')
        tags = soup.find_all(True)
        split_text = {}
        for tag in tags:
            if tag.name == 'h3':
                current_key = tag.text
            if tag.name == 'p':
                split_text[current_key] = tag.text

        for key in split_text.keys():
            split_text[key] = str(split_text[key]).strip('\n').replace('\n', ' ').replace('\\','')

        result_dict = {}
        for key in split_text.keys():
            text = split_text[key]
            origin_lang = detect(text)
            nlp = spacy.load(languages.get(origin_lang, 'en'))
            doc = nlp(text)

            sentences = []
            for s in doc.sents:
                try:
                    text = gt.translate(text=str(s), dest='en').text
                    sentences.append(text)
                except Exception as e:
                    print(e)

            API_URL = huggingfacemodels.get(lm_key)
            sentences = np.array(sentences)
            pad_size = 3 - (sentences.size % 3)
            padded_a = np.pad(sentences, (0, pad_size), mode='empty')
            paragraphs = padded_a.reshape(-1, 3)

            output = []
            text = ""
            for i in paragraphs:
                length = len(str(i))
                result = self.query({"inputs": str(i),"parameters": {"max_length": length},"options":{"wait_for_model":True}}, API_URL)

                try:
                    if 'generated_text' in result[0]:
                        text = result[0].get('generated_text')

                    if 'summary_text' in result[0]:
                        text = result[0].get('summary_text')
                except Exception as e:
                    print(e)

                lang = detect(text)
                try:
                    text = gt.translate(text=str(text),src=lang, dest='nl').text 
                except Exception as e:
                    print(str(e))
                    
                output.append(text)
            result_dict[key] = output
        return(result_dict)            

    """@returns a translated sentence"""
    def translate_sentence(self, sentence):
        translator  = Translator()
        result = translator.translate(
            text=sentence,
            dest='nl')
        return result.text

class GPT():
    """ @sets openai.api_key """
    def __init__(self, key=None):
        global gpt_api_key
        if key is None:
            gpt_api_key = 'not-submitted'
            openai.api_key = key
        else:
            gpt_api_key = key
            openai.api_key = key

    """ @returns prompt, result from gpt """
    def look_up_word_gpt(self, word, context):
        try:
            prompt = f"""
            Give a simple Dutch explanation in one sentence for this word in the given context. Give the PoS-tag and Dutch definition: '{word}'
            context: {context}
            format: PoS-tag | definition
            ///
            """
            result = openai.Completion.create(
                    prompt=prompt,
                    temperature=0,
                    max_tokens=50,
                    model=COMPLETIONS_MODEL,
                    top_p=0.9,
                    stream=False
                    )["choices"][0]["text"].strip(" \n")    
            return result, word, prompt
        except Exception as e:
            return 'error', str(e), str(e)
        
    """ @returns prompt, result from gpt """
    def give_synonym(self, word, context):
        try:
            prompt = f"""
            Give a Dutch synonym for '{word}'. If there is no Dutch synonym available, explain it between curly brackets.
            context:
            {context}
            """
            result = openai.Completion.create(
                    prompt=prompt,
                    temperature=0,
                    max_tokens=10,
                    model=COMPLETIONS_MODEL,
                    top_p=0.9,
                    stream=False
                    )["choices"][0]["text"].strip(" \n")    
            return result, word, prompt
        except Exception as e:
            return 'Open AI outage of problemen', str(e)



    def personalised_simplify(self, sentence, personalisation):
        if 'summary' in personalisation:
            prompt = f"""
            Simplify the sentences in the given text and {", ".join(personalisation)}
            :return: A list of simplified sentences divided by a '|' sign
            ///
            {sentence}
            """
        else:
            prompt = f"""
            Explain this in own Dutch words and {", ".join(personalisation)}
            ///
            {sentence}
            """

        try:
            result = openai.Completion.create(
                    prompt=prompt,
                    temperature=0,
                    max_tokens=len(prompt),
                    model=COMPLETIONS_MODEL,
                    top_p=0.9,
                    stream=False
            )["choices"][0]["text"].strip(" \n")


            if 'summary' in personalisation:
                result = result.split('|')
            else:
                result = [result]
            
            return result, prompt
        except Exception as e:
            return str(e), prompt 
        
    def personalised_simplify_w_prompt(self, sentences, personalisation):
        try:
            result = openai.Completion.create(
                    prompt=personalisation,
                    temperature=0,
                    max_tokens=len(personalisation)+len(sentences),
                    model=COMPLETIONS_MODEL,
                    top_p=0.9,
                    stream=False
            )["choices"][0]["text"].strip(" \n")
            return result, personalisation
        except Exception as e:
            return str(e), personalisation
        
        
    def simplify(self, full_text_dict, personalisation):
        soup = BeautifulSoup(full_text_dict, 'html.parser')
        tags = soup.find_all(True)
        split_text = {}

        for tag in tags:
            if tag.name == 'h3':
                current_key = tag.text
            
            if tag.name == 'p':
                split_text[current_key] = tag.text

        for key in split_text.keys():
            split_text[key] = str(split_text[key]).strip('\n')\
            .strip('\\').replace('\\','')

        new_text = {}
        for title in split_text.keys():
            text = split_text[title]
            if len(text) > 1000:
                index = len(text) // 2
                text_to_prompt = [text[:index], text[index:] ]
            else:
                text_to_prompt = [text]

            full_chunk_result = ""
            for chunk in text_to_prompt:

                # TODO aangepast voor oorspronkelijke taal
                if 'summation' not in personalisation:
                    prompt = f"""
                    Rewrite this with {", ".join(personalisation)}
                    ///
                    {chunk}
                    """
                else:
                    prompt = f"""
                    Rewrite this as a list of simplified Dutch sentences with {", ".join(personalisation)}
                    :return: A list of simplified sentences divided by a '|' sign
                    ///
                    {chunk}
                    """

                full_chunk_result += str(openai.Completion.create(prompt=prompt,temperature=0,max_tokens=500,model=COMPLETIONS_MODEL,top_p=0.9,stream=False)["choices"][0]["text"].strip(" \n"))
            new_text[title] = [full_chunk_result]
        return new_text
        
        
class WordScraper():
    def look_up(self, woord):
        url = f"https://www.vertalen.nu/betekenis?woord={woord}&taal=nl"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        def_blocks = soup.find_all('ul', class_='def-block')
        definities = []
        for card in def_blocks:
            card_text = str(card.get_text()).strip(' ')
            definities.append(card_text)
        return definities   