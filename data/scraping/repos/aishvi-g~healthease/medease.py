import cohere
import os
from health_data import health_database, train_data
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem

api_key = os.environ['API_KEY']
azure_key = os.environ['azure_key']
endpoints = os.environ['endpoint']
region = os.environ['region']

credential = TranslatorCredential(azure_key, region)
text_translator = TextTranslationClient(endpoint=endpoints,
                                        credential=credential)

co = cohere.Client(api_key)


class cohereExtractor():

  def __init__(self, examples, example_labels, labels, task_desciption,
               example_prompt):
    self.examples = examples
    self.example_labels = example_labels
    self.labels = labels
    self.task_desciption = task_desciption
    self.example_prompt = example_prompt

  def make_prompt(self, example):
    examples = self.examples + [example]
    labels = self.example_labels + [""]
    return (self.task_desciption + "\n---\n".join([
        examples[i] + "\n" + self.example_prompt + labels[i]
        for i in range(len(examples))
    ]))

  def extract(self, example):
    extraction = co.generate(model='xlarge',
                             prompt=self.make_prompt(example),
                             max_tokens=15,
                             temperature=0.1,
                             stop_sequences=["\n"])
    return (extraction.generations[0].text[:-1])


cohereHealthExtractor = cohereExtractor(
    [e[1] for e in train_data], [e[0] for e in train_data], [], "",
    "extract the Keywords from the medical terminology related answers:")
text = cohereHealthExtractor.make_prompt(
    'What are alternatives for paracetamol')
target_language_code = "en"


def translate_text(text, target_language):

  target_languages = [target_language]
  input_text_elements = [InputTextItem(text=text)]

  response = text_translator.translate(content=input_text_elements,
                                       to=target_languages)
  translation = response[0] if response else None
  if translation:
    for translated_text in translation.translations:
      return translated_text.text
  else:
    return text


def extract_keywords(input_text):
  extraction = cohereHealthExtractor.extract(input_text)
  keywords = extraction.split(',')
  keywords = [keyword.strip().lower() for keyword in keywords]
  return keywords


def search_answer(keywords):
  for keyword, answer in health_database:
    if keyword.lower() in keywords:
      return answer
  return "I'm sorry, but I'm unable to provide information on that topic. For accurate and reliable information, please consult a healthcare professional or trusted educational resources."


def generate_response(user_input, target_language):

  keywords = extract_keywords(user_input)
  answer = search_answer(keywords)

  translated_answer = translate_text(answer, target_language)

  return translated_answer + "\n\n" + "Keywords: " + ", ".join(keywords)
