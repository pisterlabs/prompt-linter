# ! pip install openai --quiet
# ! pip install regex --quiet
# ! pip install appdirs --quiet
# ! pip install markupsafe==2.0.1 --quiet
# ! pip install jinja2 --quiet
# ! pip install promptify --quiet
# ! pip install pandas --quiet
# ! pip install -U pip setuptools wheel --quiet
# ! pip install -U spacy -- quiet
# ! python -m spacy download en_core_web_sm
# ! python3 -m pip install nltk --quiet

import json
import time
import pandas as pd
import spacy
import openai
import promptify
import nltk

from openai.embeddings_utils import get_embedding, cosine_similarity
from promptify import OpenAI
from promptify import Prompter
from nltk.sentiment import SentimentIntensityAnalyzer

def main():
    # set up imported models
    sp = spacy.load('en_core_web_sm')
    openai.api_key = 'OPENAI API KEY'
    model = OpenAI(openai.api_key)
    nlp_prompter = Prompter(model)
    nltk.download(['names',
                   'stopwords',
                   'state_union',
                   'twitter_samples',
                   'movie_reviews',
                   'averaged_perceptron_tagger',
                   'vader_lexicon',
                   'punkt'])

    # prepare the input data
    # 1) engineer input data
    #    description: upwork top 6 results for "industrial design" jobs
    upwork_1 = "finish existing enclosures and/or redesign if necessary. I am creating a device that will test lighting systems on commercial automotive vehicles. One enclosure holds a PCB, battery, and othervarious electrical components. The second enclosure is for a remote that will hold the samecomponents, will be used to operate the unit. I plan on creating the enclosures with Resin Casting,which is the main focus on this project. After casting, th enclosure will be CNC operated to drillhole cut-outs. Experienced Skills -Weatherproof compatibilities -Remote designs with conductivemembranes -Silicone shell casing -Communication -Creativity -CNC file creation Project DeviceRequirements -Ergonomics and Visually Appealing -Design created for Resin Casting -Exceptional Durability"
    upwork_2 = "looking for a furniture or industrial design engineer for a 3-6 month project to help design an ergonomic desk. There has never been a larger population of Americans who work and spend the majority of their waking hours at a desk. This has resulted in a growing number of repetitive motion injuries and sprains/strains in the neck and back from incorrect posture. I'm frustrated that there is a complete lack of options in the US for fully modular zero gravity ergonomics desks (see attached). I''m looking for someone who can help me design a desk that can allow you to recline while working at the computer/laptop which I feel could revolutionize the way office employees work."
    upwork_3 = "I need a functional design for a clip-on accessory for a mobile phone similar to this. https://a.co/d/2RsNSEs The light should have two options (RGB sound activated or bright white). Should have its own power source, preferably coin batteries, to make it lightweight. And has an interchangeable face for custom logos/branding. Deliverables will be - Solidworks files (parts and assembly) - STLs for 3D printing - Renderings"
    upwork_4 = "We are looking for someone who can design a adaptable packing station for use in a warehouse with interchangeable packing materials holders, find relevant competitive suppliers for parts and provide BOM and assembly guide. Basic requirements: -height adjustable - should hold bar code scanner, 2 label printers, computer, monitor, scales - have lightning - be mobile - be ergonomic It will be positioned next to a conveyor. Further details will be provided in the messaging. Price for the service is not relevant at this point and will be negotiated once you will gather necessary information."
    upwork_5 = "Good day, folks. Please don't drop your application without carefully reading the job instructions. I'm looking for two 3D modelers to design a wearable device (i.e.) a wristband that will detect body temperature, heart rate, etc. Have a look at this Loom, https://www.loom.com/share/f606c13335d5423d8f28497bed3d1bec Technical responsibilities: - Design a 3D model for a wearable device. - Must share a model implementation plan. - Modeling must follow our requirements. - First version needs to be done within one week. Tools requirements: - You must use Jira - Integrate Jira with Upwork for accurate activity tracking. - Use GitHub for ticket submission. - Communicate using Loom. Processual concerns: - I'll assist you with your onboarding steps."
    upwork_6 = "company, looking for someone in greater Los Angeles area. If you are someone who is creative, and like to design. This is a great opportunity. We have multiple projects on-hand, and would love to have a designer to work with on regular bases. This person should experiences in design/modifying outdoor furniture category. Has experiences in design/modifying working in the outdoor furnishing fields. This project: new product concept/design - Garden Hose Reel/storage *combine the current design on the market to create new product *need to understand basic functionality of the product *base will be metal and house is plastic *create a new reel that is functional and good looking candidate must be in the greater Los Angeles area"
    prompts_info = [upwork_1, upwork_2, upwork_3, upwork_4, upwork_5, upwork_6]

    # process the input data in 3 separate ways (for comparison to each other)
    # response_1: extract embeddings for each prompt as is
    # response_2: take the keywords for each prompt
    #             ask GPT-3 to dig into the core reason of being of each keyword (so length of text turned into embedding doesn't affect)
    #             extract the embedding of the GPT-3-given descriptions
    # response_3: ask GPT-3 to summarize the prompt (to minimize the number of repeated keywords)
    #             take the keywords for each prompt
    #             ask GPT-3 to dig into the core reason of being of each keyword (so length of text turned into embedding doesn't affect)
    #             extract the embedding of the GPT-3-given descriptions
    response_1 = extract_embeddings(prompts_info)
    response_2 = extract_keyword_embeddings(prompts_info)
    response_3 = extract_summarized_keyword_embeddings(prompts_info)

    # prepare the projected output data
    # 2) biologist projected output data (to match engineer input to)
    #    description: asknature top 28 articles
    bio_functions = ["build strong but flexible reefs using minerals in a sticky protein web",                                                #  0
                    "allow water to escape as vapor, drawing more water up through the plant from the roots",                                 #  1
                    "destroy fungi by breaking down their cell membranes",                                                                    #  2
                    "create color by causing light waves to diffract and interfere",                                                          #  3
                    "attracts erstwhile predators which then become unwitting nannies and bus drivers for the sedentary animal's offspring",  #  4
                    "causes water flow to stretch across fish skin, reducing turbulence and minimizing drag",                                 #  5
                    "allows it to dive into the water without splashing",                                                                     #  6
                    "enables it to swim forward and backward, as well as keep it afloat, by creating propulsive water jets",                  #  7
                    "help the Swiss cheese plant capture intermittent light",                                                                 #  8
                    "intercept and de-energize harmful ultraviolet radiation before it reaches the plant's cells",                            #  9
                    "expose more skin to the sun and create an insulating layer of air to reduce heat loss",                                  # 10
                    "concentrates hormones that alter the water levels in cells causing plants to bend toward the light source",              # 11
                    "fake having a broken wing in order to lead predators away from their nest and protect their young",                      # 12
                    "prevents bacterial attachment",                                                                                          # 13
                    "expel a pheromone that mimics the alarm pheromone of their aphid predators",                                             # 14
                    "depend more on protein diversity than quantity",                                                                         # 15
                    "use fringed sheets of keratin to strain food from water before they swallow",                                            # 16
                    "send out broadband whistles and bursts of clicks to prevent messages from being distorted underwater",                   # 17
                    "develop lifelong friendships early on that will benefit them through shared information, cooperation, and other means",  # 18
                    "adapt to changing situations by learning from peers and not just mothers",                                               # 19
                    "lay down proteins in an organized way to create a scaffod for minerals that produce rock-hard reefs",                    # 20
                    "produce color by using the sun's energy to generate specific wavelengths of light",                                      # 21
                    "protect the core colony by programmed breakage",                                                                         # 22
                    "anchor sponges in soft sediments using barbed tips",                                                                     # 23
                    "are protected from excess sun by blue iridescence",                                                                      # 24
                    "regain shape after dehydration due to hierarchical structure of palisade and spongy layers",                             # 25
                    "hyperaccumulate toxic arsenic using a special transporter protein that spatially isolates the chemical in vacuoles",     # 26
                    "survive extreme water loss thanks to dehydrin proteins",                                                                 # 27
                    "enables it to swim forward and backward, as well as keep it afloat, by creating propulsive water jets"]                  # 28

    # process the bio projected output data to represent it with the benefit presented by each article
    # target_response: extract embeddings for each article
    target_responses = extract_benefit_of_bio_article_embeddings(bio_functions)

    # compute which input matches best with each target, for each type of input processing
    # look for the value of 1.x (should display the maximum similarity in the group when you subtract 1 from that entry)
    # the 1 was just added to make the answer (predicted output) more easy to find
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(embedding_similarity_table(response_1, target_responses))
        print(embedding_similarity_table(response_2, target_responses))
        print(embedding_similarity_table(response_3, target_responses))

    # prepare the input data
    # 3) alternative engineer input data
    #    description: upwork top 6 results for "industrial design" jobs, manually extracted the product from each
    products = ["remote enclosure to test lighting on cars", 
                "ergonomic zero gravity desk", 
                "clip-on lighted accessory for phone", 
                "height adjustable packing station", 
                "wearable health tracking device", 
                "garden hose reel"]

    # preprocess the input data to find potential pain points when engineering specified product
    # turn the GPT-3-generated response into an embedding
    responses = []
    for product in products:
        dictionary = {}
        dictionary['Prompt'] = product
        response = openai.Completion.create(model = 'text-davinci-003',
                                            prompt = "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: What are some issues that can arise when designing a " + prod + "?\nA:",
                                            temperature = 0,
                                            max_tokens = 100,
                                            top_p = 1,
                                            frequency_penalty = 0.0,
                                            presence_penalty = 0.0,
                                            stop = ["\n"])
        dictionary['Embeddings'] = [openai.Embedding.create(input = response['choices'][0]['text'], 
                                                            model = 'text-embedding-ada-002')['data'][0]['embedding']]
        responses.append(dictionary)

    # compute which input matches best with each target, for each type of input processing
    # look for the value of 1.x (should display the maximum similarity in the group when you subtract 1 from that entry)
    # the 1 was just added to make the answer (predicted output) more easy to find
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(embedding_similarity_table(responses, target_responses))

def extract_embeddings(prompts):
  prompt_embeddings = []
  for prompt in prompts:
    dictionary = {}
    dictionary['Prompt'] = prompt
    dictionary['Embedding'] = [openai.Embedding.create(input = prompt,
                                                       model = 'text-embedding-ada-002')['data'][0]['embedding']]
    prompt_embeddings.append(dictionary)
  return prompt_embeddings

def extract_keyword_embeddings(prompts):
  prompt_keyword_embeddings = []
  for prompt in prompts:
    dictionary = {}
    dictionary['Prompt'] = prompt
    keywords = nlp_prompter.fit('ner.jinja',                                                                                                  # take the keywords for each prompt
                                domain = None,
                                text_input = prompt,
                                labels = None)
    keywords = keywords['text'][3:-2].split("}, {")
    keyword_info_list = []
    for keyword in keywords:
      keyword_info = {}
      if ("Branch" in keyword or "branch" in keyword):
        continue
      keyword_info['Type'] = keyword[keyword.index("'T': '") + 6 : keyword.index(",") - 1]
      keyword_info['Entity'] = keyword[keyword.index("'E': '") + 6 : -1]
      if (keyword_info['Type'] != "Job Title" and
          keyword_info['Type'] != "Person" and 
          keyword_info['Type'] != "Location" and 
          keyword_info['Type'] != "price" and
          "Time" not in keyword_info['Type']):
        keyword_info_list.append(keyword_info['Entity'])
    dictionary['Keywords'] = keyword_info_list
    functions = []
    for i in range(len(dictionary['Keywords'])):
      function = openai.Completion.create(model = 'text-davinci-003',                                                                         # ask GPT-3 to dig into the core reason of being of each keyword
                                          prompt = "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: What is the core function of " + dictionary["Keywords"][i] + "?\nA:",
                                          temperature = 0,
                                          max_tokens = 100,
                                          top_p = 1,
                                          frequency_penalty = 0.0,
                                          presence_penalty = 0.0,
                                          stop = ["\n"])
      function = function['choices'][0]['text']
      functions.append(function)
    dictionary['Functions'] = functions
    embeddings = []
    for function in dictionary['Functions']:
      embeddings.append(openai.Embedding.create(input = function,                                                                             # extract the embedding of the GPT-3-given descriptions
                                                model = 'text-embedding-ada-002')['data'][0]['embedding'])
    dictionary['Embeddings'] = embeddings
    prompt_keyword_embeddings.append(dictionary)
    time.sleep(5)
  return prompt_keyword_embeddings

def extract_summarized_keyword_embeddings(prompts):
  tldr = " Tl;dr"
  dictionaries = []
  for input_prompt in prompts:
    dictionary = {}
    dictionary['Prompt'] = prompt
    summary = openai.Completion.create(model = 'text-davinci-003',                                                                            # ask GPT-3 to summarize the prompt
                                       prompt = input_prompt + tldr,
                                       temperature = 0,
                                       max_tokens = 90,
                                       top_p = 1.0,
                                       frequency_penalty = 0.0,
                                       presence_penalty = 0.0)
    summary = summary['choices'][0]['text']
    dictionary['Summary'] = summary
    keywords = nlp_prompter.fit('ner.jinja',                                                                                                  # take the keywords for each prompt
                                domain = None,
                                text_input = summary,
                                labels = None)
    keywords = keywords['text'][3:-2].split("}, {")
    keyword_info_list = []
    for keyword in keywords:
      keyword_info = {}
      if ("Branch" in keyword or "branch" in keyword):
        continue
      keyword_info['Type'] = keyword[keyword.index("'T': '") + 6 : keyword.index(",") - 1]
      keyword_info['Entity'] = keyword[keyword.index("'E': '") + 6 : -1]
      if (keyword_info['Type'] != "Job Title" and
          keyword_info['Type'] != "Person" and 
          keyword_info['Type'] != "Location" and 
          keyword_info['Type'] != "price" and
          "Time" not in keyword_info['Type']):
        keyword_info_list.append(keyword_info['Entity'])
    dictionary['Keywords'] = keyword_info_list
    functions = []
    for i in range(len(dictionary['Keywords'])):
      function = openai.Completion.create(model = 'text-davinci-003',                                                                         # ask GPT-3 to dig into the core reason of being of each keyword
                                          prompt = "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: What is the core function of " + dictionary['Keywords'][i] + "?\nA:",
                                          temperature = 0,
                                          max_tokens = 100,
                                          top_p = 1,
                                          frequency_penalty = 0.0,
                                          presence_penalty = 0.0,
                                          stop = ["\n"])
      function = function['choices'][0]['text']
      functions.append(function)
    dictionary['Functions'] = functions
    embeddings = []
    for function in dictionary['Functions']:
      embeddings.append(openai.Embedding.create(input = function,                                                                             # extract the embedding of the GPT-3-given descriptions
                                                model = 'text-embedding-ada-002')['data'][0]['embedding'])
    dictionary['Embeddings'] = embeddings
    dictionaries.append(dictionary)
    time.sleep(5)
  return dictionaries

def extract_benefit_of_bio_article_embeddings(articles):
    embeddings = []
    for function in functions:
        doc = sp(function)
        list_indices = []
        count = 0
        for token in doc:                                                                                                                     # divide up each article title into phrases starting with verbs
            if token.pos_ == "VERB":
                result = function.index(token.text)
                list_indices.append(result)
                count += 1
        parts = [function[i:j] for i,j in zip(list_indices, list_indices[1:]+[None])]
        sentiments_for_parts = []
        for part in parts:
            sentiments_for_parts.append(SentimentIntensityAnalyzer().polarity_scores(part)['compound'])
        function = parts[sentiments_for_parts.index(max(sentiments_for_parts))]                                                               # the phrase with the most positive sentiment is used to represent
        embeddings.append(openai.Embedding.create(input = function,                                                                           # the article for presenting the benefit of using the bio mechanism
                                                  model = 'text-embedding-ada-002')['data'][0]['embedding'])                                  # covered in the article
        time.sleep(5)

def embedding_similarity_table(response_dictionary, bio_embeddings):
  dictionary = {}
  for prompt in response_dictionary:
    count = 0
    for embedding in prompt['Embeddings']:
      similarities = []
      for bio_embedding in bio_embeddings:
        similarity = cosine_similarity(embedding, bio_embedding)
        similarities.append(similarity)
      # find max in row -> add 1
      index = similarities.index(max(similarities))
      similarities[index] += 1
      name = prompt['Prompt'][:15] + "-" + str(count)
      dictionary[name] = similarities
      count += 1
  return pd.DataFrame(dictionary)

if __name__ == "__main__":
    main()
