import os
import openai
import spacy
from string import punctuation
import nltk
from nltk.tokenize import word_tokenize

#last test
openai.api_key = "openai key"

def gpt3_curie(input_text, model="text-curie-001", temperature=0, max_tokens=250):
    response = openai.Completion.create(
    model= model,
    prompt=f"Extract 5 keywords from this text as a list of words: {input_text}",
    temperature= temperature,
    max_tokens= max_tokens,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0
    )
    keywords = response["choices"][0]["text"][1:]
    return keywords


def gpt3_davinci(input_text, model="text-davinci-002", temperature=0, max_tokens=250):
    response = openai.Completion.create(
    model= model,
    prompt=f"Extract 5 keywordsf rom this text as a list of words: {input_text}",
    temperature= temperature,
    max_tokens= max_tokens,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0
    )
    keywords = response["choices"][0]["text"][1:]
    return keywords

def keyword_preprocessing(text):
    keywords_curie = gpt3_curie(text)
    if len(keywords_curie) < 150:
        original_string = keywords_curie
        characters_to_remove = "12345.-"
        keyword_list = []
        new_string = original_string

        for character in characters_to_remove:
            new_string = new_string.replace(character, "")

        keyword_list.append(new_string.strip().split("\n"))


        kw_list = keyword_list[0]
        print(kw_list)
        final_list = []
        for n in kw_list:
            final_list.append(n.strip().lower())

        return final_list

    elif len(keywords_curie)>150:
        keywords_davinci = gpt3_davinci(text)

        if len(keywords_davinci) < 100:

            original_string = keywords_davinci
            characters_to_remove = "12345."
            keyword_list = []
            new_string = original_string

            for character in characters_to_remove:
                new_string = new_string.replace(character, "")

            keyword_list.append(new_string.split(","))


            kw_list = keyword_list[0]
            final_list = []
            for n in kw_list:
                final_list.append(n.strip().lower())
            return final_list

    else:
        return "Not Possible"




# def zstc_filter(user_text, keywords):
#     #zstc
#     classifier = pipeline("zero-shot-classification",
#                       model="facebook/bart-large-mnli")

#     sequence_to_classify = user_text
#     candidate_labels = keywords
#     zstc_labels = classifier(sequence_to_classify, candidate_labels, multi_label=False)['labels']
#     zstc_scores = classifier(sequence_to_classify, candidate_labels, multi_label=False)['scores']
#     zstc_result = dict(zip(zstc_labels, zstc_scores))
#     print(zstc_result)
#     #filter
#     score_list = list(zstc_result.values())
#     key_list = list(zstc_result.keys())

#     calc_list = []
#     final_list = []
#     for i in range(len(score_list)):
#         calc_list.append(1 - (score_list[i-1]/score_list[0]))

#     for score in calc_list:
#         if score < 0.8:
#             final_list.append(score)

#     num_kw = len(final_list)
#     filterd_keywords = key_list[0:num_kw]
#     return filterd_keywords


# def keyword_func(input_text, model="text-babbage-001", temperature=0.3, max_tokens=250):
#     response = openai.Completion.create(
#     model= model,
#     prompt=f"Extract 5 keywords from this text, for example 1. Keyword, 2. Keyword, 3. Keyword, 4. Keyword, 5. Keyword: {input_text}",
#     temperature= temperature,
#     max_tokens= max_tokens,
#     top_p=1.0,
#     frequency_penalty=0.8,
#     presence_penalty=0.0
#     )
#     keywords = response["choices"][0]["text"][1:]
#     original_string = keywords
#     characters_to_remove = "12345."
#     keyword_list = []
#     new_string = original_string

#     for character in characters_to_remove:
#         new_string = new_string.replace(character, "")

#     keyword_list.append(new_string.split("\n"))


#     kw_list = keyword_list[0][1:]
#     final_list = []
#     for n in kw_list:
#         final_list.append(n.strip())

#     return final_list

#article ="LeCun, who is chief scientist at Meta’s AI lab and a professor at New York University, is one of the most influential AI researchers in the world. He had been trying to give machines a basic grasp of how the world works a kind of common sense by training neural networks to predict what was going to happen next in video clips of everyday events. But guessing future frames of a video pixel by pixel was just too complex. He hit a wall. Now, after months figuring out what was missing, he has a bold new vision for the next generation of AI. In a draft document shared with MIT Technology Review, LeCun sketches out an approach that he thinks will one day give machines the common sense they need to navigate the world. (Update: LeCun has since posted the document online.) For LeCun, the proposals could be the first steps on a path to building machines with the ability to reason and plan like humans what many call artificial general intelligence, or AGI. He also steps away from today’s hottest trends in machine learning, resurrecting some old ideas that have gone out of fashion. But his vision is far from comprehensive; indeed, it may raise more questions than it answers. The biggest question mark, as LeCun points out himself, is that he does not know how to build what he describes. The centerpiece of the new approach is a neural network that can learn to view the world at different levels of detail. Ditching the need for pixel-perfect predictions, this network would focus only on those features in a scene that are relevant for the task at hand. LeCun proposes pairing this core network with another, called the configurator, which determines what level of detail is required and tweaks the overall system accordingly. For LeCun, AGI is going to be a part of how we interact with future tech. His vision is colored by that of his employer, Meta, which is pushing a virtual-reality metaverse. He says that in 10 or 15 years people won’t be carrying smartphones in their pockets, but augmented-reality glasses fitted with virtual assistants that will guide humans through their day. “For those to be most useful to us, they basically have to have more or less human-level intelligence,” he says. “Yann has been talking about many of these ideas for some time,” says Yoshua Bengio, an AI researcher at the University of Montreal and scientific director at the Mila-Quebec Institute. “But it is good to see it all together, in one big picture.” Bengio thinks that LeCun asks the right questions. He also thinks it’s great that LeCun is willing to put out a document that has so few answers. It’s a research proposal rather than a set of clean results, he says. “People talk about these things in private, but they’re not usually shared publicly,” says Bengio. “It’s risky.”"
#test = keyword_func(article)
#print(test)


def get_named_entities(text):
    "returns unique entities, list_org, list_person, list_dates and list_rest"
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    list_of_entities =[]
    for entity in doc.ents:
        list_of_entities.append((entity.text, entity.label_))

    unique_entities = list(set(list_of_entities))

    list_org = []
    list_person = []
    list_date = []
    list_rest = []
    complete_list = []

    for x in unique_entities:
        if x[1] == "DATE":
            list_date.append(x[0])

        elif x[1] == "ORG":
            list_org.append(x[0])

        elif x[1] == "PERSON":
            list_person.append(x[0])

        else:
            list_rest.append(x[0])

    for x in unique_entities:
        complete_list.append(x[0].lower())

    return complete_list,list_org, list_person, list_date, list_rest



def remove_duplicate_keywords(keywords,named_entities):
    keywords = keywords
    named_entities = named_entities
    processed_named_entities=[]
    for instance in named_entities:
        processed_named_entities.append(instance.replace("the", "").strip())

    final_keywords = [elt for elt in keywords if elt not in processed_named_entities]
    return final_keywords



def get_keywords_and_nes(text, keywords):

    named_entities_complete,list_org, list_person, list_date, list_rest = get_named_entities(text)
    final_keywords = remove_duplicate_keywords(keywords,named_entities_complete)
    return final_keywords, list_org, list_person,list_date,list_rest
