# this program was originally intended to act as a qualitative analysis assistant, generating an interview script
# as an interview is still going, for more tailored and insightful questions. the find_most_similar_texts function
# can be adapted for use with openai embeddings to act as an analogical replacement for the word2vec most_similar function

# more in depth: this program generates an interview protocol for the user, with the purpose of optimizing the 
# interviewer's attention to the interviewee's stories. it contains functions to assist with the generation of 
# questions prior to the interview, a pre and post script for consent purposes, and questions and lines generated 
# spontaneously during the interview.

# i apologize in advance for unhinged code comments, i only left in the ones that were semi-useful to understanding the code ::::)

import openai
import readline
import spacy

from openai.embeddings_utils import cosine_similarity

def main():
    sp = spacy.load('en_core_web_sm')
    openai.api_key = 'OPENAI API KEY'

    peeps = input("what people will have access to the raw interview data?")
    tag = tagline()
    script = "hi! for your knowledge, this interview's information will only be accessible by these people: " + peeps + ". "
    ans = input("do you consent to being questioned? how about recorded? is it okay for you if we take a transcript of this recording?" + 
                "(the transcript will be sent to you for a member check before officially being used. ")
    print(script + tag)
    suggested_connection(curr_q, curr_a)
    extended_meanings(curr_q, curr_a)
    if (next_question == True and len(q_a[curr_q]) == 0):
        steer_back_on_track(curr_q, curr_a)

# function taken from stack overflow
def input_with_prefill(prompt, text):
    def hook():
        readline.insert_text(text)
        readline.redisplay()
    readline.set_pre_input_hook(hook)
    result = input(prompt)
    readline.set_pre_input_hook()
    return result

def tagline():
    end_loop = False
    explanation = input("Enter a background of what you are studying and why you are studying it (don't be afraid to be verbose): ")
    explanation_words = sp(explanation)
    jargon = [word for word in explanation_words if word.text.isupper() and len(word) != 1]
    for word in jargon:
    definition = input("Is it crucial to include " + word.text + " in your explanation? If so, provide a short definition, if not, type in \"False\": ")
    if definition != False:
        explanation += definition
    iterations = 1
    while not end_loop:
    if iterations != 1:
        explanation = input_with_prefill(explanation, explanation)
    explanation = openai.Completion.create(model = 'text-davinci-003',
                                           prompt = "Summarize this for a second-grade student:\n" + explanation,
                                           temperature = 0.7,
                                           max_tokens = 64,
                                           top_p = 1.0,
                                           frequency_penalty = 0.0,
                                           presence_penalty = 0.0)['choices'][0]['text']
    end_loop = input("Is this accurate and complete (answer with True/False)? " + explanation)
    if end_loop == "False":
        end_loop = False
        iterations += 1
    return explanation

def suggested_connection(current_question, current_answer):
    # input = current question, current answer
    # output = answers that could potentially be related, according to cosine similarity of embeddings of answers
    max = -1
    best_answer = None
    for question in questions_and_answers:
        for answer in questions_and_answers[question]:
            sim = cosine_similarity(question[answer][0], questions_and_answers[current_question][current_answer][0])
            if sim > max:
                max = sim
                best_answer = answer
    return best_answer

def extended_meanings(current_question, current_answer):
    # input = current_question, current_answer
    # output = potential assumption along with extended meaning that goes along with it
    return response = openai.Completion.create(model = 'text-davinci-003',
                                               prompt = current_answer + ". So I also ",
                                               temperature = 0.7,
                                               max_tokens = 6,
                                               top_p = 1,
                                               frequency_penalty = 0,
                                               presence_penalty = 0)['choices'][0]['text']

def steer_back_on_track(): # if next question is selected and current_question has 0 answer keywords
    # input = global vars, current_question, 
    # output = generated question
    # follow-up question that leads back to the focus
    # original question: how do you feel about geese?
    # answer: i think my history of geese started off very young, back in the days of the great gold rush. my pals, sil and bobfusco, were quite the sportsmen, they played a great game of futbol
    # follow-up question: ahh what a time to be alive, in the glory years of canada! see, the futbol of the 1800s were made of goosefeathers huh? do you believe they symbolize a sort of vengeful stance towards the great league of geese?
    return openai.Completion.create(model = 'text-davinci-003',
                                    prompt = current_question + "\n\n" + current_answer + "\n\n",
                                    temperature = 0.7,
                                    max_tokens = 42,
                                    top_p = 1,
                                    frequency_penalty = 0,
                                    presence_penalty = 0)['choices'][0]['text'] + 
           openai.Completion.create(model = 'text-davinci-003',
                                    prompt = current_question + "\n\n" + current_answer + "\n\n" + " How ",
                                    temperature = 0.7,
                                    max_tokens = 42,
                                    top_p = 1,
                                    frequency_penalty = 0,
                                    presence_penalty = 0)['choices'][0]['text']

def track_time(time_left):
    questions_left = {}
    for question in questions_and_answers: # start replacement
        if questions_and_answers[question] == None:
            questions_left[question] = openai.Embedding.create(input = question,
                                                               model = 'text-embedding-ada-002')['data'][0]['embedding']
    for question1 in questions_left:
        for question2 in questions_left:
            if (question != question):
                cosine_similarity(questions_left[question1]) # end replacement

def find_most_similar_texts(list_of_texts):
    comparisons = {}
    for i in range(len(list_of_texts)-1):
        for j in range(i + 1, len(list_of_texts)):
            embedding1 = openai.Embedding.create(input = list_of_texts[i],
                                                 model = 'text-embedding-ada-002')['data'][0]['embedding']
            embedding2 = openai.Embedding.create(input = list_of_texts[j],
                                                 model = 'text-embedding-ada-002')['data'][0]['embedding']
            comparisons[frozenset({list_of_texts[i], list_of_texts[j]})] = cosine_similarity(embedding1, embedding2)
    max_val = max(comparisons.values())
    for comparison in comparisons:
        if max_val == comparisons[comparison]:
            return list(comparison)
            break
    return []

if __name__ == "__main__":
    main()
