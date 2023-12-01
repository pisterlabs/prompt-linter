import streamlit as st
import pandas as pd
import numpy as np
import os
from trubrics import Trubrics
import pickle
import tiktoken
import datetime
import uuid
from trubrics.integrations.streamlit import FeedbackCollector
from streamlit_extras.customize_running import center_running
 

import openai

openai.api_key = st.secrets["apikey"]

st.set_page_config(
    page_title="FagBotten",
    page_icon="🌍")
center_running()

if 'df' not in st.session_state:
    df = pd.read_csv('df_enc.csv')
    st.session_state['key'] = df

with open('document_embeddings.pkl', 'rb') as fp:
    document_embeddings = pickle.load(fp)

collector = FeedbackCollector(
    email=st.secrets.TRUBRICS_EMAIL,
    password=st.secrets.TRUBRICS_PASSWORD,
    project="FagBotten"
)


def make_clickable(val, kilde):
    return f'<a target="_blank" href="{val}">{kilde}</a>'



def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = openai.Moderation.create(input=question)
    if response.results[0].flagged:
        # get the categories that are flagged and generate a message
        result = [
            error
            for category, error in errors.items()
            if response.results[0].categories[category]
        ]
        return result
    return None

def get_embedding(text: str, model: str="text-embedding-ada-002") -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def construct_prompt(question: str, previous_questions, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    # st.write(f"Vigtigste {len(chosen_sections)} kilder:")
    #st.write("\n".join(chosen_sections))df_styled = df.head().style.format({'url': lambda x: make_clickable(x, df.loc[df['url'] == x]['Kilde'].values[0])})
    kilder = df[['Kilde', 'url']].iloc[chosen_sections_indexes]
    # for i in range(kilder.shape[0]):
    #     url = kilder.iloc[i]['url'] 
    #     st.write("[" + str(kilder.iloc[i]['Kilde']) + "](%s)" % url)
    #st.write(df[['Kilde', 'url']].iloc[chosen_sections_indexes].style.format({'url': lambda x: make_clickable(x, df.loc[df['url'] == x]['Kilde'].values[0])}))
        
    return chosen_sections, chosen_sections_len, kilder

def get_response(new_question, df, document_embeddings):
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot

    Returns:
        The response text
    """
    # build the messages
    
    prompt, section_lenght, kilder = construct_prompt(
        new_question,
        st.session_state.messages,
        document_embeddings,
        df
    )

    context= ""
    for article in prompt:
        context = context + article 

    

    # messages.append({"role" : "user", "content":context})
    # # add the previous questions and answers
    # for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
    #     messages.append({ "role": "user", "content": question })
    #     messages.append({ "role": "assistant", "content": answer })
    # # add the new question
    # messages.append({ "role": "user", "content": new_question })
    #st.write([{ "role": "system", "content": INSTRUCTIONS }] + [{ "role": "system", "content": context}] + st.session_state.messages)
    completion = openai.ChatCompletion.create(
        model=COMPLETIONS_MODEL,
        messages=[{ "role": "system", "content": INSTRUCTIONS }] + [{ "role": "system", "content": context}] + st.session_state.messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message, kilder

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-4-1106-preview"

TEMPERATURE = 0.1
TOP_P = 0
MAX_TOKENS = 1500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 3

#INSTRUCTIONS = """Du skal komme med et forkert svar hver gang"""
INSTRUCTIONS = """Du er en rådgiver chatbot, som rådgiver teknikere som laver tilstandsrapporter ude hos kunder. De kan spørge dig hvis de er i tvivl om hvad de skal svare. 
Der er lavet en embedding model, som finder de relevante tekster at forholde dig til. Disse tekster findes i beskederne fra rollen "system".
Du kan kun svare ud fra den kontekst du er blevet tilført igennem beskederne fra rollen "system". 
Noget af den kontekst du er tilført kommer fra forskellige år f.eks. bygningsreglementet. Det nyeste er det du skal forholde dig til. Hvis der kommer med svar fra et andet år skal du specificere at du har brugt dokumentet og fra hvilket år.  
Svar så godt på spørgsmålet du kan, men hvis du ikke kan svare på spørgsmålet ordentligt, men mangler mere information så spørg efter det. Tror du der kunne være et andet spørgsmål så foreslå om det hander om det istedet.
Hvis du ikke kan svare så forklar hvad du mangler information omkring. Hvis du ikke forstår spørgsmålet i konteksten så forklar det. """
 
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()

if "logged_prompt" not in st.session_state:
    st.session_state.logged_prompt = None
if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0

#prompt = st.chat_input('Indtast spørgsmål til ERFA-bladene, sikkerhedsstyrelsens guider eller håndbogen:', )
# if prompt:
#     c = st.container()
#     c.write(prompt)
#     errors = get_moderation(prompt)
#     if errors:
#         st.write(errors)
#     response, sections_tokens = get_response(INSTRUCTIONS, st.session_state.previous, prompt, df, document_embeddings)
#     c.write(response)

#     st.session_state.previous.append((prompt, response))
#     #st.write(df)
# st.write('Stil spørgsmål i søgefeltet i bunden')
# url = "https://forms.office.com/e/dtxKLNNWx8"
# st.write("Du kan komme med feedback [her](%s)" % url)

st.title("💬 FagBotten")
url = "https://forms.office.com/e/dtxKLNNWx8"
url2 = 'https://erfa-nrgi.streamlit.app/'
st.write("Du kan komme med feedback [her](%s)" % url)
st.write("Åben FagBotten i et nyt vindue ved at trykke [her](%s)" % url2)
c = st.container()
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Stil mig gerne et spørgsmål"}]

for msg in st.session_state.messages:
    c.chat_message(msg["role"]).write(msg["content"])

#st.session_state.messages.append({ "role": "system", "content": INSTRUCTIONS })

if prompt := st.chat_input('Indtast spørgsmål her'):

    openai.api_key = st.secrets["apikey"]
    st.session_state.messages.append({"role": "user", "content": prompt})
    c.chat_message("user").write(prompt)
    errors = get_moderation(prompt)
    if errors:
        c.write(errors)
    response, kilder = get_response(prompt, df, document_embeddings)
    msg = response#.choices[0].message
    st.session_state.messages.append(msg)

    lines = []  # Create an empty list to store the strings
    for i in range(kilder.shape[0]):
        url = kilder.iloc[i]['url']
        line = "[" + str(kilder.iloc[i]['Kilde']) + f"]({url}) \n"
        lines.append(line)  # Append the line to the list

    all_lines = '\n'.join(lines)  # Join all the lines into a single string
    c.chat_message("assistant").write(msg.content + '   \n ' + all_lines)#"\n [" + str(kilder.iloc[i]['Kilde']) + "](%s)" % kilder.iloc[i]['url'] )
    


    st.session_state.logged_prompt = collector.log_prompt(
        config_model={"model": COMPLETIONS_MODEL},
        prompt=prompt,
        generation=msg.content,
        session_id=str(st.session_state.session_id),
    )

    ct = str(datetime.datetime.now())
    questions = open("prompts.txt", "a")
    questions.write(ct + ': ' + prompt + " \n")
    questions.close()

    answers = open("responses.txt", "a")
    answers.write(ct + ': ' + msg.content + " \n")
    answers.close()

    conv = open("conversations.txt", "a")
    conv.write(ct + ': ' + prompt + '; ' + msg.content + " \n")
    conv.close()

    user_feedback = collector.st_feedback(
        component="FagBotten Feedback",
        feedback_type="faces",
        open_feedback_label="[Optional] Provide additional feedback",
        model=st.session_state.logged_prompt.config_model.model,
        prompt_id=st.session_state.logged_prompt.id,
        key=st.session_state.feedback_key,
        align="flex-end",
    )
