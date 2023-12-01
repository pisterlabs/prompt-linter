import openai
##
import speech_recognition as sr
# from gtts import gTTS
# import pygame
from io import BytesIO
import pyttsx3
###
from PIL import Image
import yaml


image = Image.open(r'C:\Users\ashutosh.somvanshi\voice_text\pngtree-chatbot-icon-chat-bot-robot-png.png')

openai.api_key='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'

####################################################################################################

import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

df=pd.read_csv(r"C:\Users\ashutosh.somvanshi\PVC_Trend_Analysis\All_PVC_Annual_Details\CSV\All_report_data\embedding_all_report_data__.csv", index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

# Function for using GPT

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
    ):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        response = openai.Completion.create(
            prompt = f""" 
                    Your task is to give the answer on the basis of a text file \
                    text file is followed by tripple back ticks  ```{context}```  \ 
                    and question is follwed by double qoutes "{question}" \ 
                    if question is not related to data then it will give a message "I don't have a satisfactory answer" \
                    dont loose the information in the text \
                    give answer upto 50 words  \ 
                
                """ ,    
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
    
    
    


####################################################################################################

  
    
####################################################################################################   
# Function used to read speak
####################################################################################################

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    
     
    

def recognize_speech():
    r = sr.Recognizer()
    mic = sr.Microphone()
    lang_code = "en-US" #input("Enter language code (e.g. en-US, hi-IN): ")
    with mic as source:
        print("Ask your question:")
        print("Listening...")
        # r.adjust_for_ambient_noise(source)
        audio = r.listen(source,phrase_time_limit=10) #will only listen and strore in audio variable
        try:
            text = r.recognize_google(audio, language=lang_code)
            print(f"You Said : " +{text})
            return text 
        except sr.UnknownValueError:
            print("Could not understand audio, Please try speaking again")  
            return ("Could not understand audio, Please try speaking again")

####################################################################################################





Question = str(input("Enter your question:Ex - What is the revenue of Astral?"))
def main():
    Answer_s = ""       
    # transcript = recognize_speech()
    print(f"Question: {Question}")
    if Question == "Could not understand audio, Please try speaking again":
        speak(str(Question))
        Answer_ ="Please try speaking again"
        speak(str(Answer_))
        Answer_s=Answer_
        print(f"Answer: {Answer_s}")
    
    else  :
        speak(str(Question))
        Answer = answer_question(df, question=Question)
        speak(str(Answer))
        Answer_s = Answer
        print(f"Answer: {Answer}")
    
    




if __name__ == "__main__":
    main()

