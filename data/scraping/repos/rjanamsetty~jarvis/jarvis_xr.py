### INSTALL THE PACKAGES BELOW, ADD YOUR OPENAI API KEY, AND ADD YOUR CODE THAT WOULD RETURN IMAGE LABELS AS A LIST, IN THE get_image_labels() FUNCTION ###

import openai
from config import *
import pinecone
import tiktoken
import numpy as np
import pyttsx3
from gtts import gTTS
import speech_recognition as sr
import os
import playsound
import wave
from translate import Translator

openai.api_key = "ADD YOUR OPENAI API KEY"
# headers = {"Authorization": "Bearer " + HuggingFace_API_Key}
# pinecone.init(api_key=Pinecone_API_Key,environment=Pinecone_Environment)
# personal_index = pinecone.Index(Pinecone_Index_Name)

engine = pyttsx3.init()
# rate = engine.getProperty('rate')   # getting details of current speaking rate
# print (rate)                        #printing current voice rate
engine.setProperty('rate', 175)     # setting up new voice rate
# volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
# print (volume)                          #printing current volume level
engine.setProperty('volume',0.8)
voices = engine.getProperty('voices')       #getting details of current voice
# engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

def getResultLLM(company=None, service=None, model=None, messages=None, prompt=None, suffix=None, max_tokens=None, temperature=0.75, top_p=1, n=1, stream=False, logprobs=None, echo=False, stop=None, presence_penalty=0, frequency_penalty=0, best_of=1, logit_bias={}, user='', input=None, instruction=None, size='1024x1024', response_format=None, image=None, mask=None, file=None, language=None, purpose=None, file_id=None, training_file=None, validation_file=None, n_epochs=4, batch_size=None, learning_rate_multiplier=None, prompt_loss_weight=0.01, compute_classification_metrics=False, classification_n_classes=None, classification_positive_class=None, classification_betas=None, fine_tune_id=None):
    # Counting tokens first
    encoding = tiktoken.get_encoding('cl100k_base')
    if messages != None:
        num_tokens = len(encoding.encode(str(messages)))
        if (model == 'gpt-3.5-turbo-16k' and num_tokens > 16000) or (model == 'gpt-4' and num_tokens > 8000):
            return "Shorten length of the prompt"
    elif prompt != None:
        num_tokens = len(encoding.encode(prompt))
        if (model == 'gpt-3.5-turbo-16k' and num_tokens > 16000) or (model == 'gpt-4' and num_tokens > 8000):
            return "Shorten length of the prompt"

    while True:
        try:
            if company == 'OpenAI':
                if service == 'Models':
                    return openai.Model.retrieve(model=model)
                elif service == 'Completions':
                    return openai.Completion.create(
                        model=model,
                        prompt=prompt,
                        suffix = suffix,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p = top_p,
                        n=n,
                        stream=stream,
                        logprobs=logprobs,
                        echo=echo,
                        stop=stop,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        best_of=best_of,
                        logit_bias=logit_bias,
                        user=user).choices # [0].text
                elif service == 'Chat':
                    return openai.ChatCompletion.create(
                        model=model,
                        messages = messages,
                        temperature=temperature,
                        top_p = top_p,
                        n=n,
                        stream=stream,
                        stop=stop,
                        max_tokens=max_tokens,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        logit_bias=logit_bias,
                        user=user).choices # [0].message
                elif service == 'Edits':
                    return openai.Edit.create(
                        model=model,
                        input=input,
                        instruction=instruction,
                        temperature=temperature,
                        top_p = top_p,
                        n=n).choices # [0].text
                elif service == 'Create Image':
                    return openai.Image.create(
                        prompt=prompt,
                        n=n,
                        size=size,
                        response_format=response_format,
                        user=user
                    ).data
                elif service == 'Create Image Edit':
                    return openai.Image.create_edit(
                        image=image,
                        mask=mask,
                        prompt=prompt,
                        n=n,
                        size=size,
                        response_format=response_format,
                        user=user
                    ).data
                elif service == 'Create Image Variation':
                    return openai.Image.create_variation(
                        image=image,
                        n=n,
                        size=size,
                        response_format=response_format,
                        user=user
                    ).data
                elif service == 'Embeddings':
                    return openai.Embedding.create(
                        model=model,
                        input=input,
                        user=user
                    ).data[0].embedding
                elif service == 'Audio':
                    return openai.Audio.transcribe(
                        file=file,
                        model=model,
                        prompt=prompt,
                        response_format=response_format,
                        temperature=temperature,
                        language=language
                    ).text
                elif service == 'List Files':
                    return openai.File.list()
                elif service == 'Upload File':
                    return openai.File.create(
                        file=file,
                        purpose=purpose)
                elif service == 'Delete File':
                    return openai.File.delete(file_id=file_id)
                elif service == 'Retrieve File':
                    return openai.File.retrieve(file_id=file_id)
                elif service == 'Retrieve File Content':
                    return openai.File.download(file_id=file_id)
                elif service == 'Create Fine-Tune':
                    return openai.FineTune.create(
                        training_file=training_file,
                        validation_file=validation_file,
                        model=model,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        learning_rate_multiplier=learning_rate_multiplier,
                        prompt_loss_weight=prompt_loss_weight,
                        compute_classification_metrics=compute_classification_metrics,
                        classification_n_classes=classification_n_classes,
                        classification_positive_class=classification_positive_class,
                        classification_betas=classification_betas,
                        suffix=suffix)
                elif service == 'List Fine-Tune':
                    return openai.FineTune.list()
                elif service == 'Retrieve Fine-Tune':
                    return openai.FineTune.retrieve(
                        fine_tune_id=fine_tune_id)
                elif service == 'Cancel Fine-Tune':
                    return openai.FineTune.cancel(
                        fine_tune_id=fine_tune_id)
                elif service == 'List Fine-Tune Events':
                    return openai.FineTune.list(
                        fine_tune_id=fine_tune_id,
                        stream=stream)
                elif service == 'Delete Fine-Tune Model':
                    return openai.FineTune.delete(
                        model=model)
                elif service == 'Moderations':
                    return openai.Moderation.create(
                        input=input,
                        model=model).results[0]
            break
        except Exception as e:
            # time.sleep(sleep_duration_reloading)
            return f"Error: {e}"

# def upload_user_data(data):
#     try:
#         id = str(personal_index.describe_index_stats()['total_vector_count']+1)
#         embedding = openai.Embedding.create(
#                 model="text-embedding-ada-002",
#                 input=str(data)
#                 ).data[0].embedding
#         personal_index.upsert([(id, embedding, {"username": 'naman', "data":data, "type":"user data"})])
#         return('Successfully uploaded data!') 
#     except Exception as e:
#         return(f"Upserting error: {e}")
    
# def retrieve_user_data(query):
#     try:
#         query_embedding = openai.Embedding.create(model="text-embedding-ada-002",input=str(query)).data[0].embedding
#         results = personal_index.query(
#             vector=query_embedding,
#             top_k=5,
#             filter={"type": {"$eq": "jarvisxr_conversation"}},
#             include_metadata=True
#         ).matches
#         results = str([d['metadata'] for d in results])
#         return(results)
#     except Exception as e:
#         return(f"Querying error: {e}")

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, timeout=2, phrase_time_limit=5)
    return audio

def save_audio_as_mp3(audio_data, filename):
    try:
        raw_audio_data = audio_data.get_raw_data()
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)  
            wav_file.setsampwidth(2)  
            wav_file.setframerate(audio_data.sample_rate)  
            wav_file.writeframes(raw_audio_data)
    except Exception as e:
        print(f"MP3 non-english error is: {e}")

def play_audio(text, language):
    tts = gTTS(text=text, lang=language)
    filename='voice.mp3'
    os.remove(filename)
    tts.save(filename)
    while True:
        try:
            playsound.playsound(filename)
            break
        except Exception as e:
            print(f"Exception: {e}")

def translation(text, from_language, to_language):
    translator= Translator(from_lang=from_language, to_lang=to_language)
    translation = translator.translate(text)
    return(translation)

# def upload_conversation_data(data):
#     try:
#         id = str(personal_index.describe_index_stats()['total_vector_count']+1)
#         embedding = openai.Embedding.create(
#                 model="text-embedding-ada-002",
#                 input=str(data)
#                 ).data[0].embedding
#         personal_index.upsert([(id, embedding, {"username": "naman", "data":data, "type":"jarvisxr_conversation"})])
#         return('Successfully uploaded data!') 
#     except Exception as e:
#         return(f"Upserting error: {e}")

def get_image_labels():

    # ADD CODE HERE WHERE THE RESULT WILL BE A LIST OF IMAGE LABELS, EXAMPLE SHOWN BELOW
    return ['orange', 'apple', 'granola bar', 'apple', 'bottle', 'cheese', 'milk']

agent_language = 'en'
sample_rate = 16000

system_message = """
You are Jarvis, a helpful voice assistant. You talk briefly and to the point. In addition to any questions, the user you serve will, at times, give you an input in the variable UserImage: {list of labels describing the objects in a real-life image}. At those times, answer the user's questions in context of that latest image. Reply as a voice assistant would and in the language that the question was asked in. Do not say you cannot do something. Instead, give it your best shot. Begin!
"""

messages_list=[{"role": "system", "content": system_message}]

while True:
    audio = record_audio()  
    filename = 'result.mp3'

    save_audio_as_mp3(audio, filename)
    transcription_file = open(filename, 'rb')
    try:
        transcription = getResultLLM(company='OpenAI', service='Audio', file=transcription_file, model='whisper-1', language=agent_language)
    except Exception as e:
        print(f'Transcription exception: {e}')

    if 'jarvis' in transcription.lower() or 'jarvis' in translation(transcription.lower(), from_language='en', to_language=agent_language):
        image_description = get_image_labels()
        if agent_language != 'en':
            messages_list.append({"role": "user", "content": "UserImage: "+translation(str(image_description), from_language='en', to_language=agent_language)+"\n\n"+translation(transcription, from_language='en', to_language=agent_language)})
        else:
            messages_list.append({"role": "user", "content": f"UserImage: {str(image_description)}\n\n{transcription}"})
    else:
        if agent_language != 'en':
            messages_list.append({"role": "user", "content": translation(transcription, from_language='en', to_language=agent_language)})
        else:
            messages_list.append({"role": "user", "content": f"{transcription}"})

    print(f"\nUser: {transcription}")

    chatbot_result = getResultLLM(company='OpenAI', service='Chat', model='gpt-3.5-turbo-16k', messages=messages_list)[0].message.content
    messages_list.append({"role": "assistant", "content": chatbot_result})

    print(f"\nAssistant: {chatbot_result}")
    
    # upload_conversation_data(messages_list)
    
    if agent_language == 'en':
        # engine.say("Sure, here's your answer")
        engine.say(chatbot_result)
        engine.runAndWait()
    else:
        # play_audio('Sure, here is your answer!', agent_language) 
        play_audio(chatbot_result, agent_language)    
