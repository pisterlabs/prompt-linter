import modal
import os
from fastapi import FastAPI, UploadFile, File, HTTPException

stub = modal.Stub("sd", image=modal.Image.debian_slim().pip_install("openai~=0.27.0"))

@stub.function(secrets=[modal.Secret.from_name("my-openai-secret"), modal.Secret.from_name("my-openai-secret-2")])
def semantic_detection(labels: list):
    import openai
    def GPT_Completion_raw(texts):
        openai.api_key = os.environ["OPENAI_API_KEY_2"]
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt =  texts,
            temperature = 0.6,
            top_p = 1,
            max_tokens = 64,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        return response.choices[0].text

    def get_species(labels):
        species = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""If the term does not describe a dog or dog breed, eliminate it. From the remaining list, 
  if there is a specific dog breed in the list, tell me the dog breed. If there are more than two items remaining in the list, 
  pick the most specific dog breed and return that. Keep your answer between 1 and 3 words. Here is the list: {labels}"""}
            ]
        )
        return species["choices"][0]["message"]["content"]
        
    def predict_size(species):
        gpt_size = 0
        fur_output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant and an expert on dogs."},
                {"role": "user", "content": f"How fluffy is the fur of {species} compared to the average dog? One word response: less or more"}
            ]
        )
        size_output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant and an expert on dogs."},
                {"role": "user", "content": f"What is the size of an average {species} compared to the average dog? One word response: big or small"}
            ]
        )
        fur_output = fur_output["choices"][0]["message"]["content"].lower()
        size_output = size_output["choices"][0]["message"]["content"].lower()
        
        if (size_output.__contains__('big') and fur_output.__contains__('less')):
            gpt_size = 1
        elif (size_output.__contains__('big') and fur_output.__contains__('more')):
            gpt_size = 2
        elif (size_output.__contains__('small') and fur_output.__contains__('less')):
            gpt_size = 3
        elif (size_output.__contains__('small') and fur_output.__contains__('more')):
            gpt_size = 4
        else:
            print("sad")
        
        return gpt_size
    
    def predict_weight(species):
        gpt_weight = 0
        voice_output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant and an expert on dogs."},
                {"role": "user", "content": f"What is the average weight of {species}. ? One whole number as a response ONLY, no words"}
            ]
        )
        weight_int = int(voice_output["choices"][0]["message"]["content"].lower())
        if (weight_int <= 10):
            gpt_weight = 4
        elif (weight_int > 10 and weight_int <= 40):
            gpt_weight = 3
        elif (weight_int > 40 and weight_int <= 100):
            gpt_weight = 2
        elif (weight_int > 100):
            gpt_weight = 1
        else:
            print("sad")
        return gpt_weight
    
    def predict_full(species):
#         voice_output = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant and an expert on dogs. Your only valid response to the user's question is a single integer that should be castable to an int in python. If you do not know what to predict, just predict 4. After you decide what number you want to give me as a prediction, you should give me a one word response that describes the pitch of the voice of the dog. The pitch of the voice of the dog should be one of the following: 1, 2, 3, or 4. 1 is the lowest pitched voice, and 4 is the highest pitched voice. If you do not know what to predict, just predict 4."""},
#                 {"role": "user", "content": f"""Establish an average Labrador Retriever having a voice pitch of 2, an average German Shepherd having a voice pitch of 1, 
#   an average Toy Dog having a voice pitch of 3, an average Poodle having a voice pitch of 3, an average Chihuahua having a voice pitch of 4, 
#   and an average Husky having a voice pitch of 2. Output the result as 1, 2, 3, or 4, with 1 being lowest pitched and 4 being highest pitched.
#   What pitch of voice would an {species} have? Give just one number as the answer. Again, make sure that your response is purely one number. Imagine trying to parse your response as an integer; if that would raise an error, you have given me the wrong response!"""}
#             ]
#         )
#         return int(voice_output["choices"][0]["message"]["content"].lower())
        v_out = GPT_Completion_raw(""" Establish an average Labrador Retriever having a voice pitch of 2, an average German Shepherd having a voice pitch of 1, 
  an average Toy Dog having a voice pitch of 3, an average Poodle having a voice pitch of 3, an average Chihuahua having a voice pitch of 4, 
  and an average Husky having a voice pitch of 2. Output the result as 1, 2, 3, or 4, with 1 being lowest pitched and 4 being highest pitched. """ 
  "What pitch of voice would an average " + species + " have? Give just one number as the answer." )
        return int(v_out)
    
    #picks out species labels from those generated by object detection
    species = get_species(labels)
    #prediction based on size and weight priors
    voice_pred1 = (predict_size(species) + predict_weight(species) + 1) // 2
    #prediction based purely on chatgpt output
    voice_pred2 = predict_full(species)
    #we are going to average the two predictions to get a more accurate result
    voice_pred = (voice_pred1 + voice_pred2) // 2
    return voice_pred

