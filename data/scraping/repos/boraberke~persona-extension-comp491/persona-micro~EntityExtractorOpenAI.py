import os
import openai
import json



#from dotenv import load_dotenv
#load_dotenv()

openai.api_key = os.getenv('OPEN_API_SECRET_KEY')


def generate_entities(text):
  
  search_text = f'Process the following sentence: "{text}"'
  the_rest =  ' Give the results in a JSON format with the following string keys with double quotation marks ("): "Music Interests", "Movie Interests", "Sport Interests", "Hobby", "Nationality", "Product", "Location", "Profession" If you cannot find a value for key, set value to be "unknown" as string.'
  search_text += the_rest 
  #search_text = "Process the following sentence: \"{input_text}\" Give the results in the following format: {\"Music Interests\": \"Movie Interests\": \"Sport Interests\": \"Hobby\": \"Nationality\": \"Product\": \"Gender\": \"Age\": \"Name\": \"Location\": \"Profession\":} If you can't find a value for key, set value to be \"unknown\" as string.".format(input_text = text)
  #search_text = "Give interests, nationality, gender, age and the name of the person as json that says : \"{input_text}\" On your output, don't have any comment in string. If you can't find a value for key, set value to be unknown as string".format(input_text = text)
  print(search_text)
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt= search_text,
    temperature=0.6,
    max_tokens=150,
    top_p=1,
    frequency_penalty=1,
    presence_penalty=1
  )
  print("response",response["choices"][0]["text"])

  response_fix = response["choices"][0]["text"].replace('“','"')
  response = response_fix.replace('”','"')
  response = response_fix.replace('\'','"')

  print("response fix",response_fix)
    
  return json.loads(response_fix)

#text ="I am Emre a football player lives in London. I am interested in jazz."
#response = generate_entities(text)
#print(response)
#json_text = """{
#  "name": "Emre",
#  "nationality": "unknown",
#  "gender": "unknown",  
#  "age": "null",  
#  "interests": ["football","jazz"],  
#  "location":"London"   
#}"""
#print(json.loads(json_text))

# add if __name__: