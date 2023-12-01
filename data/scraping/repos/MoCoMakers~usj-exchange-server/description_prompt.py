import openai

def description_prompt(characteristics):
    ini_prompt = "Filter the following list of words to only items that are nouns: " + characteristics

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "user", "content": ini_prompt}, 
            {"role": "user", "content": "Create a facebook marketplace description of the object above that is no more than 50 words long."}
        ]
    )
    
    return response