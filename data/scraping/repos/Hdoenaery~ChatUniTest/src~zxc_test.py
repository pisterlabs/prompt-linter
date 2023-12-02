import openai
import os


#openai.api_key = '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'
openai.api_key = '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'
def chat_gpt(prompt):
     prompt = prompt
     model_engine = "text-davinci-003"
     completion = openai.Completion.create(
         engine=model_engine,
         prompt=prompt,
         max_tokens=1024,
         n=1,
         stop=None,
         temperature=0.5,
         timeout=1000,
     )
 
     response = completion.choices[0].text
     print(response)

chat_gpt("现在几点了")
