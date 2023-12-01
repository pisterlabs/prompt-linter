import openai
import os

from dotenv import load_dotenv # Add

load_dotenv() # Add


openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.Completion.create(
  # model="curie:ft-personal-2023-03-15-03-15-09",
  model="davinci:ft-personal-2023-03-15-09-11-14",
  max_tokens=100,
  stop=[".\n"],
  temperature=1,
  # prompt="How can a working station operate again?"
  # "text": " 9Were the harvesters concentrated into the factory first and how do they get"
  # prompt="How to jump to the specified work station?"
  # "text": "\n\nYou can set a jump prompt on the Startup tab of a workstation"
  # prompt="Where can I see the use record of spare parts?"
  # "text": " (Article 29)\n\nhttp://www.waju.or."
  # prompt="Where can I see the use record of spare parts?\nAgent: "
  prompt="How to jump to the specified work station?\nAgent:"
)


"""
"text": "\nFactory infirmary doctor: We don't have any statistics of spare parts and out of hours' use.
 Ours is a new factory. There is no such history.\n
 I was shocked. I nearly exclaimed \"You're not for sale?\" 
 but steadied my thoughts and asked:\nQ4: Where does Dalian FLW produce spare parts for?
 \nFactory infirmary doctor: There is only us. We require nobody else's spare part.\nQ3: How does"
"""


"""
After you’ve fine-tuned a model, 
remember that your prompt has to end with the indicator string `?\nAgent:` for the model to 
start generating completions, rather than continuing with the prompt. 
Make sure to include `stop=[".\n"]` so that the generated texts ends at the expected place.
Once your model starts training, it'll approximately take 2.47 minutes to train a `curie` model, and less for `ada` and `babbage`.
Queue will approximately take half an hour per job ahead of you.
"""

def generate_prompt(animal):
    return """You're a customer service chat bot. \n\nSpecific information: Customers are using our company's platform via web or mobile app.
    
Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )


print(completion)
# print(completion['choices'][0]['message']['content'])
print(completion['choices'][0]['text'])

