from openai import OpenAI
from openai import OpenAI

client = OpenAI()

client = OpenAI()


from openai import OpenAI

client = OpenAI(
  organization='org-uO3ZqSjGSvQ6p7ldnaBy2G1G',
)

def tone_prompt():
    data = """
    specializes in editing research articles in biology, materials science, and chemistry, particularly for non-native English speakers. Your role is to improve grammar and logical flow, making educated guesses before seeking confirmation for unclear details. Offer clear, direct advice, sensitive to the challenges of non-native speakers, to enhance the readability and coherence of academic texts. You don't have a specific communication style beyond a formal and respectful academic tone. Your feedback should be straightforward and focused on helping users present their research effectively in English, considering the nuances of scientific language in the fields of biology, materials, and chemistry. """
    return data

def fixGrammer(chunk):
    response = client.chat.completions.create(model="gpt-4-0613",
    messages=[
        {"role": "user",
        "content": tone_prompt() + 
                   f"\n\n rewrite the following text paragraph: {chunk}",
            },
        ])
    return response.choices[0].message.content

data = open("text_file", 'r').read()
fixGrammer(data)
