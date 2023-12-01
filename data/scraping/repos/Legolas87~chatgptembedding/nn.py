# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

from IPython.display import display


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

openai.api_key = "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"

AcuAbout = "list of people working in Acumen:  David Avetisyan Co-Founder and the director his favourite smiley in Slack is :ghost: , Armen Khachatryan - software developer likes playing fooball,  Marlena Mirzoyan - Project Manager enjoys playing a game 'chto gde kogda', Anna Snkhchyan - .net developer does great with one of the big customers of Acumen: Cinchy,  Sophie Mehrabyan Co-Founder she lives in Barcelona, Suren Mardanyan developer he likes playing football, Hripsime Manukyan react developer"

query = f"""Use the below information about Acumen as a software development company to answer the subsequent question. If the answer cannot be found, write "I don't know."

Information:
\"\"\"
{AcuAbout}
\"\"\"

Question: anyone living in europe?"""

response = openai.ChatCompletion.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about Acumen'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response['choices'][0]['message']['content'])