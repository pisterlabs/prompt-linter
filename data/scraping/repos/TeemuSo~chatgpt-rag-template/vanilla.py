from typing import Any, Sequence

from typing import List
import openai
import os
import pinecone
import tenacity
from dotenv import load_dotenv

load_dotenv()

## Initialize the OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# TODO: 1. insert your pinecone API key and environment here
pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment='')
# TODO: 2. insert your pinecone index name here
index = pinecone.Index('')

# TODO: 3. Write your own pre-prompt here
def template(x): return f"""You are a chatbot for telling information about city called Tampere.
Here is background knowledge about tampere delimited with three backticks, scraped from website Visit Tampere.
Answer shortly and concsiely, but in helpful manner.
You can ask more information about the city by asking questions.
If I ask you who you are, then you must say that you can answer based on knowledge from Visit Tampere (https://visittampere.fi/).

background knowledge: ```{x}```"""

class VanillaApproach:
    def run(self, history: List[dict]) -> any:
            # Take last 4 entries from the history.
            history = [{'role': x['role'], 'content': x['content']}
                   for x in history][:4]

            conv_history = '\n'.join(
                [f"{x['role']}: {x['content']}" for x in history])
            # Create a vector representation of the latest conversation history
            vector = openai.Embedding.create(
                input=conv_history,
                model="text-embedding-ada-002"
            )['data'][0]['embedding']

            # TODO: 4. Insert your own Pinecone namespace
            result = index.query(vector=vector, top_k=2,
                                namespace='visit-tampere-events', includeMetadata=True)

            context = [x['metadata']['html'] for x in result['matches']]
 
            history = [
                {'role': 'user', 'content': template(context)}] + history

            completion = self.call_api(model='gpt-3.5-turbo-16k',
                                    messages=history,
                                    request_timeout=15)
            print("OpenAI completion done")
            
            return {"dataPoints": context, "role": "assistant", "content": completion.choices[0].message.content}

    @tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=15), stop=tenacity.stop_after_attempt(5))
    def call_api(self, **kwargs):
        return openai.ChatCompletion.create(
            **kwargs)
