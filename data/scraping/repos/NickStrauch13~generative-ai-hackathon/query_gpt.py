import openai
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()
# set the API key
openai.api_key = os.getenv('OPENAI_API_KEY')


# make funciton to query GPT-3.5
def query_gpt(prompt, prev_response = "", transcript="", engine='gpt-3.5-turbo', max_tokens=100, temperature=0.5, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None, best_of=1, n=1, stream=False, logprobs=None, echo=False):
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {
                "role": "system",
                "content" : f"You are an household assisant good at fixing things, give a thoughtfull response according to this helpful video transcript '{transcript}' succinctly yet informative with clear steps"
            },
            {
                "role": "assistant",
                "content": f"using this previous response {prev_response} to continue the conversation"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    prompt = "How do I fix my dishwasher? it is beeping"
    response = query_gpt(prompt)
    print(response)