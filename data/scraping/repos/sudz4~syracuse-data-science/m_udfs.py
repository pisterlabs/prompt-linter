"""
Batch API calls

https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks

"""
def batcher(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n] # n = the length of chunks / batches
        

"""
Codex robot_analyst() user defined function.

This function is used during development to help the programmer
learn about and understand technical stock terms.
"""
import openai
from m_config import OPENAI_API_KEY

def robot_analyst():
    openai.api_key = OPENAI_API_KEY
    answer_data = openai.Completion.create(
        model="text-davinci-002",
        prompt=input('Question: -> '),
        temperature=0.3,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0  
    )
    answer = answer_data['choices']
    
    for ans in answer:
        final_answer = ans['text']
        print(final_answer)
        