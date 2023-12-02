import cohere
import os
from dotenv import load_dotenv
load_dotenv()
co = cohere.Client(os.environ.get('cohere_key'))
big_prompt = f"""Summarize the following text into a suitable text to image ai prompt

I want someone to talk to
I am always here to listen to your worries!
I am having friendship troubles
Could you please describe your problem a bit more? 
I am feeling useless
As I have gotten to know you, I can guarantee you that you are important and unique
My best friend stopped talking to me a few days ago and I am panicking
First take a deep breath, and then maybe once you're feeling better, try your best to talk over your problems with your friend

Summary: "Friendship troubles, feeling useless, best friend stopped talking and causing panic"
-- 
I am feeling good today for a change!
That's awesome, I'm glad to know your feeling better about yourself!
I want to actually go out today and enjoy myself
I completely think you should. It'd be helpful to take a breath of fresh air and enjoy nature
I am feeling down
There are many reasons to be happy! Is there anything I can do to help?

Summary: "Feeling good, wanting to go out and enjoy, feeling down"
"""

def summarize(prompt):
    initial_prompt = big_prompt
    response = co.generate(
    model='command-xlarge-nightly',
    prompt=initial_prompt + prompt,
    max_tokens=30,
    temperature=0.8,
    stop_sequences=["--"],
    return_likelihoods='NONE')
    print('*****', response.generations[0].text, '*****')
    text = response.generations[0].text
    if '"' in text:
        text = text[:text.rindex('"')]
    return "Create a hopeful image about " + text