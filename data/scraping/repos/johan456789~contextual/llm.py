import anthropic
import os
from dotenv import load_dotenv
load_dotenv()
ANTHROPIC_API = os.getenv("ANTHROPIC_API")

client = anthropic.Client(ANTHROPIC_API) # type: ignore

max_tokens_to_sample = 100
def define(word, sentence, model='claude-v1'):
    prompt = f'''\
You are an ESL dictionary. You show the definition of the word in its context. You output the definition and nothing else. Do not repeat the word nor the context. Your definition is not in quotes. Remember to check if your answer has the matching part of speech in the context. Avoid giving one-word answers and use multiple words to define the word.

Word: tender
Context: The national currency is legal tender in practically every country.
Definition: n. something that may be offered in payment
Word: gave
Context: He gave her his hand
Definition: v. place into the hands or custody of
Word: contextual
Context: he included contextual information in footnotes
Definition: adj. depending on or relating to the circumstances that form the setting for an event, statement, or idea.

Word: {word}
Context: {sentence}
Definition:
            '''
    resp = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model,
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=0,
    )
    return resp['completion']
