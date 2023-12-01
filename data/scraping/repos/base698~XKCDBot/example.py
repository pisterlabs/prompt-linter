import os
import openai
import sys
import json

from openai.embeddings_utils import cosine_similarity

openai.api_key = os.environ.get('OPENAI_KEY')

EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embeddings(transcript_str):
    embeddings = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=transcript_str
    )
    return embeddings

compiling_embed = get_embeddings("""Title: The #1 Programmer Excuse for Legitimately Slacking Off:My code's compiling.Two programmers are sword-fighting on office chairs in a hallway. An unseen manager calls them back to work through an open office door. Manager: Hey! Get back to work! Programmer 1: Compiling! Manager: Oh. Carry on. Are you stealing those LCDs? Yeah, but I'm doing it while my code compiles.""")

output = {}

def main():
   phrases = [
     'we are playing games and not working.',
     'my code is compiling',
     'I work baking cakes',
     'i really wish I could get back to work.',
     'I hope my boss is not going to see us leave.',
     'Sally and Jim are sitting in the park.',
     'Big red dog sitting and relaxing on the beach',
     'Jim is not wasting time, he is always working hard.']
   for p in phrases:
      embed = get_embeddings(p)
      cs = cosine_similarity(embed['data'][0]['embedding'],compiling_embed['data'][0]['embedding'])
      
      output[p] = cs
    
   with open('output.json','w') as f:
       f.write(json.dumps(output))

if __name__ == '__main__':
   main()
