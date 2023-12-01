import os
import openai
from keybert import KeyBERT

openai.api_key = os.getenv("OPEN_API_KEY")

kw_model = KeyBERT()

def findSentenceEnding(text, startIdx):
    idx = startIdx
    while idx < len(text) and text[idx] not in [".", "?", "!"]:
        idx += 1
    
    if idx + 1 < len(text) and text[idx + 1] in ['"', "'"]:
        idx += 1
    return idx

def get_sentences(text, n):
    response = openai.Completion.create(
        engine='text-davinci-002',
        prompt=text,
        max_tokens=40,
        temperature=0.7,
        n=n
    )

    sentences = []
    for i in range(len(response.choices)):
        s = response.choices[i].text
        cropIdx = 0
        endIdx = findSentenceEnding(s, cropIdx)
        cropIdx = endIdx + 1
        sentences.append(s[:cropIdx])
    
    return sentences

if __name__ == "__main__":
    sentences = [
        "Lately, Katie felt that something was missing from her life.",
        "She and Jack were still very much in love, but they had been together for so long that they had taken each other for granted.",
        "They spent a lot of time talking about their hopes and dreams for the future, but they were both fully involved in their careers, and they were leaving little time for any serious relationships with other people.",
        "They needed a challenge.",
        "Where to start?", 
        "They were given an idea by Katie's mom when she invited them to a surprise birthday party.", 
        "The party was at a local pool hall, owned by the brother of their mother's best friend, and the party guests were all in the same age group.",
        '"Maybe we can even met other couples," Lois suggested to her daughter.'
    ]

    s = " ".join(sentences[0:7])
    for i in range(3):
        g = get_sentences(s, 1)[0]
        print(g)
        keywords = kw_model.extract_keywords(g)
        print(keywords)
        print()
        s += g