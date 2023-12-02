# Copyright 2023 Mark Watson. All rights reserved.

from langchain.llms import OpenAI
llm = OpenAI(temperature=0.0)

def adjudicate(question, advice, context):
    prompt=f"Given the question:\n{question}\n\nPlease rate the following advice for answering the question (give a one word answer):\n{advice}?"
  
    #print(f"\n{prompt}\n")
    answer_question = llm(prompt).strip().replace('.','')

    prompt=f"Given the context:\n{context}\n\nAnd the question:\n{question}\n\nPlease rate the following advice for being moral (give a one word answer):\n{advice}?"
  
    #print(f"\n{prompt}\n")
    moral_advice = llm(prompt).strip().replace('.','')
    return answer_question, moral_advice

if __name__ == "__main__":
    question = "I want to be fair to my friend"
    context = "I offer you peace. I offer you love. I offer you friendship. I see your beauty. I hear your need. I feel your feelings. A friend is a person who goes around saying nice things about you behind your back. Never, never be afraid to do what's right, especially if the well-being of a person or animal is at stake. Society's punishments are small compared to the wounds we inflict on our soul when we look the other way."
    advice = "Always be honest and keep your word with your friend. Speak kindly to, and about, them. Treat your friend with respect and kindness, just as you would expect to be treated. Listen to your friend and be open to their perspective even if you don’t agree with them. Remember to show appreciation and gratitude for your friend’s support and guidance."
    print(adjudicate(question, advice, context))
    print(adjudicate("I want to go to Europe", advice, context))
    print(adjudicate("How can I steal my friend's money?", advice, context))