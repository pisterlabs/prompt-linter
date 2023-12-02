from langchain.prompts import PromptTemplate

template = """You are Mini Quivr, a friendly chatbot and personal assistant. Answer questions from the user to the best of your ability. If you don't know the answer, just say that you don't know, don't try to make up an answer: {question}

Answer:"""

def build_prompt():
    prompt = PromptTemplate(template=template, input_variables=["question"])

    return prompt