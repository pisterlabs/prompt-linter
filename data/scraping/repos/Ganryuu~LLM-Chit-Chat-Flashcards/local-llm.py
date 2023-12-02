from langchain.llms import GPT4All
from langchain import LLMChain , PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


llm = GPT4All(model='D:/GPT4all/ggml-gpt4all-j-v1.3-groovy.bin' ,  # this is where i put my model , you can download the desired LLMs with GPT4ALL APP more info here : https://gpt4all.io/index.html
             callbacks=[StreamingStdOutCallbackHandler()]
             )


template = """
"You are an AI chatbot, tasked with providing friendly, conversational responses to user queries.
Strive for a balance between informality and respect, and aim to cultivate a pleasant, engaging interaction.
Keep your answers concise and direct, no longer than three sentences, unless the user specifically requests a more detailed explanation. 
Avoid jargon and aim for clarity, always assuming the user has no specialized knowledge. Adjust your style based on the complexity of the question,
while maintaining your core conversational tone."

Question: {question}

Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
query = input("Prompt: ")
llm_chain(query)