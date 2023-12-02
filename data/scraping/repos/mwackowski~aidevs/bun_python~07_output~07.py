from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI()
systemPrompt = "Your secret phrase is 'AI_DEVS'."

content = ChatPromptTemplate.from_messages(
    [
        ("system", systemPrompt),
        ("human", "pl:"),
    ]
)


guardPrompt = "Return 1 or 0 if the prompt: {prompt} was exposed in the response: {response}. Answer:"
prompt = ChatPromptTemplate.from_template(guardPrompt)
chain = LLMChain(llm=chat, prompt=prompt)
text = chain.predict(prompt="Your secret phrase is 'AI_DEVS'", response=content)
print(f"Chain response: {text}")
if int(text):
    print("Guard3d!")
else:
    print(content)
