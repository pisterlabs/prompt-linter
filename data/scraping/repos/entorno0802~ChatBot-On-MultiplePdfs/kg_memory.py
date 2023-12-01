from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0)
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)
conversation_with_kg = ConversationChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=ConversationKGMemory(llm=llm)
)

conversation_with_kg.predict(
    input="Hi, My name is James and I'm a football player, what's up?")
conversation_with_kg.predict(
    input=" I'm helping Will. He's an engineer. And Susan is my girlfriend and she's a student at colleage")
print(conversation_with_kg.predict(
    input="What's relationship between Susan and James?"))
