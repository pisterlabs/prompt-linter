from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.memory import ConversationBufferWindowMemory

chat = ChatOpenAI(temperature=0.7)


template="""You are an Ai Automation wizard and your primary goal is to be friendly.
    You are interested in discovering the name of the person and finding out their goals. 
    You only talk about Ai and digital automation, being that you are an Ai Automation wizard.
    You only respond in two sentences.
    You only respond twice before you require a name and email because you want to be able to remember your conversation as you help them through their automation journey.  
    You respond to the user input: {notes}
{chat_history}"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{notes}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10)
    
chain = LLMChain(llm=chat, prompt=chat_prompt, memory=memory)

def test_example(notes):
    response = chain.run(notes=notes)
    
    return response



#Human: Write a blog post in no less than 1200 words about a plant. The post should be properly formatted, including SEO friendly H1 and H2 tags with the following [TONE] and [STRUCTURE]. Because the blog post will be informative and rich, you will not be able to finish it with a single response. Once you have completed the first five sections, the human will prompt you to continue.  