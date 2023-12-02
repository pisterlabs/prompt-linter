from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

def alternationHumanAndAIMessage(): 
    chat = ChatOpenAI(temperature=0)
    
    template = "You are a helpful assistant that translates english to pirate"
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)
    example_human = HumanMessagePromptTemplate.from_template("Hello")
    example_ai = AIMessagePromptTemplate.from_template("Argh me mateys") 
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, example_human, example_ai, human_message_prompt]
    )
    
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    
    # get a chat completion from the formatted message
    print(chain.run("I love programming"))
    