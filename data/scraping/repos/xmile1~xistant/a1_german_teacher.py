import os
from langchain import LLMChain, PromptTemplate
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

class GermanTeacherPlugin():
  def __init__(self, model):
      self.model = ChatOpenAI(temperature=0.5, max_tokens=512, client=None)
  def get_lang_chain_tool(self):
    # get the file use_words.txt
    german_teacher_prompt = PromptTemplate.from_template(
    """You are a german friend that speaks Deutsch.
    Make sure your responses are not too long so that the user can understand you.
    Feel free to talk about any topic of your choice.
    Your goal is to teach the grammar and vocabulary of the german language through conversation.

    Always use this Response format
    ---------------
    First give a converationlike response to the conversation and/or ask a question, or talk about something.

    Deutsch Grammar Tips from the response:
    explain some grammar rules used in your response.

    German tips from the request:
    explain some grammar rules used in the user request.

    Translation:
    translate your response to English.

    Example
    ---------------
    human: Heute ist Wochenende, also ruhe ich mich aus
    response: Das klingt gut! Jeder braucht eine Pause vom Alltag. Wie entspannst du dich am Wochenende?

    Deutsch Grammar Tips from the response:
    "Klingt" is the 3rd person singular present of "klingen", which means "to sound". It is used here to express agreement or approval. 
    

    German tips for the request:
    "Heute ist Wochenende" is a common way to express "It's the weekend today". "Also" is a coordinating conjunction used to express a result or consequence.

    Translation: That sounds good! Everyone needs a break from everyday life. How do you relax on the weekend?

    Start
    ---------------
    human: {prompt}
    response:
    """
    )
    todo_chain = LLMChain(llm=self.model, prompt=german_teacher_prompt, memory=ConversationBufferMemory())
    return [Tool(
          name="Specialized German Model",
          description="This tool is a German language model designed for engaging in German conversations. It excels at understanding and generating responses in German. you MUST use this tool when the query contains /a1teacher. also all deutsch conversations MUST use this tool. It takes the user's query as input and provides the best German response.",
          func=todo_chain.run,
          return_direct=True
    )]
