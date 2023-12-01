import guidance
import os
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def yesNoToTrueFalse(text: str) -> str:
  if "yes" in text.lower():
    return True
  return False

highLevelProgram = guidance(
"""
{{#system~}}
You should answer all questions solely based on the following message: "{{message}}"
{{~/system}}

{{#user~}}
Is the message a question? Answer simply "yes" or "no"
{{~/user}}
{{#assistant~}}
{{gen "isQuestion"}}
{{~/assistant}}

{{#user~}}
Does this message contain a doubt? Answer simply "yes" or "no"
{{~/user}}
{{#assistant~}}
{{gen "isDoubt"}}
{{~/assistant}}

{{#user~}}
Is the message offensive? Answer simply "yes" or "no"
{{~/user}}
{{#assistant~}}
{{gen "isOffensive"}}
{{~/assistant}}

{{#user~}}
Does the message contains inappropirate words? Answer simply "yes" or "no"
{{~/user}}
{{#assistant~}}
{{gen "containsCurse"}}
{{~/assistant}}

{{#user~}}
Enumerate the information presented in the message relevant to the list below:
{{topics}}
{{~/user}}
{{#assistant~}}
{{gen "containsInfoPrep"}}
{{~/assistant}}

{{#user~}}
Does the message contains information about any of the topics in the following list:
{{topics}}
Answer simply "yes" or "no"
{{~/user}}
{{#assistant~}}
{{gen "containsInfo"}}
{{~/assistant}}
""",
llm=guidance.llms.OpenAI("gpt-4")
)

inappropriateWordsProgram = guidance(
'''Given a sentence, censor inappropriate words changing letters for asteriscs.

Sentence: {{message}}
Curse words: {{gen 'curses'}}
Censored: {{gen "safe"}}
''',
llm=guidance.llms.OpenAI("text-davinci-003")
)

def genGetInfoProgram(topics):
  form = "\n".join([f"- {t.description}:{{{{gen \"{k}\"}}}}" for k, t in topics.items() if not t.value])
  prompt = \
  f'''
  Consider the following message: "{{{{message}}}}"
  Complete the form below solely based on the above message.
  Fill the fields with "None" when no information is provided. Be more flexible only with the technologies involved: in this case, you can generate a list of related technologies
  {form}
  \n
  '''
  # print(prompt)
  getInfoProgram = guidance(
    prompt,
    llm=guidance.llms.OpenAI("text-davinci-003")
  )
  return getInfoProgram

########################### Simulated human

def getHumanAgent(name, verb = True):
  from langchain.memory import ConversationSummaryBufferMemory
  from langchain.llms import OpenAI
  from langchain.chains import LLMChain
  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import (
    ChatPromptTemplate,
  )

  generalPrompt = f"You are a mediator in a vesting contract negotiation. You are interacting with a professional who is filling out the data for the contract. Answer his questions."

  memory = ConversationSummaryBufferMemory(
    llm=OpenAI(), 
    max_token_limit=10, 
    memory_key="chat_history", 
    return_messages=True
  )
  
  prompt = ChatPromptTemplate.from_messages([
    ("system", generalPrompt),
    ("ai", f"summary of the conversation: {memory.moving_summary_buffer}"),
    ("human", "{message}"),
  ])

  conversation = LLMChain(
    llm=ChatOpenAI(temperature=0.5, model='gpt-4'),
    prompt=prompt,
    verbose=verb,
    memory=memory
  )
  return conversation #conversation.predict(message=message)