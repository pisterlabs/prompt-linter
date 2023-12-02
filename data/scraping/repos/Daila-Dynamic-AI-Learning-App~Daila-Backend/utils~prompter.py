#!/usr/bin/env python3
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import re
import keys

# set the openai api key
os.environ['OPENAI_API_KEY'] = keys.OPEN_API_KEY

def getConversation() -> ConversationChain:
    # using memory buffer for the current one in daila
    template = """
    {history}
    Human: {input}
    Professor:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    llm = ChatOpenAI(temperature=0.5, model_name='gpt-4')
    memory = ConversationBufferMemory(
        ai_prefix="Professor"
    )

    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=memory
    )

    return conversation

def get_question_or_assessment(convo: ConversationChain, input: str, assess: bool):
    """
        Function gets question or assesposment string
    """
    q_or_assess = convo.predict(input=input)

    if 'End$' in q_or_assess:
        # get the index of the end$ string and take only the area from after the end$ string
        index = q_or_assess.index('End$') + 4

        # get the final assessment
        final_assessment = q_or_assess[index:].strip()
        return (final_assessment, True)
    
    # final assessment has not been reached
    pattern = r'Question\s\d:\s(.+)'
    question = re.findall(pattern, q_or_assess)

    # return Question free from whitespaces
    return (question[0].strip(), False)