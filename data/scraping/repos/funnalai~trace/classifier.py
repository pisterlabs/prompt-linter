# import langchain
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv


load_dotenv()


def get_conv_classification(convStr, projNames):
    """
    Given a conversation, get the project it belongs to
    """
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                 temperature=0.7, max_tokens=5)
    prompt = f"{convStr}\n This conversation is about one of these projects {', '.join(projNames)}. Select which project matches the conversation best. Return None if no project is related."
    # make llm call
    response = llm(prompt)
    return response


def conversation_keywords(summary):
    """
    Generate three keywords about the summary
    """
    # return ""
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    prompt = PromptTemplate(
        input_variables=["summary"],
        template="Create a comma separated list of 3 keywords to represent this summary of a slack conversation{summary}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    words = chain.run(summary)
    return words
