from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.wikipedia_keywords.wikipedia_keywords_human_prompt import human_prompt
from chains.story_chain.prompts.wikipedia_keywords.wikipedia_keywords_system_prompt import system_prompt
import json
from typing import List

_messages = [
    SystemMessagePromptTemplate(prompt = system_prompt), 
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm = ChatOpenAI(temperature = 0.7, model = "gpt-4")
_chat_prompt = ChatPromptTemplate.from_messages(messages = _messages)
_wikipedia_keywords_chain = LLMChain(
    llm = _llm,
    prompt = _chat_prompt,
    verbose = True
)

def wikipedia_keywords_chain_v1(
    idea: str
) -> List[str]:
    keywords_str = _wikipedia_keywords_chain.run(
        {
            "idea": idea
        }
    )
    print(f"wikipedia_keywords_str = {keywords_str}")
    keywords_json = json.loads(keywords_str)
    return keywords_json["wikipedia_keywords"]
