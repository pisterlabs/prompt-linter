from typing import Tuple

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_party.linkedin import scrape_linkedin_profile

from output_parser import person_intel_parser, LinkedinProfile


def build_chain():
    summary_input = """
        give the information about {information} that includes: 
        1. a short summary, less than 100 words
        2. two interesting facts about {information}
    """
    prompt = PromptTemplate(input_variables=["information"], template=summary_input)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def ice_break(name, description="") -> Tuple[LinkedinProfile, str]:
    """
    流程： 1. 给一个名字通过agent找到linkedin的url 2. 爬取linkedin的信息， 3. 通过chain生成summary， 4. 通过parser解析结果
    :return:
    """
    linkedin_profile_url = linkedin_lookup_agent(name=name, description=description)
    summary_template = """
        give the information {information} about a persion from I want you to create: 
        1. a short summary, less than 100 words
        2. two interesting facts about them
        3. a topic that may interest them
        4. 2 creative Ice breakers to open a conversation with them
        \n{format_instruction}
    """
    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instruction": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = LLMChain(llm=llm, prompt=prompt)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url)

    res = chain.run(information=linkedin_data)
    person_parser = person_intel_parser.parse(res)
    return person_parser, person_parser.pic


if __name__ == "__main__":
    load_dotenv()  # This loads the .env file into the environment
    person_parser, pic_url = ice_break(
        "Yifan Wu",
        description="He is a phd and currently is doing postdoctoral research in Havarad University",
    )
    print(person_parser)
    print(pic_url)
