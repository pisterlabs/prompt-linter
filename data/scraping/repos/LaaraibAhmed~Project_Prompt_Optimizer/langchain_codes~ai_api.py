import openai
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain
from langchain.memory import SimpleMemory
# from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import DirectoryLoader

import config
import os


os.environ['OPENAI_API_KEY']=config.openai_keys.API_KEY
llm=OpenAI(temperature=0.2)
# user_reqs_chain=None
# query_chain=None


# pipeline in seperate apicall
def input_user_reqs(vrole="",vai_formal="",vlong_short="",vOpinion="",vname_tocall="",vuser_place="",vuser_interest="",userprompt=""):
    full_template="""what should I ask if I want to know about: 
    {user_query}
    Use the following context:
    {role}
    {ai_formal}
    {long_short}
    {Opinion}
    {name_tocall}
    {user_place}
    {user_interest}
    """
    full_prompt=PromptTemplate.from_template(full_template)

    role_template="""answer as if you are {v_role}"""
    role_prompt=PromptTemplate.from_template(role_template)

    formal_template="""answer in the manner: {v_formal}"""
    formal_prompt=PromptTemplate.from_template(formal_template)

    long_template="""answer must be {v_long}""" #long/short
    long_prompt=PromptTemplate.from_template(long_template)

    opinion_template="""answer must have : {v_opinion}""" #opinion/no Opinion
    opinion_prompt=PromptTemplate.from_template(opinion_template)

    name_template="""answer must address the user by name: {v_name}"""
    name_prompt=PromptTemplate.from_template(name_template)

    user_place_template="""answer must be related to the location: {v_location}"""
    user_place_prompt=PromptTemplate.from_template(user_place_template)

    user_interest_template="""answer must be related to the interest: {v_interest}"""
    user_interest_prompt=PromptTemplate.from_template(user_interest_template)

    user_query_template="""{v_user_query}"""
    user_query_prompt=PromptTemplate.from_template(user_query_template)

    input_prompts=[
        ("role",role_prompt if (vrole) else role_prompt+"any as suited"),
        ("ai_formal",formal_prompt if (vai_formal) else formal_prompt+"any as suited"),
        ("long_short",long_prompt if (vlong_short) else long_prompt+"any as suited"),
        ("Opinion",opinion_prompt if (vOpinion) else opinion_prompt+"any as suited"),
        ("name_tocall",name_prompt if (vname_tocall) else name_prompt+"any as suited"),
        ("user_place",user_place_prompt if (vuser_place) else user_place_prompt+"any as suited"),
        ("user_interest",user_interest_prompt if (vuser_interest) else user_interest_prompt+"any as suited"),
        ("user_query",user_query_prompt)
        ]
    prompt_pipeline=PipelinePromptTemplate(final_prompt=full_prompt,pipeline_prompts=input_prompts)

    # all vars from app.py
    print(llm(prompt_pipeline.format(
        v_role=vrole,
        v_formal=vai_formal,
        v_long=vlong_short,
        v_opinion=vOpinion,
        v_name=vname_tocall,
        v_location=vuser_place,
        v_interest=vuser_interest,
        v_user_query=userprompt
    )))
    

input_user_reqs(userprompt="study the basics of quantum mechanics")