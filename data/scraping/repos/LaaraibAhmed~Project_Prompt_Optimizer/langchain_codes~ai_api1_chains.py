import openai
import config
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain

promptlist=['give only question that user should ask openai if he/she want to know: {query}. '+
        'question must be as if you are {v_role}'+
        'question must be in the manner: {v_formal}'+
        'question must indicate that response should be {v_opinion}'+   #opinion/no Opinion
        'question must be must address the openai by name: {v_name}'+
        'question must be must be related to the location: {v_location}'+
        'question must be must be related to the interest: {v_interest}']

promptvarslist={'userprompt':"",'vrole':"counsellor",
                'vai_formal': "formal",
                'vOpinion':"no opinion except only facts",
                'vname_tocall':"dont include it as it was not provided",
                'vuser_place':"dont include it as it was not provided",
                'vuser_interest':"dont include it as it was not provided"}
promptuservars=['query','v_role','v_formal','v_opinion','v_name','v_location','v_interest']

getchains={}

model=ChatOpenAI(openai_api_key=config.openai_keys.API_KEY,temperature=0.2,model='gpt-3.5-turbo')# add further parameters as per user preference



n=0
def getmsgs(**kwargs)->None:
    global n
    # vrole="",vai_formal="",vOpinion="",vname_tocall="",vuser_place="",vuser_interest="",userprompt=""
    if n>=len(kwargs.items):
        return 
    
    template=HumanMessagePromptTemplate.from_template(
        kwargs[promptlist][n]
    )
    chat_prompt=ChatPromptTemplate.from_messages([template])
    chat_prompt=chat_prompt.format_prompt(kwargs[promptuservars][n]+'='+kwargs[promptvarslist][n])
    messages=[SystemMessage(content=str(kwargs[promptlist][n])),
        chat_prompt.to_messages()[0]
        ]
    
    chat = ChatOpenAI(openai_api_key=config.openai_keys.API_KEY,temperature=0.2,model='gpt-3.5-turbo')
    getchains[n] = LLMChain(llm=chat, prompt=messages)
    
    getmsgs(kwargs)
    return None

    # response=model(messages)
    # print(response.content)
    # pass


def usermsg(**kwargs):

    getmsgs(kwargs)
    sendchain=SimpleSequentialChain(chains=getchains)
    res=sendchain.run()
    print(res)

usermsg(vrole="a researcher of plants",vai_formal="formal",v_location='Pakistan',userprompt="what to pollination",promptlist=promptlist,promptdic=promptvarslist,promptuservars=promptuservars)