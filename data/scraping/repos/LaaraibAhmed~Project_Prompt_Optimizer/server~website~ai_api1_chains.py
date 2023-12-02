from config import openai_keys
from sampleprompts import sampleprompts
import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain.schema import SystemMessage,HumanMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import SimpleMemory
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

                            ###################### ###################### ###################### 
                            #################### CODE FOR PROMPT GENERATION #################### 
                            ###################### ###################### ###################### 

os.environ['OPENAI_API_KEY']=openai_keys.API_KEY

promptlist=["""give a comprehensive question that user should ask if he/she wants to know: 
            {query0}.""" ,
            """{query1} 
            change the question so that it can be asked by {v_role}""",
            """change the question such that it sounds {v_formal}:
            {query2}""",
            """change the question so it indicates that answer returned must be in {v_wordlength} words:
            {query3}""",   #opinion/no Opinion
            """change the question below so that it matches the interest of {v_interest}:
            {query4}""",
            """{query5}"""]

# promptvarslist input by user on frontend..
promptvarslist={'userprompt':"",
                'v_role':"",
                'v_formal': "",
                'v_wordlength':"",
                'v_interest':""}


promptuservars=['userprompt','v_role','v_formal','v_wordlength','v_interest']
promptuservars1=['userprompt','v_role','vai_formal','v_wordlength','vuser_interest']

# predefined vars
getchains={}
n=0
model=ChatOpenAI(openai_api_key=openai_keys.API_KEY,temperature=0.1,model='gpt-4')# add further parameters as per user preference



def getmsgs(**kwargs)->None:
    global n,model
    if n>=len(kwargs['promptuservars'])+1:
        return 
    
    chat_prompt=HumanMessagePromptTemplate.from_template(
        kwargs['promptlist'][n] ,
        input_variable=[kwargs['promptuservars'][n] if n!=0 and n!=5 else None,f'query{n}']
    )

    messages=ChatPromptTemplate.from_messages([SystemMessage(content="""change the question purely to include the parameter as per user instructions"""),
        chat_prompt
        ])
    
    # print(n)
    # print("\n\n\n\n")
    getchains[n] = LLMChain(llm=model, prompt=messages, output_key=f'query{n+1}'if n!=7 else 'out')
        
    n+=1
    getmsgs(**kwargs)
    return None

    

# build promptvarslist
def build_promptvarslist(**kwargs):
    for key,val in kwargs.items():
        if key in promptvarslist and val!="":
            promptvarslist[key]=val
    return promptvarslist
        # print(kwargs['promptvarslist'])   
    # print(kwargs['promptvarslist'])


def mainchain(**kwargs):
    global getchains
    chain=[]
    for _,value in getchains.items():
        chain.append(value)
    # print(kwargs['promptvarslist'])

    mainchain=SequentialChain(
        memory=SimpleMemory(memories=kwargs['promptvarslist']),
        chains=chain,
        input_variables=['query0'],
        output_variables=['query5']
    )
    return mainchain


# code for displaying output of the prompt generated for the user by chatmodel
def return_output(**kwargs):
    res=kwargs['res']
    messages=[
        SystemMessage(content="Answer the question as per directed"),
        HumanMessage(content=res)
    ]
    print("\nThe answer to generated user prompt is: \n")
    # print(model(messages).content)
    return model(messages).content

def usermsg(**kwargs)->str:
    kwargs['promptvarslist']=build_promptvarslist(**kwargs)
    # print(kwargs)
    getmsgs(**kwargs)
    finalchain=mainchain(**kwargs)
    res=finalchain.run(kwargs['userprompt'])
    print('\n\n')
    print(kwargs['v_formal'])
    print(res)
    # fewshot(userpromptout=res,**kwargs)
    # print("\nand the answer is :\n")
    # print(type(res))
    print(return_output(res=res))
    return res

def userinput(**kwargs)->str:
    return usermsg(promptlist=promptlist,promptvarslist=promptvarslist,promptuservars1=promptuservars1,promptuservars=promptuservars,**kwargs)    


def fewshot(**kwargs):
    global model
    print(kwargs)
    examples=sampleprompts
    to_vector=[" ".join(eachexample.values()) for eachexample in examples]
    embeddings=OpenAIEmbeddings()
    customvectorstore=Chroma.from_texts(to_vector,embeddings,metadatas=examples)

    example_selector=SemanticSimilarityExampleSelector(
        vectorstore=customvectorstore,
        k=2
    )

    fewshotprompt=FewShotChatMessagePromptTemplate(
        input_variables=["userpromptout","role","interest","formality","wordlength"],
        example_selector=example_selector,                  
        example_prompt=ChatPromptTemplate.from_messages(
        [("human","{userpromptout},{role},{interest},{formality},{wordlength}"),("ai","{final_prompt}")]
        ),
    )

    fewshotprompt=fewshotprompt.format(userpromptout=kwargs['userpromptout'],wordlength=kwargs['v_wordlength'],formality=kwargs['v_formal'],interest=kwargs['v_interest'],role=kwargs['vrole'])
    print("\nresponse after fewshot is : \n")
    # temp=HumanMessagePromptTemplate.from_template(fewshotprompt)
    # final=ChatPromptTemplate.from_messages([temp])
    print ((fewshotprompt))
    # print(examples)
    return None

# test for tones by ibad
formals=['creative','humorous','Unimaginative','intelligent','straightforward']

for x in range(len(formals)):
    userinput(vrole="a science fiction author researching futuristic concepts",v_formal=formals[x] ,v_wordlength='200',v_interest='delving into advanced scientific ideas and translating them into engaging stories',userprompt="what are energy waves")


