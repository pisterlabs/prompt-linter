from langchain.llms import HuggingFaceEndpoint
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

''' Based on: https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/51 '''

HUGGINGFACEHUB_API_TOKEN = ""
from dotenv import load_dotenv
load_dotenv()

endpoint_url = (
            'https://YOUR_ENDPOINT.us-east-1.aws.endpoints.huggingface.cloud'
)
hf = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token= HUGGINGFACEHUB_API_TOKEN,
    task = 'text-generation' 
)

template = """
The men's high jump event at the 2020 Summer Olympics took place between 30 July and 1 August 2021 at the Olympic Stadium.
33 athletes from 24 nations competed; the total possible number depended on how many nations would use universality places 
to enter athletes in addition to the 32 qualifying through mark or ranking (no universality places were used in 2021).
Italian athlete Gianmarco Tamberi along with Qatari athlete Mutaz Essa Barshim emerged as joint winners of the event following
a tie between both of them as they cleared 2.37m. Both Tamberi and Barshim agreed to share the gold medal in a rare instance
where the athletes of different nations had agreed to share the same medal in the history of Olympics. 
Barshim in particular was heard to ask a competition official "Can we have two golds?" in response to being offered a 
'jump off'. Maksim Nedasekau of Belarus took bronze. The medals were the first ever in the men's high jump for Italy and 
Belarus, the first gold in the men's high jump for Italy and Qatar, and the third consecutive medal in the men's high jump
for Qatar (all by Barshim). Barshim became only the second man to earn three medals in high jump, joining Patrik Sjöberg
of Sweden (1984 to 1992).

Question: {question} """

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=hf, verbose=True)

question = "Who won the 2020 Summer Olympics men's high jump?"

print(llm_chain.run(question))