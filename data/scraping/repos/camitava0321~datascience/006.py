import logging
import os

from dotenv import load_dotenv

from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams, TokenParams

logging.getLogger("genai").setLevel(logging.DEBUG)


# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>
load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)
GENAI_KEY="pak-dlgq9J79SHjqkFeeD0r2FvneplVqkOfLVzIjy7bCy6Y"
GENAI_API="https://bam-api.res.ibm.com/v1/"
#GENAI_API="https://workbench-api.res.ibm.com/v1/"
api_key = GENAI_KEY
api_endpoint = GENAI_API

creds = Credentials(api_key=api_key, api_endpoint=api_endpoint)  # credentials object to access GENAI
#%%
#Inventory Source
space_offerings = [
"Marriott space A, people: 20, available: y", 
"Marriott space A is available to any customer", 
"Marriott space B, people: 20-40, available: n", 
"Marriott space C for more than 50 people", 
"Marriot space D for 15 people is not available", 
"Marriott space E is for upto 20 people and is for premium customer only", 
"Marriott space P,Q and R is each for upto 10 people and is for non-premium customer only", 
"Marriot space F for 15 people is available for premium customer only"
    ]

deal_combinations = [
"Marriott space A comes with private swimming pool", 
"Marriott space B comes with a lavish banquet service", 
"Marriott space P,Q,R comes with no special services", 
"Marriott space C with airport pickup/drop service", 
"Marriot D and F provides every service possible"
]

customer_profiles = [
"ClientA is premium customers", 
"ClientB is general customers", 
"ClientC is premium customer"
]

prompt = "Agent: Okay, I am awaiting your instruction   \
           User: Watson, here are your instructions:   \
1. You will be given a document that should be used to reply to user questions.  \
2. You should generate the next response using information available in the document.  \
3. You should combine deal combinations with the space offerings.  \
4. If you can't find an answer, say 'I don't know'.  \
5. Your responses should not be long and just have about 1-2 sentences.  \
6. You should not repeat your answers.  \
7. Do not use any other knowledge.  \
8. Tone: energetic \
Please repeat the instructions back to me so that I know you understood.  \
Agent: Ok, here are my instructions:  \
1. I will be given a document that should be used to reply to user questions.  \
2. I should generate the next response using information available in the document.  \
3. You should combine deal combinations with the space offerings.  \
4. If you can't find an answer, say 'I don't know'.  \
5. My responses should not be long and just have about 1-2 sentences.  \
6. I should not repeat your answers.  \
7. I will not use any other knowledge.  \
8. Tone: energetic \
I am ready, please give me the document.  \
User: Here's the document: " 
document = "{  \
Marriott as an event organiser offers services to provide well-executed events that attract \
guests and corporate clients. In any event we provide all our hotel facilities and services, \
for example different types of rooms in attractive rates, banquet catering service, \
non-catering services like pad and paper, overhead projector, sound system, microphones, \
standees, banners, airport or station drop and pickup etc. We also arrange rooms and hotel \
services in other hotels in the vicinity for guests participating in bigger events \
involving large number of people. We organise events that spans multiple cities. \
In such case we also assist in booking hotel services in other non-group hotels. \
For an event, we offer the complete service involving even arrangement of flowers, decoration.\
Event organiser will even be able to use our special services towards specially abled \
participants. But for a very small event, say less than 4 people, our services might become too costly for a customer. \
There will be a dedicated Event Manager from Marriott who will be responsible for planning, organizing, and executing various events, conferences, and functions. Event Managers ensure seamless coordination between clients, hotel staff, and vendors, ensuring successful and memorable events. Their expertise in logistics, budget management, and attention to detail contributes to the overall success and reputation of our hotel.  \
Data  "
Customer_profiles = "Customer Profiles: Marriott space B is for premium customer"

so = "Space offerings: "
for i in range(0, len(space_offerings)):
    so = so + space_offerings[i]+" "
dc = "Deal Combinations: "
for i in range(0, len(deal_combinations)):
    dc = dc + deal_combinations[i]+" "
    
document = document + so + " " + dc + Customer_profiles + "} "  
endText = "Agent: I am ready to answer your questions from the document. I will not repeat answers I have given."

def create_prompts (extra_knowledge, question):
    prompts = [prompt + document + extra_knowledge + endText + question]
    return prompts

prompts = create_prompts("","")
print(prompts[0])
params = GenerateParams(decoding_method="sample", 
                                 temperature=0,
                                 random_seed=111,
                                 top_k=50,
                                 max_new_tokens=100, min_new_tokens=10)

flan_ul2 = Model("google/flan-ul2", params=params, credentials=creds)
params1 = GenerateParams(decoding_method="greedy",
                                 max_new_tokens=1, min_new_tokens=1)
flan_ul2_class = Model("meta-llama/llama-2-70b-chat", params=params1, credentials=creds)
#%%
questions = [
    "Do you cover event offering for specially abled people?",
    "I am a premium customer. I am looking for an event involving 15 people. I prefer Marriott space B. Can you pls tell me that can be arranged?",
    "In that case, is there an alternative?",
    "Ok, what other services comes with it?",
    "Any deal?",
    "I have a premium customer who is looking for an event involving 15 people. He prefers Marriott space B. Can you pls tell me that can be arranged?",
    "Customer clientA is looking for an large event involving 200 people. Can you pls let me know what we can offer?",
    "Customer clientB is looking for an event involving less than 10 people. Can you pls let me know what we can offer?",
    "Customer ClientC is looking for an event involving 25 people. Can you pls tell me that can be arranged?  ",
    "Customer ClientA is looking for an event involving 3 people. Can you pls tell me that can be arranged?  ",
    "No more questions, Thank You!"
    ]

for i in range(0, len(questions)):
    print ("Question : "+str(i))
    question = "User: "+questions[i]
    print(question)
    prompts = create_prompts("", question)
    response = flan_ul2.generate(prompts);
    #print(len(response))
    generated_text = response[0].generated_text
    instr = "Classify this review as positive or negative. "
    print("Agent: "+generated_text)
    print (flan_ul2_class.generate([instr+generated_text])[0].generated_text+"\n")

#%%
import os
from dotenv import load_dotenv

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams, ModelType
from genai.credentials import Credentials


print("\n------------- Example (LangChain)-------------\n")

#params = GenerateParams(decoding_method="greedy")

print("Using GenAI Model expressed as LangChain Model via LangChainInterface:")

langchain_model = LangChainInterface(model=ModelType.FLAN_UL2, params=params, credentials=creds)

prompts = create_prompts("", questions[6])
response = langchain_model(prompts[0])
print(response)

#print(langchain_model("Answer this question: What is life?"))


