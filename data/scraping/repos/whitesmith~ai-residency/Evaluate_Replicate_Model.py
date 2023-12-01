from langchain.evaluation.loading import load_evaluator
from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage)
from langchain.prompts.chat import (
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  )
import os

from dotenv import load_dotenv

load_dotenv()


## MODEL
BASE_URL = os.environ["BASE_URL"]
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]
API_KEY = os.environ["API_KEY"]

model1 = AzureChatOpenAI(
openai_api_base=BASE_URL,
openai_api_version="2023-05-15",
deployment_name=DEPLOYMENT_NAME,
openai_api_key=API_KEY,
openai_api_type="azure",
temperature = 0)
###


###EXTRACT EXAMPLES

examples = []
question = ""
answer = ""
Q = False
with open("examples.txt", "r") as f:
 for line in f:
   if line.lower() == "question\n":
       Q = True
       if question != "":
           examples.append({"query":question,"answer":anwer})
       question = ""
       answer = ""
       continue
   if line.lower() == "answer\n":
       Q = False
       continue
   if Q:
       question += line
   else:
       answer += line
 f.close()
examples.append({"query":question,"answer":answer})
for dictionary in examples:
 dictionary["context"] = "You are a Large Language Model that can use a database and evaluate this question in Correctness and compare it with the Reference Answer. The answer can be CORRECT, PARTIALLY CORRECT and INCORRECT, when compared with the reference."

###



def results_model(model,examples): # FUNCTION TO RETAIN THE ANSWERS OF A GIVEN MODEL TO THE QUESTIONS
  results = []
  for example in examples:
   results.append({"result":model( messages = [HumanMessage(content = example["query"]) ] ).content})
  return results

def results_model_LLM(model,examples):
   temp = PromptTemplate(template = "You will be given a question. You will need to answer this question to the best of your ability and show your reasoning.\n \
                 QUESTION:\n'''{question}'''\n \
                 Be sure to present your answer in the following form:\n \
                 Thoughts:<Insert first impression about question> \n \
                 Reasoning: <Reason a valid answer> \n \
                 Be sure to repeat the above process until you reach a Final Answer.\n\nANSWER: \n",input_variables = ["question"])
   chain = LLMChain(prompt = temp, llm = model, verbose = False)
   results = []
   for example in examples:
       results.append({"result":chain.run(question = example["query"])})
   return results


evaluator_context = load_evaluator("context_qa",llm = model1)
evaluator = load_evaluator("qa",llm = model1)  ## GPT-3.5-turbo evaluators


## WILL REQUIRE REPLICATE API KEY AS ENV VARIABLE
llm = Replicate(
model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
 input={"temperature": 0.5, "max_length": 500, "top_p": 1},
)
results = results_model_LLM(model,examples)
evaluation = evaluator_context.evaluate(examples,results)
## WILL EVALUATE llama13b-v2 with a complete prompt using gpt-3.5-turbo as the evaluator
## WILL OUT PUT LIST OF DICTIONARIES [{text:CORRECT/PARTIALLY CORRECT/INCORRECT},...], for each question



