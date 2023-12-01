#just pick 3 LLM's from replicate and keep their links in a list

LLMs = [
    "replicate/llama-7b:ac808388e2e9d8ed35a5bf2eaa7d83f0ad53f9e3df31a42e4eb0a0c3249b3165", #llama-7b
    "a16z-infra/llama7b-v2-chat:a845a72bb3fa3ae298143d13efa8873a2987dbf3d49c293513cd8abf4b845a83", #llama-2-7b
    "stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb", #stableLM

]

#these are all 7b models. Let's see if there's a difference for each when I run the same prompt. Note: may want to do some prompt engineering for each so i have alternatives.

from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain
import replicate
load_dotenv()

# Define the path 
data_path = Path('kaggle-llm-science-exam/train.csv')

# Use the path
df = pd.read_csv(data_path)

llm = Replicate(
    model=LLMs[2],
)
 
# let's get this prompt engineering bussing
template = """
You are a PhD level expert in a wide range of scientific fields. 
You are given a question and a set of answers. Think step by step to choose the correct answer.
Provide only the letter corresponding to the correct answer as a final output.

Example:
Which of the following statements accurately describes the impact of Modified Newtonian Dynamics (MOND) on the observed ""missing baryonic mass"" discrepancy in galaxy clusters?",
A: "MOND is a theory that reduces the observed missing baryonic mass in galaxy clusters by postulating the existence of a new form of matter called ""fuzzy dark matter.""
B: MOND is a theory that increases the discrepancy between the observed missing baryonic mass in galaxy clusters and the measured velocity dispersions from a factor of around 10 to a factor of about 20.
C: MOND is a theory that explains the missing baryonic mass in galaxy clusters that was previously considered dark matter by demonstrating that the mass is in the form of neutrinos and axions. 
D: MOND is a theory that reduces the discrepancy between the observed missing baryonic mass in galaxy clusters and the measured velocity dispersions from a factor of around 10 to a factor of about 2.
E: MOND is a theory that eliminates the observed missing baryonic mass in galaxy clusters by imposing a new mathematical formulation of gravity that does not require the existence of dark matter.
ANSWER: D



--------------------
Question:
{question}

A:
{A}
B:
{B}
C:
{C}
D:
{D}
E:
{E}


ANSWER:

OUTPUT ONLY A SINGLE CHARACTER 

Example: The answer is D. Output: D

---------------------
"""

prompt = PromptTemplate(template=template, input_variables=['question', 'A', 'B', 'C', 'D', 'E'])


# have to use chains 
chain = LLMChain(llm=llm, prompt=prompt)

# need to split the csv into an ingestible format
for i in range(1):
    row = df.iloc[i]
    id = row['id']
    question = row['prompt']
    A = row['A']
    B = row['B']
    C = row['C']
    D = row['D']
    E = row['D']
    answer = row['answer']
    print(id, question, answer)
    print(chain.run({'question': question,'A': A, 'B': B, 'C': C, 'D': D, 'E': E}))




  
#print(chain.run('who was robert hooke?'))


# need some logging for the benchmarking 
# also let's just do this with the test CSV to save time


