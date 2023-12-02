from langchain.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd



df = pd.read_csv("1000_random_sample.csv")
model_path = " "
#define the language parameters
llm = VLLM(
    model=model_path,
    tensor_parallel_size=8,
    trust_remote_code=True,  # mandatory for hf models
    top_p=1,
    temperature=0,

)


#define the prompt input and output template
template = """Question: {question}

Answer:Give your answer for the question as either "Yes" or "No"."""



prompt = PromptTemplate(template=template, input_variables=["question"])


#feed the llm to the Langchain
llm_chain = LLMChain(prompt=prompt, llm=llm)


quest_list = [ 
'disease',
'medical condition',
'drug',
'device',
'Dose or measurements',
'clinical trial phase',
'population',
'Time',
'Medical Procedure',
'Biomarker'
]

question = f""" Does the following text includes any named entities with {{}} type?:
  ```{{}}```
"""


# feeding input to the model in batches of 1000.
BATCH_SIZE = 1000

for i in range(0,df.shape[0], BATCH_SIZE):
        for column_ in quest_list:
            
            generated_df = pd.read_pickle('bin_random_named_entities.pkl')    
            inputs_ = [{'question':question.format(column_,t)} for t in df['sum'][i:i+BATCH_SIZE].values]
            nct_ =  [ nct for nct in df['NCTID'][i:i+BATCH_SIZE].values]

            outputs = llm_chain.generate(inputs_)
            outputs = [[outputs.generations[i][0].text] for i in range(len(outputs.generations))]
          
            for output, nct in  zip (outputs, nct_):
                res = output[0]
                yes_no = None
                
                index_of_explanation = res.lower().find("explanation")
                if index_of_explanation != -1:
                    res = res[:index_of_explanation]
                
                if 'yes' in res.lower():
                    yes_no = 1
                elif 'no' in res.lower():
                    yes_no = 0
                    
                generated_df.loc[generated_df['NCTID'] == nct, column_] = yes_no

            generated_df.to_pickle('bin_random_named_entities.pkl')    


            
