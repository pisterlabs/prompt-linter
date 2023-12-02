import guidance
import openai
import config
import pandas as pd

openai.api_key = config.openai_api_key
guidance.llm = guidance.llms.OpenAI('gpt-3.5-turbo-0613', temperature=0.7)

df = pd.read_csv('FronxOwnerManual-Dataset.csv')
df = df[df['question'].isnull() & df['answer'].isnull()]
print(df)

for index,row in df.iterrows():
    context = row['context']

    create_plan = guidance('''
    {{#system~}}
    You will create a question and answer based from the given context.
    The context came from a car manual guide.
    The questions you will make should not be absurd and it common for a human to ask.
    {{~/system}}
                        
    {{#user~}}
    Here is the context:        
    {{context}}
                        
    You will create a question first. so dont answer yet.
    Question:
    {{~/user}}
                        
    {{#assistant~}}
    {{gen 'question' temperature=0 max_tokens=500}}
    {{~/assistant}}

    {{#user~}}
    Ok, now that you have a question, you can answer it.
                        
    Answer:
    {{~/user}}      
                        
    {{#assistant~}}
    {{gen 'answer' temperature=0 max_tokens=500}}
    {{~/assistant}}
    ''')
                        
    out = create_plan(context=context)

    df.at[index, 'question'] =  out['question']
    df.at[index, 'answer'] =  out['answer']

    # Save the updated dataframe to a new CSV file
    df.to_csv('UpdatedFronxOwnerManual-Dataset.csv', index=False)