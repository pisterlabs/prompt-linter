from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAIChat
from langchain.llms.base import LLM
import os 
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
import pandas as pd
from time import sleep

pandas_dataframe = pd.read_csv("db.csv")

def fake_db_lookup(query: str, llm: LLM):
    print("Making Lookup")
    examples = examples = [    {        
        "question": "Can you select the 'name' and 'age' columns from the database",       
        "answer": "pandas_dataframe.loc[:, ['name', 'age']]"
        },    
        {        
        "question": "Can you calculate the average of the 'age' column from the database",        
        "answer": "pandas_dataframe['age'].mean()"    
        },    
        {        
        "question": "Can you plot the 'age' and 'income' columns from the database",        
        "answer": "pandas_dataframe.plot(x='age', y='income')"    
        }]

    database_query_prompt = PromptTemplate(
        input_variables=['columns', 'query'],
        template=f"Following the reasoning: \nFor example: {examples[0]['question']}\n{examples[0]['answer']}\n{examples[1]['question']}\n{examples[1]['answer']}\n{examples[2]['question']}\n{examples[2]['answer']}\nWithin the following database we have the following columns"+"{columns}. How can I {query} from the variable pandas_dataframe?. Give only one answer and don't include any library imports. Actual code: ```py import pandas as pd\nimport matplotlib.pyplot as plt\npandas_dataframe('db_csv')``` continue the code",
    )



    code_extraction_prompt = PromptTemplate(
        input_variables=['generated_text'],
        template= """
        This are 3 examples of how extract code from text:
           Begin Example 1
              User Input:
                The answer for the question can you make a program that prints hello world is:
                ```
                print("hello world")
                ```
                However, the code is very basic because it only prints hello world.
              Your Output:
                ```
                    print("hello world")
                ```
            End Example 1
            Begin Example 2
                User Input:
                    The answer for the "fibonacci sequence of 10" is:
                    def fibonacci(n):
                        if n <= 1:
                            return n
                        else:
                            return(fibonacci(n-1) + fibonacci(n-2))
                    print(fibonacci(10))
                    Note that the variable n is the number of the fibonacci sequence. is the number of the fibonacci sequence.
                Your Output:
                    ```
                    def fibonacci(n):
                        if n <= 1:
                            return n
                        else:
                            return(fibonacci(n-1) + fibonacci(n-2))
                    print(fibonacci(10))
                    ```
            End Example 2
            Begin Example 3
                User Input:
                    The code for select the columns 'name' and 'age' from the database is:
                        pandas_dataframe.loc[:, ['name', 'age']]
                    However it seems the columns names that you provide are not corect.
                Your Output:
                    ```
                    pandas_dataframe.loc[:, ['name', 'age']]
                    ```
            End Example 3

            Now that you have seen 3 examples of how to extract code from text, please extract the code from the following text:
            Important:  Only give your answer, dont give any additional text, only the answer.
            
            User Input:
                {generated_text}
            Your Output: 
         """
    )
    


    code_analysis_prompt = PromptTemplate(
        input_variables=['generated_code'],
        template="""
        There is a lot of information that can be extracted from code. I will give you 3 examples of how to extract information from code.
        Begin Example 1
            User Input:
                pandas_dataframe['Sex'].value_counts().plot(kind='bar')
                plt.xticks([0, 1], ['Man', 'Woman'])
                plt.show()
            Your Thought Process Will be:
                Dependencies: Pandas, Matplotlib
                Requires Plotting: True
                Requires Result Printing: False
            Your Output:
                ["Pandas, Matplotlib", True, False]
        End Example 1

        Begin Example 2
            User Input:
                def fibonacci(n):
                    if n <= 1:
                        return n
                    else:
                        return(fibonacci(n-1) + fibonacci(n-2))
                print(fibonacci(10))
            Your Thought Process Will be:
                Dependencies: None
                Requires Plotting: False
                Requires Result Printing: True
            Your Output:
                ["None", False, True]
        End Example 2
        
        Begin Example 3
            User Input:
                pandas_dataframe['age'].mean()
            Your Thought Process Will be:
                Dependencies: Pandas
                Requires Plotting: False
                Requires Evaluation: True
            Your Output:
                ["Pandas", False, True]
        Can you give me Your Output for the following code supressing your thought process, is important that you only give me the output, not the input or any other text, and dont repeat "Your Output:" Only give me the dependecies, requires ploting, and requires result printing nothing more.
            {generated_code}
        Your Output:
        """
    )

    print("Original question: ", query)


    chain_code_generation = LLMChain(llm=llm, prompt=database_query_prompt)
    chain_code_extraction = LLMChain(llm=llm, prompt=code_extraction_prompt)
    chain_code_analysis = LLMChain(llm=llm, prompt=code_analysis_prompt)

    params ={
        "columns": pandas_dataframe.columns,
        "query": query
    }
    #print('The question was: ', query)
    output = chain_code_generation.run(**params)

    #print("\n")
    #print("Waiting 1 minute for the code to be generated")
    code = chain_code_extraction.run(generated_text=output)

    # Actual code is between the ```py and ```
    code = code.replace("```py", "").replace("```", "")
    #print("Waiting 1 minute for the code to be extracted")
    analysis = chain_code_analysis.run(generated_code=code)

    #print("====Begin Generated Code====")
    #print(code)
    #print("====End Generated Code====")
    #print("\n")
    #print("======Code Analysis Results======")
    #print(analysis)
    #print("======End Code Analysis Results======")

    return code, analysis

#llm = YouChatLLM(api_key=os.environ['CHATAPI'])
#llm = OpenAILLM(model="gpt-3.5-turbo", temperature=0.1)
#llm = HuggingFaceEndpoint(endpoint_url="https://api-inference.huggingface.co/models/togethercomputer/GPT-JT-6B-v0",
#                          huggingfacehub_api_token=os.environ['HFAPI'], task="text-generation", 
#                          model_kwargs={"max_length": 1000, "min_lenght": 50})

def code_inyection(generated_code_path, db_path, model_code, model_analysis):
    print("Inyection")
    generated_code = []
    generated_code.append("import pandas as pd")
    generated_code.append("import matplotlib.pyplot as plt")
    generated_code += f"pandas_dataframe = pd.read_csv('{db_path}')", model_code.replace("plt.show()", "").replace("\t", "").replace(" ", "")
    # Model Analysis input: ["Pandas, Matplitlib", True, False]
    model_analysis = model_analysis.replace("[", "").replace("]", "").replace("'", "").split(", ")
    plot_image_path = None
    plot_image_path = "plot.png"
    generated_code.append(f"plt.savefig('{plot_image_path}')")
    generated_code = "\n".join(generated_code)
    with open(generated_code_path, "w+") as f:
        print("Guardando")
        f.write(generated_code)

    return generated_code_path, plot_image_path, generated_code

def execute_code(file_path):
    print("Excecuting")
    os.system(f"python {file_path}")

