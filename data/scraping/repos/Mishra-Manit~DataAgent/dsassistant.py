import openai
import APIKEY
import io
import sys

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

openai.api_key = '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'


#Replace this with the actual file path

filePath = 'Toy Datasets/Large/Real Estate/New Real Estate.csv'
datasetcsv = pd.read_csv(filePath)

#Replace this with the question
actualQuestion = 'How many columns are in the dataset '


allContext = {
        #"datasetcsv.head()" : datasetcsv.head(),
        #"datasetcsv.describe()" : datasetcsv.describe(),
        "file path of dataset": filePath
}

textContext = ''
for key, value in allContext.items():
    textContext += f'{key}: {value}\n'

#print(textContext)


# Define the system message
system_msg = 'You are an assistant that is a data scientist. Your objective is to execute given tasks with utmost accuracy and precision. Avoid providing unnecessary explanations and focus solely on delivering the exact results or outputs requested. Only give answers and do not provide supplementary information unless specifically asked to do so. Make sure to always make variables if you are going to call them later!'



#This first message asks for code to get any information from the dataset for this project
user_msg1 = '''
For the question going to be asked of you, only answer with the code needed access the information from the dataset needed to answer the question at hand. 

Here is some background information about the dataset: {}

DO NOT REQUEST LARGE AMOUNTS OF DATA SUCH AS THE WHOLE COLUMN. 

You will only respond with the python code that can be directly run inside of an IDE, with no extra explanation. Write the code to access specific information about the dataset, since the dataset is not provided to you. Only write question that should access preliminary information from the dataset that is needed to solve the question asked.

Use a print statement to display the information.

The variable where the dataset is stored is named datasetcsv

DO NOT ANSWER THE QUESTION IN ANY CAPACITY.

IMPORTANT: For this step, you can only get text information, no graphs or plots can be created. Make sure you only generate information that is in text format. For example, you can not write code to generate a graph here.

Example Question and Answer:
Question: "Make a barplot displaying different columns"
Answer:  print(datasetcsv.columns)

Real Question: {} but do not write code to generate the answer'

'''.format(textContext, actualQuestion)

response1 = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[{"role": "system", "content": system_msg},
                                                  {"role": "user", "content": user_msg1}],

                                        temperature=0.00,
                                        max_tokens=2048,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0
                                         )
answer1 = response1["choices"][0]["message"]["content"]

print("answer1: " + answer1)

backup = sys.stdout
sys.stdout = io.StringIO()
#Below is what actually executes
exec(answer1)
# Get the printed value and store it
output = sys.stdout.getvalue()

# Restore stdout
sys.stdout = backup

output1 = output.strip()
print(f'this is what GPT wanted as context: "{output.strip()}"')


#Updating the context dictionary
allContext[answer1] = output1


#Remake the textContext variable with the new addition
textContext = ''
for key, value in allContext.items():
    textContext += f'{key}: {value}\n'

#This step generates the list of action items to be completed
user_msg2 = '''

For the question going to be asked of you, only answer with a list of steps needed to execute in order to find the final answer. DO NOT give the final answer, instead, in a array format, give a list of steps needed to arrive at the answer. After generating these steps, review them and make sure they are correct. 

Here is some background information about the dataset: {}

The output generated will be turned into an array, so ensure that correct array syntax is made. There should be no extra characters, just the array.

In a previous prompt, some preliminary context about the dataset was generated, it is in the dictionary text above.

In your generate an array of steps and write your answer giving the steps such as this EXAMPLE:
Example Question: "what is the number of columns and largest data value out of these columns"
ExampleGPTAnswer: ["Load the dataset from the file "boston.csv", "Get the number of columns in the dataset.", "Find the largest data value in each column.", "Identify the column with the largest data value.", "Determine the largest data value."]

Real Question: {}

'''.format(textContext, actualQuestion)

response2 = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[{"role": "system", "content": system_msg},
                                                  #make sure to change the user_msg variable
                                                  {"role": "user", "content": user_msg2}],

                                        temperature=0.00,
                                        max_tokens=2048,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0
                                         )
answer2 = response2["choices"][0]["message"]["content"]

print("This is answer two: " + answer2)
#this changes from text to a list
answer2 = eval(answer2)

print("Array of steps: " + str(answer2))

#Updating the context dictionary
allContext["array of steps"] = answer2
textContext = ''
for key, value in allContext.items():
    textContext += f'{key}: {value}\n'

#Variable defined to store previous context
previousAnswer = "This is the previous question: "+ str(answer1) + ", and this is the output that was generated: " + str(output1)

for i in range(len(answer2)):
        
        if i < len(answer2):

                textContext = ''
                for key, value in allContext.items():
                        textContext += f'{key}: {value}\n'
    
                        #I have already imported the correct libraries and datasets. Do not import and libraries. 
                user_msg3 = '''
                        In order to answer a larger question, there are a series of smaller steps generated. Each step will be completed and then the result will be calculated in the last step.

                        You will only respond with the python code that can be directly run inside of an IDE, with no extra explanation.Use a print statement to display the information.

                        Here is some information after running some commands on the dataset: {}

                        DO NOT ANSWER THE QUESTION IN ANY CAPACITY until the last step in the array. The loop is currently on this number step {}, out of {} total steps.

                        Make sure not to use .plot for any of the steps. 

                        Example Question and Answer:
                        Q: "what are the name of the columns in the dataset"
                        A:  print(boston.head())

                        In a previous prompt, some preliminary context about the dataset was generated. This was the previous code and output: {}

                        The file is titled "boston.csv"

                        Because these steps are not the last steps, make sure not to use .plot() in any of the steps. Furthermore, don't repeat steps such as opening the dataset or checking the name of the columns.

                        IMPORTANT: Only write code steps and no english language as the writing is being executed.

                        Instruction: {}'

                        '''.format(textContext, i+1, len(answer2), previousAnswer, answer2[i])

                response3 = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                        messages=[{"role": "system", "content": system_msg},
                                                                {"role": "user", "content": user_msg3}],

                                                        temperature=0.00,
                                                        max_tokens=2048,
                                                        top_p=1,
                                                        frequency_penalty=0,
                                                        presence_penalty=0
                                                        )       
                answer3 = response3["choices"][0]["message"]["content"]
                #print(answer3)
                print("answer3: " + answer3)
                # Backup of stdout
                backup = sys.stdout

                # Create a string IO buffer
                sys.stdout = io.StringIO()

                # Execute the code
                exec(answer3)

                # Get the printed value and store it
                output = sys.stdout.getvalue()

                # Restore stdout
                sys.stdout = backup

                output4 = output.strip()

                
                #Updating the context dictionary
                allContext[answer2[i]] = output4

                previousAnswer = "This is the previous question: "+ str(answer3) + ", and this is the output that was generated: " + str(output4)


        textContext = ''
        for key, value in allContext.items():
                textContext += f'{key}: {value}\n'

        if (i+1) == len(answer2):
                user_msg4 = '''
                Here is some information after running some commands on the dataset: {}.
                
                This is the last step. step {}, this was the previous instruction and output: {}. You are to only write the code for this last step, which upon executing will give the answer to our originial question: {}. Make sure that you only write the code, and display the ANSWER WITH A PRINT statement for numerical outputs. DO NOT use print statements when plotting graphs with matplotlib.

                Do not provide additional instructions as the code generated will be executed in an IDE. 

                Example of a few instructions and answers:
                Q: Display the heatmap.
                A: sns.heatmap(datasetcsv.corr())
                   plt.show()

                   
                Print Statement Example: 
                Q: Calculate the correlation between the 'zn' and 'crim' columns.
                A: print(datasetcsv['zn'].corr(datasetcsv['crim']))
                
                IMPORTANT: DO USE print statements when displaying final answers which are not graphs

                DO NOT use print statements when plotting graphs with matplotlib. 

                This is the final instruction: {}.

                IMPORTANT: When the final answer is a graph or a plot, MAKE SURE TO USE plt.show() function to display the final answer.
                IMPORTANT: Use a print statement on the final answer for all other answers!

                '''.format(textContext, len(answer2), previousAnswer, actualQuestion, answer2[i])

                response4 = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                        messages=[{"role": "system", "content": system_msg},
                                                                {"role": "user", "content": user_msg4}],

                                                        temperature=0.00,
                                                        max_tokens=2048,
                                                        top_p=1,
                                                        frequency_penalty=0,
                                                        presence_penalty=0
                                                        )       
                answer4 = response4["choices"][0]["message"]["content"]

                print("This is answer 5: " + answer4)

                backup = sys.stdout
                sys.stdout = io.StringIO()
                #make sure to change the following line of code with the right answer choice
                exec(answer4)
                output5 = sys.stdout.getvalue()
                sys.stdout = backup
                output5 = output5.strip()

                print(output5)

                print("Final Answer given by GPT: ", output5)
                

        



