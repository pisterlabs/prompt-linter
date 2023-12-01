from datasets import load_dataset
import time
import json
from langchain.prompts import PromptTemplate
from langchain.llms import  GooglePalm



def gen_rationale_lable(input):
    prompt_context = """
        For the provided input statement, generate a JSON object only. This object should include two key fields: 'classification' and 'rationale'. The 'classification' field should be determined as 'Enquiry' if the input is a question or request for information, and 'Non-Enquiry' if it is not. The 'rationale' field should provide a detailed, step-by-step thinking(Step 1,Step 2 … Step N) of why the given input was classified as such.

        Input: Summarize in one sentence this article about a famous song.
        Output: {{
        "classification": "Enquiry",
        "rationale": {{
        "Step 1": "Identifying the verb 'Summarize' and the preposition 'in' suggest a request for a specific action to be taken.",
        "Step 2": "The phrase 'this article about a famous song' implies that there's an article that the user wants information about, which is being requested in a condensed form, specifically in one sentence.",
        "Step 3": "Considering these elements, the input can be categorized as a request for information, making it an enquiry."
        }}
        }}
        Input: How do I start running?
        Output: {{
        "classification": "Enquiry",
        "rationale": {{
        "Step 1": "The input starts with the word 'How', which is commonly used to start questions, indicating a request for information.",
        "Step 2": "The phrase 'do I start running?' further indicates a request for information, as the user is seeking advice or instructions.",
        "Step 3": "Given these elements, the input is classified as an 'Enquiry' because it is a question seeking information or advice."
        }}
        }}

        Input: When boiling butter, when it's ready, you can.
        Output: {{
        "classification": "Non-Enquiry",
        "rationale": {{
        "Step 1": "The structure of the sentence suggests it's the beginning or part of an instructional statement or a fact, not seeking any information.",
        "Step 2": "There is no explicit question asked, and it doesn't request any information.",
        "Step 3": "The use of the phrase 'you can' indicates an instruction or advice that is about to be provided, which doesn't qualify as an enquiry.",
        "Step 4": "Based on these observations, the statement is classified as a 'Non-Enquiry'."
        }}
        }}

        Input : When did the First World War start?
        Output: {{
        "classification": "Enquiry",
        "rationale": {{
        "Step 1": "The sentence starts with the word 'When' which is often used in question sentences to ask about the time something happened.",
        "Step 2": "The verb 'did' and the verb 'start' are in past tense, indicating that the user is asking about an event that happened in the past.",
        "Step 3": "The sentence is asking about a specific historical event, 'the First World War', indicating a request for information.",
        "Step 4": "Considering all these points, the sentence is classified as an enquiry as it is a direct question asking for specific historical information."
        }}
        }}

        Input :  Write a short story about a person who discovers a hidden room in their house. The story should include a plot twist and a clear resolution at the end.
        Output: {{
        "classification": "Non-Enquiry",
        "rationale": {{
        "Step 1": "The verb 'Write' at the beginning of the sentence indicates a command or a request for a creative action, not a question or request for specific information.",
        "Step 2": "The rest of the sentence provides context and details for the action being requested, but does not ask for any specific information or clarification.",
        "Step 3": "Therefore, considering these elements, the input can be categorized as a command or request for action, rather than a request for information or a question, making it a non-enquiry."
        }}
        }}
        Input : {input}
        Output: """

    prompt_template = PromptTemplate(
        template= prompt_context,
        input_variables=["input"]
        )
    prompt = prompt_template.format(input=f"{input}")
    llm = GooglePalm() 
    output = llm(prompt)    
    return output


# Process each column
def process_column(column):
    # Generate rational and label for each input
    for i in range(len(dataset)):
        # Check if the input has more than 5 words
        if len(dataset[i][column].split()) > 10:
            input = dataset[i][column]
            rationale_label=gen_rationale_lable(input)  
            print(rationale_label)
            # the rationale and label are separated by follwing json: {"classification": "Non-Enquiry","rationale": {"Step1: ... StepN:"}}
            # load rationale_label as json
            try:
                rationale_label = json.loads(rationale_label)
                classification = rationale_label["classification"]
                rationale = rationale_label["rationale"]
                rationale =  ', '.join(f"{k}: '{v}'" for k, v in rationale.items())
                # remove single and double quote from rationale
                rationale = rationale.translate(str.maketrans("", "", "'\""))
                with open('dataset.jsonl', 'a') as f:
                    f.write(json.dumps({"input": "Classify:"+input ,"target":classification})+"\n")
                    f.write(json.dumps({"input": "Rationale:" +input ,"target":str(rationale)})+"\n")
                time.sleep(2)
            except:
                print("Error: rationale_label is not json, skip")
                continue
        else:
            print("Input has less than 10 words")
            continue

# Dolly
# Process dataset
dataset = load_dataset("databricks/databricks-dolly-15k",split="train[:10]")
# Process each column
columns= ['instruction','response']
for column in columns:
    process_column(column)

#SQUAD
# Process dataset
dataset = load_dataset("squad",split="train[:10]")
# Process each column
columns= ['question','context']
for column in columns:
    process_column(column)
    
