import openai
import re
import os
import json

def create_prompt_qgen(question):
    q_gen_prompt = f'''Q: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years
    old, how old is Kody?
    A: To answer the question [How old is Kody?], we need to know: [How old is Mohamed?], [How
    old was Mohamed four years ago?], [How old was Kody four years ago?]
    Q: {question}
    A:To answer the question '''
    return q_gen_prompt

def find_subquestions(question):
    # breakpoint()
    p = create_prompt_qgen(question)
    ans = get_answer(p)
    ## Find all the text inside double quotes 
    subquestions = []
    for i in range(len(ans)):
        if ans[i] == '[':
            j = i+1
            while ans[j] != ']':
                j += 1
            subquestions.append(ans[i+1:j])
    # breakpoint()
    return subquestions

def get_answer(prompt):
        openai.api_key = "060db6b6c3ff468ca2215e0ef75b9cc1"
        openai.api_base = "https://kglm.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15' # this may change in the future
        deployment_name='kglm-text-davinci-003'
        # breakpoint()
        try:
            response = openai.Completion.create(engine=deployment_name, prompt= prompt, temperature=0, max_tokens=512)
            return response["choices"][0]["text"]
        except:
            return "[[Answer: 33]]"

def extract_answer(answer):
    # extract the number inside [[]]
    answer = re.findall(r'\[\[(.*?)\]\]', answer)
    for ans in answer:
        return ans[7:]
    return "33"
    
def answer_with_subquestions(question, subquestions):
    # breakpoint()
    final = " "
    for i in range(1, len(subquestions)):
        subquestion = subquestions[i]
        p = f'''Q: {question} \n Q:{subquestion} \n A: The answer is, '''
        sol = get_answer(p)
        final += f"Q: {subquestion} \n A: {sol} \n"
    # breakpoint()
    final += f"The answer is among the options given in the question, The answer is ony among \"A\",\"B\",\"C\",\"D\", and \"E\". Final answer should be reported in brackets, for example [[Answer: B]], Q: {question} \n A: "
    final_answer = get_answer(final)
    answer = extract_answer(final_answer)
    return answer

def create_dataset_MATH(directory, t):
    Questions = []
    Reasoning = []
    Answers = []
    Tag = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                if(extract_answer_math(data["solution"]).isdigit()):
                    # Extract problem and solution from the JSON data
                    Questions.append(data["problem"])
                    rs = data["solution"].split('.')
                    step = ""
                    for r in rs:
                        step+=r+'\n'
                    Reasoning.append(step[:-2])
                    Answers.append(extract_answer_math(data["solution"]))
                    Tag.append(t)


    return Questions, Answers, Tag

def extract_answer_math(sentence):
    # Example string
    string = sentence

    # Regular expression pattern to match content inside curly braces
    pattern = r"boxed\{(.*?)\}."

    # Find all matches of the pattern in the string
    matches = re.findall(pattern, string)
    # print(sentence)
    # Print the extracted contents
    for match in matches:
      
        # print(match)
        return match
    return ""

def create_dataset_AQuA(filename, bs):
    Questions = []
    Answers=[]
    Rationale = []
    Answers = []
    Tag = []
    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        question = result['question']
        rs = result["rationale"]
        step = ""
        for r in rs:
            step+=r+'\n'
        Rationale.append(rs)
        # Answers.append(step)
        options = result['options']
        for option in options:
            question += " " + option + " "
        # Questions.append(result['question'])
        # Questions.append(result['problem'])
        Questions.append(question)
        # Answers.append(extract_answer(result['answer']))
        Answers.append(result['correct'])
        Tag.append("AQuA")
    return Questions, Answers, Tag


Q_pnc, A_pnc, T_pnc = create_dataset_MATH("./alpaca-lora/MATH/test/counting_and_probability", "PNC")
Q_NT, A_NT, T_NT = create_dataset_MATH("./alpaca-lora/MATH/test/number_theory", "NT")
Q_ialg, A_ialg, T_ialg = create_dataset_MATH("./alpaca-lora/MATH/test/intermediate_algebra", "IALG")
Q_alg, A_alg, T_alg = create_dataset_MATH("./alpaca-lora/MATH/test/algebra", "ALG")
Q_AQuA, A_AQuA, T_AQuA = create_dataset_AQuA("./alpaca-lora/AQuA/test.json", 1)

Q =  Q_alg[125:250] + Q_ialg[125:250] + Q_NT[125:250] + Q_pnc[125:250]
A =  A_alg[125:250] + A_ialg[125:250] + A_NT[125:250] + A_pnc[125:250]
T =  T_alg[125:250] + T_ialg[125:250] + T_NT[125:250] + T_pnc[125:250]
Q = Q_AQuA
A = A_AQuA
T = T_AQuA
breakpoint()
acc={"PNC":40, "NT":28, "IALG":21, "ALG":28, "AQuA":0}


for i in range(len(Q)):
    question = Q[i]
    answer = A[i]
    tag = T[i]
    subquestions = find_subquestions(question)
    final_answer = answer_with_subquestions(question, subquestions)
    if(final_answer.strip() == answer.strip()):
        acc[tag] += 1
    
    print(i,acc)

    