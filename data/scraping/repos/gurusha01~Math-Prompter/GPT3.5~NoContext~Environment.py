import torch
import re
from RewardFunction import reward1, reward2, reward3, reward2_cot, reward3_cot
import time
import difflib
from transformers import GenerationConfig
import openai

def clean_code(code):
    lines = code.split('\n')
    indented_code = []
    for line in lines:
        if line.strip().startswith(('def solution():', 'solution()')):
            indented_code.append(line.strip())
            continue

        # Replace underscores with underscores
        line = line.replace('\\', '')

        # Add indentation
        line = ' ' * 2 + line

        indented_code.append(line)

    return '\n'.join(indented_code)



class Reward:
    def __init__(self):
        self.bs = 1

    
    def extract_questions(self, text):
        # Define the regular expression pattern for questions
        pattern = r"(?:^|\s)[\w\s\d\W]+?\?"

        # Find all matches of the pattern in the text
        matches = re.findall(pattern, text)

        # Filter out any non-question matches
        questions = [match.strip() for match in matches if '?' in match]

        return questions
        
    def extract_comments(self, code_string):
        pattern = r"#.*"  # Regular expression pattern to match comments
        comments = re.findall(pattern, code_string)
        combined_comment = ""
        for comment in comments:
            combined_comment += comment[1:]
        return combined_comment

    def step(self, reset, actions, questions, answers, dones_prev):
        with torch.no_grad():
            if(not reset):
                set_of_prompts = []
                for action in actions:
                    questions_extracted = self.extract_questions(action)
                    # print(questions_extracted)
                    if(len(questions_extracted)>2):
                        set_of_prompts.append(questions_extracted)
                    else:
                        set_of_prompts.append([])



            contexts = []
            for question in questions: 
                contexts.append((question.rsplit("?",1)[0]).rsplit('.', 1)[0])

            input = []
            for j in range(len(questions)):
                if(reset):
                    input.append([contexts[j], questions[j], []])
                else:
                    input.append([contexts[j], questions[j], set_of_prompts[j]])
            
            # breakpoint()
            dones, outputs, rewards = self.generate_env(input, answers, dones_prev)
            comments = []
            # for output in outputs:
            #     comments.append(self.extract_comments(output))
            
            return dones, outputs, rewards, comments
                    

    def generate_env(self, input, answers, dones_prev):
        # breakpoint()
        # Get reward based on Subquestion
        rewards1 = reward1(input)
        # breakpoint()
        # Reward based on subanswers
        rewards2 = []

        # Generate sub_question and question answers
        outs = []
        for i in range(len(input)):
                prompts = []
                questions = []
                for j in range(len(input[i][2])):
                    questions.append(input[i][0] + input[i][2][j])
                    prompts.append("\n Question:"+input[i][0] + input[i][2][j] + "?\n Answer: \n Let us think step by step")
                sol_till_now = ""
                out_s = []
                for k in range(len(prompts)):
                        out=self.generate([prompts[k]])[0]
                        out_s.append(out)
                        sol_till_now += prompts[k] + out
                if(len(prompts)>0):
                    rewards2.append(reward2_cot(questions, out_s))
                if(len(prompts))==0:
                    rewards2.append(0)

                outs.append(sol_till_now + "\n Question:" + input[i][1] +"?\n Answer: \n Let us think step by step")
            
            # print("######OUTS:",outs)
        # print("OUTS:", outs[0])
        # Generate Question answer
        # breakpoint()
        s= time.time()
        outputs = self.generate(outs)
        e = time.time()
        # print("OUTPUTS:", outputs[0])
        # print("TIME INSIDE ENVIRONMENT GENERATE: ", e-s)
        # outputs = [" The largest number possible in the top cell is 9 and the smallest number possible in the top cell is 1. Therefore, the difference between the largest and smallest numbers possible in the top cell is 8. [[Answer: 8]]", ".\n \n First digit can be any of 3, 4, 5 and 6.\n Second digit can be any of 3, 4, 5 and 6 except the one chosen for the first digit.\n Third digit can be any of 3, 4, 5 and 6 except the ones chosen for the first and second digits.\n \n Therefore, the number of different three-digit odd numbers that can be formed using the digits 3, 4, 5 and 6 if digits cannot be repeated is [[Answer: 12]]"]
        
        # Get final reward
        dones, rewards3 = reward3_cot(answers, outputs, dones_prev, rewards1, rewards2)

        final_reward = rewards3


        return dones, outputs, final_reward
    

    def generate(self, input_prompt):
        
        p = '''
        Read the instructions given and answer the questions that follow.
        1. Use variables for unknown quantities.
        2. Options are given in each question, The answer should be from the options "A", "B", "C", "D" or "E"
        3. Final answer should be reported inside brackets example [[B]].        
        '''

        openai.api_key = "060db6b6c3ff468ca2215e0ef75b9cc1"
        openai.api_base = "https://kglm.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15' # this may change in the future
        deployment_name='kglm-text-davinci-003'
        # breakpoint()
        result = []
        for input in input_prompt:
            prompt = p + input
            try:
                response = openai.Completion.create(engine=deployment_name, prompt= prompt, temperature=0, max_tokens=512)
                result.append(response["choices"][0]["text"])
                # result.append("[[Answer: C]]")
            except:
                result.append("[[Answer: false]]")
            print("RESPONSE",input_prompt, response["choices"][0]["text"])
        # result = ["[[E]]", "[[Answer: A]]"]
        return result


    
    