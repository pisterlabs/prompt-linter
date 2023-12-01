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
    def __init__(self, model, tokenizer, batch_size, device, prompt_large, prompt_small, max_len = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.prompt_small = prompt_small
        self.prompt_large = prompt_large
        self. max_len = max_len
        self.model.eval()

    
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

    def step(self, reset, actions, questions, answers, dones_prev, T):
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
            dones, outputs, rewards = self.generate_env(input, answers, dones_prev, T)
            comments = []
            # for output in outputs:
            #     comments.append(self.extract_comments(output))
            
            return dones, outputs, rewards, comments
                    

    def generate_env(self, input, answers, dones_prev, T):

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
                    prompts.append("\n ###Input:"+input[i][0] + input[i][2][j] + "?\n ###Response: \n Let us think step by step")
                sol_till_now = ""
                out_s = []
                for k in range(len(prompts)):
                        out=self.generate([prompts[k]], T)[0]
                        out_s.append(out)
                        sol_till_now += prompts[k] + out
                if(len(prompts)>0):
                    rewards2.append(reward2_cot(questions, out_s))
                if(len(prompts))==0:
                    rewards2.append(0)

                outs.append(sol_till_now + "\n ###Input:" + input[i][1] +"?\n #Response: \n Let us think step by step")
            
            # print("######OUTS:",outs)
        # print("OUTS:", outs[0])
        # Generate Question answer
        # breakpoint()
        s= time.time()
        outputs = self.generate(outs, T)
        e = time.time()
        # print("OUTPUTS:", outputs[0])
        # print("TIME INSIDE ENVIRONMENT GENERATE: ", e-s)
        # outputs = [" The largest number possible in the top cell is 9 and the smallest number possible in the top cell is 1. Therefore, the difference between the largest and smallest numbers possible in the top cell is 8. [[Answer: 8]]", ".\n \n First digit can be any of 3, 4, 5 and 6.\n Second digit can be any of 3, 4, 5 and 6 except the one chosen for the first digit.\n Third digit can be any of 3, 4, 5 and 6 except the ones chosen for the first and second digits.\n \n Therefore, the number of different three-digit odd numbers that can be formed using the digits 3, 4, 5 and 6 if digits cannot be repeated is [[Answer: 12]]"]
        
        # Get final reward
        dones, rewards3 = reward3_cot(answers, outputs, dones_prev, rewards1, rewards2)

        final_reward = rewards3


        return dones, outputs, final_reward
    

    def generate(self, input_prompt, T):
        p = '''
        Below is an instruction that describes a task, paired with an input and a reasoning that provides further context. Write a response that appropriately completes the request.

        ### Instruction: solve the following question, the final answer is a number. Report the final answer inside brackets as [[Answer: 33]]. 
                    
        '''
        # breakpoint()
        
        for i in range(len(input_prompt)):
            input_prompt[i] = p + self.prompt_large + input_prompt[i]
        
        generation_config = GenerationConfig(
            temperature=T,
            top_p = 0.18
        )

        input_ids = self.tokenizer(input_prompt, return_tensors="pt", padding=True, truncation = True, max_length = self.max_len).input_ids.to(self.device)
        outputs=self.model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=128,
        )
        result =[]
        for i in range(len(outputs.sequences)):

            result.append(self.tokenizer.decode(outputs.sequences[i][input_ids[i].shape[0]:])) 
        print(result)
        
        # result = self.remove_common_prefix(input_prompt, result)
        # print(result[0])
        # print("TIME TO GENERATE SECOND PASS: ", e-s)
        return result



    
    