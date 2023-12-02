import openai
from chatgpt import *
import copy
from . import utils
# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo", 
#   messages=[{"role": "user", "content": "test"},{"role": "system", "content": "randomly say something!"}],
#   temperature=1.5,
# )
weights=[0.3,0.3,0.4]

trivial_prompt = ""

skillMatchDict={
    "model":"gpt-3.5-turbo",
    
    "messages":[{"role": "system", "content": "You are a helpful career planning assistant. You will tell user how much he could be accpted(in percentage). And tell him what else he can do if the percentage is below 0.5. And the average score of the skills you give should be 0.5. Here are some history data for you to refer to."},
        {"role" : "user",  "content": "This is an example output,please follow this format: Calculated success rate of Microsoft is 0.1. Based on your description of not having any computer knowledge, the chances of being accepted at Microsoft are quite low.\n\nHowever, I can suggest some alternative career options with a higher probability of acceptance:\n\n1. Sales Representative: Success rate 0.65\n2. Marketing Coordinator: Success rate 0.7\n3. Administrative Assistant: Success rate 0.75\n4. Customer Service Representative: Success rate 0.6\n\nThese are just a few options to consider based on your current skillset. It's important to remember that gaining computer knowledge and skills can significantly increase your chances of being accepted into the tech industry."},
        {"role": "user", "content": "I want to apply for a job, can you tell me how much possibility I cloud be accepted. And what else can I do ? You can directly reply to my questions without any explanation."},
        {"role": "assistant", "content": "Sure. Please Describe yourself and the job you are going to acquire. I will give you the chance you will be accepted and provide you with the jobs you also could be suit for.."},
        {"role": "user", "content": "I want to apply for a delivery man. I major in computer science and I know many programming languages. I have a github account and made many contributions."},
        {"role": "assistant", "content": "Although you are good in computer science I don't think you are suitble for this because this is a job that require your energy and strength. This is a relatively easy job so your success rate is 0.5, if you apply for a sports man your success rate is 0.01"},
        {"role": "system", "content": "You are a HR and you need to preview whether the applicant's skills are useful for the job. Nonsense or too much garbage informations will make you feel bad. You need to give a score between 0 and 1 to the applicant's skills. Output as such a dictionary format, DO output as such ONE dictionary format. You should put anything else in the 'reply':{'score':0.xx, 'reply': (Your reply as a HR)}"}],
    "temperature":0.1,
}


class JobFindingGPT:
    def __init__(self, API_KEY, flush=False,frequency_penalty=1.0,presence_penalty=1.0):
        openai.api_key = API_KEY

        self.flush = flush
        self.frequency_penalty=frequency_penalty
        self.presence_penalty=presence_penalty
        self.previous_questions=[] # 4 questions
        self.goals = {} # goal is a dictionary of {goal prompt:finish percent}
        self.job_description = ""
        self.final_score = 0
        self.skills = ""

    def main(self):
        print("What job are you looking for?");
        job_description = input();
        self.job_description = job_description
        self.part1();

    def part1(self):
        print("List your skills");
        self.skills = input();
        m_skill_dict = copy.deepcopy(skillMatchDict)
        m_skill_dict["messages"].append({"role": "user", "content": ("The applicant is looking for a job as a "+self.job_description)})
        m_skill_dict["messages"].append({"role": "user", "content": ("He or she has the skills or characteristic below:"+self.skills)})
        skill_ret = self.send_prompt(m_skill_dict)
        skill_dict = utils.string_to_dict(skill_ret)
        self.final_score += skill_dict["score"]*weights[0]
        

    def part2(self):
        print("give me a copy of your resume")

    def send_prompt(self,messageDict):
        streaming = False
        if(self.flush):
            streaming = True
        completion = openai.ChatCompletion.create(
            model=messageDict["model"],
            messages=messageDict["messages"],
            temperature=messageDict["temperature"],
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stream=streaming,
        )
        if(self.flush):
            collected_messages = []
            # iterate through the stream of events
            for chunk in completion:
                try:
                    chunk_message = chunk['choices'][0]['delta']['content']
                    collected_messages.append(chunk_message)  # save the message
                    print(chunk_message,end="")  # print the delay and text
                except:
                    pass
            collected_messages = "".join(collected_messages)
        else:
            collected_messages = completion.choices[0].text
            print(collected_messages)
        return collected_messages
    
    def get_job(self, job_description):

        pass

    def compare_resume(self, user_resume, generated_resume):
        pass

    def simulate_interview(self, user_answer, correct_answer):
        pass

    def calculate_success_rate(self):
        pass

    def start_simulation(self, job_description, user_resume, user_answer, correct_answer):
        pass
