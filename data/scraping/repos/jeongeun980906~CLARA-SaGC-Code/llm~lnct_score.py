import openai
import spacy
import scipy
import random
import numpy as np
import copy
from llm.affordance import affordance_score2
from llm.prompts import get_prompts, PROMPT_STARTER

import time
AGENT_NAME = {
    'cook':"cooking", 'clean':"cleaning", 'mas':"massaging"
}

class lm_planner_unct():
    def __init__(self,type=2, example=False, task= 'cook'):
        self.few_shots = get_prompts(example, task)
        self.new_lines = ""
        self.nlp = spacy.load('en_core_web_lg')
        self.type = type
        self.verbose = True
        self.normalize = False
        self.task = task
        self.objects = []
        self.people = []
        self.floor_plan = []
        self.set_func()
        
    def set_func(self):
        if self.type == 1 or self.type == 3:
            self.plan_with_unct = self.plan_with_unct_type1
            if self.type == 3:
                print("Normalize")
                self.normalize = True
        elif self.type == 2 or self.type == 4:
            self.plan_with_unct = self.plan_with_unct_type2
        elif self.type == 7:
            self.plan_with_unct = self.plan_with_unct_type6
        else:
            raise NotImplementedError
    
    def plan_with_unct_type6(self, verbose = False):
        self.set_prompt()
        action = ""
        # Only one beam search? -> N samples
        while (len(action)) < 3:
            object,_, action,_ = self.inference()
            # print(object_probs,subject_probs)
        
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += "robot action: robot.{}({})\n".format(action, object)
        inp += "robot thought: Can the robot do to this task please answer in yes or no?\nrobot thought: "
        ans = ""
        while len(ans)<2:
            response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=inp,
                    temperature=0.1,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0, stop=['robot thought:','robot action:']
                )
            ans = response['choices'][0]['text']
            print(ans)
        if ans[0] ==" ":
            ans = ans[1:]
        print(ans)
        if 'no' in ans.lower().replace(".","").replace(",",""):
            ood_unct = 1
            amb_unct = 0 
        else:
            ood_unct =  0
            inp = copy.deepcopy(self.prompt)
            inp += self.new_lines
            inp += "robot action: robot.{}({})\n".format(action, object)
            inp += "robot thought: Is this ambiguous and need more information from the user please answer in yes or no?\nrobot thought:"
            ans = ""
            while len(ans)<2:
                response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=inp,
                        temperature=0.5,
                        max_tokens=128,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0, stop='\n'
                    )
                ans = response['choices'][0]['text']
            if ans[0] ==" ":
                ans = ans[1:]
            print(ans)
            if 'yes' in ans.lower().replace(".","").replace(",",""):
                amb_unct = 1 
            else:
                amb_unct =  0
        print({'ood':ood_unct, 'amb':amb_unct, 'total': ood_unct+amb_unct})
        return ['robot action: robot.{}({})'.format(action, object)], [1], {'ood':ood_unct, 'amb':amb_unct, 'total': ood_unct+amb_unct}


    def plan_with_unct_type1(self, verbose= False):
        self.verbose = verbose
        self.set_prompt()
        action = ""
        object = ""
        # Only one beam search? -> N samples
        while (len(action) < 3) and len(object)<3:
            object,object_probs,action, action_probs = self.inference()
            # print(object_probs,action_probs)
        
        obj_entp = 0
        for logprobs in object_probs:
            for logprob in logprobs.values():
                obj_entp += -np.exp(logprob)*logprob
        
        action_entp = 0
        for logprobs in action_probs:
            for logprob in logprobs.values():
                action_entp += -np.exp(logprob)*logprob
        
        # print(self.normalize)
        if self.normalize:
            # print("norm",len(object_probs),len(subject_probs))
            if len(object_probs)>0:
               obj_entp /= len(object_probs)
            if len(action_probs)>0:
                action_entp /= len(action_probs)

        unct= {
            'object' : obj_entp,'action': action_entp,"total": obj_entp+action_entp
        }

        return ['robot action: robot.{}({})'.format(action, object)], [1], unct

    def plan_with_unct_type2(self, verbose= False):
        obj_cand = []
        action_cand = []
        self.verbose = verbose
        goal_num = 5
        inf_num = 3
        if self.type == 4:
            self.set_prompt()
        while len(action_cand)<3:
            print(action_cand)
            for _ in range(goal_num):
                if self.type == 2:
                    self.sample_prompt()
                for _ in range(inf_num):
                    object,_,action,_ = self.inference()
                    print(object,action)
                    if len(action) > 2:
                        obj_cand.append(object)
                        action_cand.append(action)
        tasks = []
        scores = []
        for x,y in zip(obj_cand, action_cand):
            prompt = 'robot action: robot.{}({})'.format(y,x)
            if prompt not in tasks:
                tasks.append(prompt)
                scores.append(1)
            else:
                scores[tasks.index(prompt)] += 1
        scores = [s/sum(scores) for s in scores]
        obj2 = self.get_word_diversity(obj_cand)
        sub2 = self.get_word_diversity(action_cand)
        # print(obj2, sub2)
        unct= {
            'obj' : obj2 /5,
            'sub': sub2/5, 
            'total': (obj2+sub2)/5
        }

        return tasks, scores, unct

    
    def set_goal(self, goal):
        self.goal = goal

    def set_prompt(self,choices=None):
        des = PROMPT_STARTER[self.task]
        des += "Follow are examples"
        if choices == None:
            choices = self.few_shots
        for c in choices:
            des += c
        des += "From this, predict the next action with considering the role of the robot and the ambiguity of the goal\n"
        if self.task =="clean":
            temp = ""
            for e, obj in enumerate(self.floor_plan):
                temp += obj
                if e != len(self.floor_plan)-1:
                    temp += ", "
            des += "objects = [" + temp + "] \n"
        
        if self.task == 'cook' or self.task =="clean":
            temp = ""
            for e, obj in enumerate(self.objects):
                temp += obj
                if e != len(self.objects)-1:
                    temp += ", "
            des += "objects = [" + temp + "] \n"
        
        if self.task == 'mas':
            temp2 = ""
            for e, obj in enumerate(self.people):
                temp2 += obj
                if e != len(self.people)-1:
                    temp2 += ", "
            des += "scene: people = [" + temp2+ "] \n"
        # des += "\n The order can be changed"
        des += "goal: {}\n".format(self.goal)
        if self.new_lines != "":
            des += self.new_lines
        self.prompt = des

    def append_reason_and_question(self, reason, question):
        self.new_lines += '\nrobot thought: this code is uncertain because ' + reason + '\n'
        self.new_lines += 'robot thought: what can I ask to the user? \nquestion: please' + question

    def question_generation(self):
        form = '\nQuestion: I am a {} robot that can only do tasks relevant to it.\
            Considering the possible action sets, Can the robot do {}?'.format(AGENT_NAME[self.task],self.goal)
        form += "\nAnswer: "
        # self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        # inp += self.new_lines
        inp += form
        affor = ""
        while len(affor) < 3:
            try:
                response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=inp,
                        temperature=0.2,
                        max_tokens=128,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0, stop=':'
                    )
                affor = response['choices'][0]['text']#.split('\n')[0]
                print(affor)
            except:
                pass
            # print(inp)
            
        # print(affor)
        temp = affor.lower().replace(".","").replace(",","").split(' ')
        # print(temp)
        if 'no' in temp or 'cannot' in temp or 'can not' in temp or "can't" in temp:
            return None, None, affor
        form =  '\nrobot thought: This is uncertain because'
        # self.new_lines += form
        inp = copy.deepcopy(self.prompt)
        inp += self.new_lines
        inp += form
        while True:
            try:
                response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=inp,
                        temperature=0.5,
                        max_tokens=128,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0, stop=':'
                    )
                break
            except:
                pass
        reason = response['choices'][0]['text'].split('\n')
        if len(reason[0].replace(" ","")) == 0:
            if len(reason[1].replace(" ","")) == 0:
                reason = reason[2]
            else:
                reason = reason[1]
        else:
            reason = reason[0]
        
        reason = reason.replace("robot thought: ","")
        print('reason: ',reason)
        inp += reason
        self.new_lines += reason + '\n'
        ques = 'robot thought: what can I ask to the user? \nquestion: please'
        inp += ques
        self.new_lines += ques
        while True:
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=inp,
                    temperature=0.5,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0, stop='\n'
                )
                break
            except:
                pass
        ques = response['choices'][0]['text']
        ques = ques.split('\n')[0]
        print('question: please',ques)
        self.new_lines += ques
        return reason, ques, affor
    
    def answer(self, user_inp):
        self.new_lines += '\nanswer:' + user_inp
        self.new_lines += '\nrobot thought: continue the previous task based on the question and answer'


    def sample_prompt(self):
        lengs = len(self.few_shots)
        # print(lengs)
        k = random.randrange(3,lengs+1)
        A = np.arange(k)
        A = np.random.permutation(lengs)
        # print(A, k)
        choices = []
        for i in range(k):
            choices.append(self.few_shots[A[i]])
        # print(choices)
        if self.verbose:
            print('select {} few shot prompts'.format(k))
        # choices = self.few_shots#[:4]
        random.shuffle(self.objects)
        random.shuffle(self.people)
        random.shuffle(self.floor_plan)
        self.set_prompt(choices)
        # print(self.prompt)
        # self.set_prompt()

    def inference(self):
        # print(self.prompt)
        while True:
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=self.prompt,
                    temperature=0.8,
                    max_tokens=128,
                    top_p=1,
                    frequency_penalty=0.2,
                    presence_penalty=0,logprobs = 5, stop=")"
                )
                break
            except:
                time.sleep(1)
                continue
        logprobs = response['choices'][0]['logprobs']['token_logprobs']
        tokens = response['choices'][0]['logprobs']['tokens']
        top_logprobs = response['choices'][0]['logprobs']['top_logprobs']
        print(response['choices'][0]['text'])
        action_probs = []
        flag2 = False
        flag = False
        object = ""
        action = ""
        object_prob = 0
        object_probs = []
        object_num = 0
        action_num = 0
        # print(tokens)
        for prob, tok, top_logprob in zip(logprobs[1:], tokens[1:],top_logprobs[1:]):
            if tok == ' done' or tok == 'done':
                return 'done', [{'none':1}],'done', [{'none':1}]
            if tok == "," or tok == ", ":
                flag = False
            if tok == "(":
                flag = True
                flag2  = False
                continue
            # print(tok)
            if tok == ".":
                flag2 = True
            elif flag and not flag2:
                # print(top_logprob)
                object_probs.append(top_logprob)
                object_prob += prob
                object += tok
                object_num += 1
            elif flag2 and not flag:
                # print(top_logprob)
                action_probs.append(top_logprob)
                action += tok
                action_num += 1
            # if tok == '\n':
            #     break
        if object_num != 0:
            object_prob /= object_num
        else:
            object_prob = 0
            object_probs = []
        if self.verbose:
            print(object,object_prob)
        # print(object_probs,subject_probs)
        return object,object_probs, action, action_probs
        
    def append(self, object, action, task=None):
        if task == None:
            next_line = "\n" + "    robot action: robot.{}({})".format(action, object)
        else:
            next_line = "\n    " + task
        self.new_lines += next_line

    def get_word_diversity(self, words):
        vecs = []
        size = len(words)
        for word in words:
            vec = self.nlp(word).vector
            vecs.append(vec)
        vecs = np.vstack(vecs)
        dis = scipy.spatial.distance_matrix(vecs,vecs)
        div = np.sum(dis)/((size)*(size-1))
        # print(div, dis)
        return div
    def reset(self):
        self.new_lines = ""

    def infer_wo_unct(self, task = None, stop=True):
        done = False
        max_tasks=3
        cont = 0
        ask_flag = False
        res = []
        for _ in range(10):
            print("iter")
            self.set_prompt()
            if cont > max_tasks or done:
                break
            # print(self.prompt)
            response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=self.prompt,
                        temperature=0.8,
                        max_tokens=128,
                        top_p=1,
                        frequency_penalty=0.2,
                        presence_penalty=0,logprobs = 5
                    )
            text = response['choices'][0]['text']
            text = text.split('\n')
            for line in text:
                line = line.lower()
                print(line)
                if 'done' in line:
                    done = True
                    break
                if "robot action: robot." in line:
                    cont += 1
                    # try: aff = affordance_score2(line, task)
                    # except: aff = 0
                    # else:
                    res.append(line)
                    self.append(None, None, line)
                elif "robot thought:" in line:
                    res.append(line)
                    self.append(None, None, line)
                elif "question:" in line:
                    res.append(line)
                    self.append(None, None, line)
                    if stop:
                        done = True
                        ask_flag = True
                        break
        return res, ask_flag