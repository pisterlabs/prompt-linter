import logging
import json
import openai
from openai import OpenAI , AzureOpenAI
import re
from pathlib import Path  
import os
from sentence_transformers import SentenceTransformer, util
import random

class summary():

    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def __init__(self,logger = None, prompt_dir="./doc", api_json = None, prompt_output = False):
        self.max_fail_cnt = 3
        self.token_used = 0
        self.logger = logger
        self.prompt_template : dict[str , str] = None
        self.example : dict[str , str] = None
        self.player_name = None
        self.prompt_output = prompt_output
        self.all_game_info = {
            "self_number" : "",
            "self_role" : "",
            "all_role_info" : "",
            "result" : "",
        }
        self.memory_stream = {} 
        self.operation_info = {}
        self.guess_role = {}
        self.chat_func = None
        self.chinese_to_english = {
            # summary
            "投票總結" : "vote_summary",
            "發言總結" : "dialogue_summary",
            "技能總結" : "operation_summary",
            "猜測角色總結" : "guess",
            "目前總結" : "current"
        }
        self.operation_to_chinese = {
            "seer" : "預言家查驗，目標是",
            "witch" : "女巫的技能，目標是",
            "village" : "村民",
            "werewolf" : "狼人殺人，目標是",
            "werewolf_dialogue" : "狼人發言，想要殺掉",
            "hunter" : "獵人獵殺，目標是"
        }

        self.role_to_chinese = {
            "seer" : "預言家",
            "witch" : "女巫",
            "village" : "村民",
            "werewolf" : "狼人",
            "hunter" : "獵人"
        }

        
        self.prompt_dir = Path(prompt_dir)
        self.__load_prompt_and_example__(self.prompt_dir)
        
        # openai api setting
        self.api_kwargs = {}
        try:
            self.__openai_init__(api_json)
        except:
            raise Exception("API Init failed")
        
        self.summary_limit = 50
        self.similarly_sentence_num = 5
        self.get_score_fail_times = 3
        

        if not os.path.exists(os.path.join(prompt_dir, "summary")):
            os.mkdir(os.path.join(prompt_dir, "summary"))

    def __load_prompt_and_example__(self , prompt_dir):
        """load prompt json to dict"""

        with open(prompt_dir / "./prompt/summary/common_prompt.json" , encoding="utf-8") as json_file: self.prompt_template = json.load(json_file)
        with open(prompt_dir / "./prompt/summary/common_example.json" , encoding="utf-8") as json_file: self.example = json.load(json_file)

        for key , prompt_li in self.prompt_template.items():
            self.prompt_template[key] = '\n'.join(prompt_li)
        for key , prompt_li in self.example.items():
            self.example[key] = '\n'.join(prompt_li)

    def __openai_init__(self , api_json):
        """azure openai api setting , can override this"""
        with open(api_json,'r') as f : api_info = json.load(f)

        if api_info["api_type"] == "azure":
            # version 1.0 of Openai api
            self.client = AzureOpenAI(
                api_version = api_info["api_version"] ,
                azure_endpoint = api_info["api_base"],
                api_key=api_info["key"],
            )

            # legacy version
            openai.api_key = api_info["key"]
            openai.api_type = api_info["api_type"]
            openai.azure_endpoint = api_info["api_base"]
            openai.api_version = api_info["api_version"] 

            self.api_kwargs["model"] = api_info["engine"]
        else:
            self.client = OpenAI(
                api_key=api_info["key"],
            )
            # legacy version
            openai.api_key = api_info["key"]

            self.api_kwargs["model"] = api_info["model"]

    def __openai_send__(self , prompt):
        # """openai api send prompt , can override this."""

        ### version 1.0 of Openai api ###
        response = self.client.chat.completions.create(
            **self.api_kwargs,
            messages = [
                {"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
    
        self.token_used += response.usage.total_tokens
        return response.model_dump()['choices'][0]['message']['content']
    
    def __process_LLM_output__(self , prompt , keyword_list , sample_output):
        """
        communication with LLM , repeat {self.max_fail_cnt} util find the {keyword_list} in LLM response .
        return the {keyword_list} dict , if fail get {keyword_list} in LLM response , return {sample_output}.
        """
        success_get_keyword = False
        fail_idx = 0

        info = {}

        while not success_get_keyword and fail_idx < self.max_fail_cnt:

            info = {}
            result = self.__openai_send__(prompt)
            # print(f"prompt = {prompt}")
            # print(f"result = {result}")
            
            # result block by openai
            if result == None:
                fail_idx+=1
                continue
            

            splited_result = result.split('\n')
            keyword_name = ""
            for line in splited_result:
                # get keyword like [XXX]
                keyword = re.search('\[(.*)\]', line)
                if keyword != None and keyword.group(1) in self.chinese_to_english.keys():
                    keyword_name = self.chinese_to_english[keyword.group(1)]
                    info[keyword_name] = ""
                elif keyword_name != "":
                    info[keyword_name] += line + "\n"

            if all(_ in info.keys() for _ in keyword_list): success_get_keyword = True
            else : fail_idx+=1
        
        
        if fail_idx >= self.max_fail_cnt: info = None
        
        return info
    
    def __process_user_role(self, data):
        
        role_info = ""
        for idx, key in enumerate(data):
            role_info += f"{idx}. {idx}號玩家({data[key]['user_name']})是{self.role_to_chinese[data[key]['user_role']]}\n"
        
        return role_info
    
    def __process_guess_role(self, stage, data):
        
        guess_info = ""
        for idx, role in enumerate(data['guess_role']):
            num = str(idx)
            guess_info += f"{idx}. {self.player_name[num]['user_name']}({idx})可能是{role}\n"

        day = stage.split('-')[0]
        if day != "check_role":
            self.guess_role[day] = guess_info

    def __process_random_guess_role(self, stage, data):
        
        all_role = ["狼人", "狼人", "女巫", "預言家", "獵人", "村民", "村民"]
        random.shuffle(all_role) 

        guess_info = ""
        for idx, role in enumerate(data['guess_role']):
            num = str(idx)
            guess_info += f"{idx}. {self.player_name[num]['user_name']}({idx})可能是{all_role[idx]}\n"

        day = stage.split('-')[0]
        if day != "check_role":
            self.guess_role[day] = guess_info


    def __memory_stream_push(self, stage, ob):
        day = stage.split('-')[0]
        if day in self.memory_stream.keys():
            self.memory_stream[day] += ob
        else:
            self.memory_stream[day] = ob

    def __operation_info_push(self, stage, ob):
        day = stage.split('-')[0]
        if day in self.operation_info.keys():
            self.operation_info[day] += ob
        else:
            self.operation_info[day] = ob

    
    def __process_announcement__(self , data):
        """add announcement to memory stream"""
        announcement = data['announcement']

        if any(data["vote_info"].values()) :
            self.__push_vote_info__(data["vote_info"] , data["stage"])

        for anno in announcement:
            ob = ""
            if len(anno['user']) > 0:
                player = str(anno['user'][0])
            if anno["operation"] == "chat":
                if data['stage'].split('-')[-1] == "werewolf_dialogue":
                    ob = "目前是狼人發言，只有狼人可以看到。"
                ob += f"{player}號玩家({self.player_name[player]['user_name']})說「{anno['description']}」\n"    
            elif anno["operation"] == "died":
                ob = f"{player}號玩家({self.player_name[player]['user_name']})死了\n"    
            
            self.__memory_stream_push(data["stage"], ob)

    def __info_init(self, stage):

        if stage.split('-')[0] != "check_role":

            day = str(int(stage.split('-')[0]))

            if day not in self.memory_stream.keys():
                self.memory_stream[day] = f"你本場的身分為{self.all_game_info['self_role']}\n你是{self.all_game_info['self_number']}號玩家\n"
            if day not in self.operation_info.keys():
                self.operation_info[day] = f"你本場的身分為{self.all_game_info['self_role']}\n你是{self.all_game_info['self_number']}號玩家\n"
            if day not in self.guess_role.keys():
                self.guess_role[day] = f"你本場的身分為{self.all_game_info['self_role']}\n你是{self.all_game_info['self_number']}號玩家\n"

    def __push_vote_info__(self , vote_info : dict , stage):
        """add vote info to memory stream"""
        prefix = "狼人投票殺人階段:" if stage.split('-')[-1] == "seer" else "玩家票人出去階段:"

        ob = ""
        for player , voted in vote_info.items():
            if voted != -1:
                ob += f"{prefix} {player}號玩家({self.player_name[player]['user_name']})投給{voted}號玩家({self.player_name[str(voted)]['user_name']})\n"
            else:
                ob += f"{prefix} {player}號玩家({self.player_name[player]['user_name']})棄票\n"

        self.__memory_stream_push(stage, ob)

    def __set_player2identity(self):
        
        self.player2identity = {}

        for number, info in self.player_name.items():
            user_name = info["user_name"]
            self.player2identity[f"玩家{number}號"] = number
            self.player2identity[f"玩家{number}"] = number
            self.player2identity[f"{number}號玩家"] = number
            self.player2identity[f"{number}號"] = number
            self.player2identity[f"{user_name}({number})"] = number
            self.player2identity[user_name] = number

    def __load_game_info(self, random_guess_role, file_path = None, game_info = None):       

        self.all_game_info = {
            "self_number" : "",
            "self_role" : "",
            "all_role_info" : "",
            "result" : "",
        }
        self.memory_stream = {} 
        self.operation_info = {}
        self.guess_role = {}

        if file_path != None:
            with open(self.prompt_dir / file_path, encoding="utf-8") as json_file: game_info = [json.loads(line) for line in json_file.readlines()]
        for val in game_info[0].values():
            self.my_player_role = val
        
        self.player_name = game_info[1]
        self.all_game_info["self_number"] = list(game_info[0].keys())[0]
        self.all_game_info["self_role"] = self.role_to_chinese[list(game_info[0].values())[0]]
        self.all_game_info["all_role_info"] = self.__process_user_role(game_info[1])
        
        self.__set_player2identity()

        no_save_op = ["dialogue", "vote1", "vote2", "check"]
        for idx, info in enumerate(game_info):

            # stage info
            if "stage" in info.keys():
                self.__info_init(info["stage"])
                self.__process_announcement__(info)

                if "guess_role" in game_info[idx+1].keys():
                    if random_guess_role:   self.__process_random_guess_role(info["stage"] , game_info[idx+1])
                    else:   self.__process_guess_role(info["stage"] , game_info[idx+1])

            # operation
            elif "stage_name" in info.keys() and (not info['stage_name'].split('-')[-1] in no_save_op):
                self. __info_init(info["stage_name"])
                ob = f"你使用了{self.operation_to_chinese[info['stage_name'].split('-')[-1]]}{info['target']}號玩家\n"
                self.__operation_info_push(info["stage_name"], ob)
                
                if "guess_role" in game_info[idx+1].keys():
                    self.__process_guess_role(info["stage_name"], game_info[idx+1])

        for anno in game_info[-2]["announcement"]:
            if anno["operation"] == "game_over":
                self.all_game_info["result"] = anno["description"]

        # self.__print_game_info()

    def __print_game_info(self):

        print(self.all_game_info)
        for i in range(1, len(self.memory_stream)+1):
            day = str(i)
            print(f"第{day}天")
            print(f"memory_stream: {self.memory_stream[day]}")
            print(f"operation_info: {self.operation_info[day]}")
            print(f"guess_role: {self.guess_role[day]}")


    def get_summary(self, file_name = "11_06_18_31_mAgent112.jsonl"):
            
        self.__load_game_info(random_guess_role = True ,file_path = f"./game_info/{file_name}")
        print(f"loading file_path: ./game_info/{file_name}")
        for day in self.memory_stream: 
            
            vote_summary = self.__get_vote_summary__(day, self.memory_stream[day], self.all_game_info["result"])
            dialogue_summary = self.__get_dialogue_summary__(day, self.memory_stream[day], self.all_game_info["result"])
            operation_summary = self.__get_operation_summary__(day, self.memory_stream[day], self.operation_info[day], self.all_game_info["result"])
            guess_role_summary = self.__get_guess_role_summary(day, self.memory_stream[day], self.guess_role[day])
            
            """summary + score"""            
            self.set_score(self.my_player_role, "vote", vote_summary)
            self.set_score(self.my_player_role, "dialogue", dialogue_summary)
            self.set_score(self.my_player_role, "operation", operation_summary)
            self.set_score(self.my_player_role, "guess_role", guess_role_summary)
        print("Finish")

    def __get_operation_summary__(self, day, day_memory, day_operation, result):
        """day summary to openai"""

        day = f"第{day}天"    
        self.max_fail_cnt = 3
    
        final_prompt = self.prompt_template['operation_summary'].replace("%l" , self.example['operation_summary']).replace("%z", day).replace("%m" , day_memory).replace("%o" , day_operation).replace("%y" , self.all_game_info["all_role_info"]).replace("%p" , result)
        sample_info = {'operation_summary' : "operation_summary_result"}        
        info = self.__process_LLM_output__(final_prompt , ['operation_summary'] , sample_info)
        if info == None:
            return None
        
        if self.prompt_output:
            print(f"day_summary_prompt: {final_prompt}")
            print(f"operation_summary_result: {info['operation_summary']}")

        return info['operation_summary']
    
    def __get_dialogue_summary__(self, day, day_memory, result):
        """dialogue summary to openai"""

        day = f"第{day}天"    
        self.max_fail_cnt = 3
    
        final_prompt = self.prompt_template['dialogue_summary'].replace("%l" , self.example["dialogue_summary"]).replace("%z", day).replace("%m" , day_memory).replace("%y" , self.all_game_info["all_role_info"]).replace("%p" , result)
        sample_info = {"dialogue_summary" : "dialogue_summary_result"}        
        info = self.__process_LLM_output__(final_prompt , ["dialogue_summary"] , sample_info)
        if info == None:
            return None
        
        if self.prompt_output:
            print(f"dialogue_summary_prompt: {final_prompt}")
            print(f"dialogue_summary_result: {info['dialogue_summary']}")

        return info["dialogue_summary"]
    
    def __get_vote_summary__(self, day, day_memory, result):
        """dialogue summary to openai"""

        day = f"第{day}天"    
        self.max_fail_cnt = 3

        final_prompt = self.prompt_template['vote_summary'].replace("%l" , self.example["vote_summary"]).replace("%z", day).replace("%m" , day_memory).replace("%y" , self.all_game_info["all_role_info"]).replace("%p" , result)
        sample_info = {"vote_summary" : "vote_summary_result"}        
        info = self.__process_LLM_output__(final_prompt , ["vote_summary"] , sample_info)
        if info == None:
            return None
        
        if self.prompt_output:
            print(f"vote_summary_prompt: {final_prompt}")
            print(f"vote_summary_result: {info['vote_summary']}")

        return info["vote_summary"]
    
    def __get_guess_role_summary(self, day, day_memory, guess_role):
        """guess_role summary to openai"""

        day = f"第{day}天"    
        self.max_fail_cnt = 3
    
        final_prompt = self.prompt_template['guess_role'].replace("%l" , self.example['guess_role']).replace("%z", day).replace("%m" , day_memory).replace("%g" , guess_role).replace("%y" , self.all_game_info["all_role_info"])
        info = {
            "guess" : "guess_role_summary",
        }        
    
        info = self.__process_LLM_output__(final_prompt , ["guess"] , info)
        if info == None:
            return None
        return info['guess']

    def set_score(self, role, stage, summary):

        if summary == None:
            return
        
        
        trans_summary = self.transform_player2identity(summary= summary)
        
        final_prompt = self.prompt_template["score"].replace("%s", trans_summary)
        response = self.__openai_send__(final_prompt)
        
        try:
            score = int(response.split(":")[1])
        except:
            self.get_score_fail_times -= 1
            if self.get_score_fail_times >= 0:
                self.set_score(role= role, stage= stage, summary= summary)
                return
            else :
                score = 0
        
        file_path = os.path.join(os.path.join("summary", role), f"{stage}.json")
        try:
            summary_set = self.__load_summary(file_path= file_path)
        except:
            summary_set = []
        updated_summary_set = self.__update_summary(summary_set= summary_set, summary= trans_summary, score= score)
    
        self.__write_summary(file_path= file_path, data= updated_summary_set)

    def __load_summary(self, file_path):
        
        with open(self.prompt_dir / file_path, encoding="utf-8") as json_file: summary_set = json.load(json_file)
        return summary_set
    
    def __write_summary(self, file_path, data):

        try:
            
            with open(self.prompt_dir / file_path, "w", encoding='utf-8') as json_file: 
                new_data = json.dumps(data, indent= 1, ensure_ascii=False)
                json_file.write(new_data)
        except:
            file = file_path.split("\\")[0] + "\\" + file_path.split("\\")[1]
            os.mkdir(self.prompt_dir / file)
            self.__write_summary(file_path, data)
        self.get_score_fail_times = 3

    def __update_summary(self, summary_set, summary, score):
        
        summary_set.append({"summary": summary, "score": score})
        summary_set = sorted(summary_set, key= lambda x : x["score"], reverse= True)
        
        if len(summary_set) > self.summary_limit:            
            summary_set.pop()
        return summary_set
    
    def __get_current_summary(self, game_info):

        self.__load_game_info(random_guess_role = False, game_info= game_info)
        final_prompt = ""
    
        try:
            day = str(len(self.memory_stream))
            final_prompt = self.prompt_template['current_summary'].replace("%l" , self.example['current_summary']).replace("%z", day).replace("%m" , self.memory_stream[day]).replace("%o" , self.operation_info[day]).replace("%y" , self.guess_role[day])
        except:
            final_prompt = self.prompt_template['current_summary'].replace("%l" , self.example['current_summary']).replace("%z", "0").replace("%m" , "無").replace("%o" , "無").replace("%y" , "無")

        if self.prompt_output:
            self.logger.debug(f"final_prompt: {final_prompt}")
            
        info = {
            "current" : "current_summary",
        }        
        info = self.__process_LLM_output__(final_prompt , ["current"] , info)        
        return info['current']
    
    def transform_player2identity(self, summary):
    
        for key_word in self.player2identity.keys():
            if key_word in summary:
                
                key_number = self.player2identity[key_word]
                identity = self.role_to_chinese[self.player_name[key_number]["user_role"]]
                summary = summary.replace(key_word, f"{identity}")

        return summary


    def find_similarly_summary(self, stage, game_info):
        
        cur_summary = self.__get_current_summary(game_info= game_info)
        self.logger.debug(f"cur_summary: {cur_summary}")
        file_path = f"./summary/{self.my_player_role}/{stage}.json"
        if not os.path.exists(self.prompt_dir / file_path):
            return "無"
        
        summary_set = self.__load_summary(file_path= file_path)
        summary_set_summary = [_["summary"] for _ in summary_set]

        query_embedding = self.embedding_model.encode(cur_summary , convert_to_tensor=True)
        embeddings = self.embedding_model.encode(summary_set_summary , convert_to_tensor=True)
        cos_sim = util.cos_sim(query_embedding, embeddings)[0]
        similarly_scores= []

        for idx, score in enumerate(cos_sim):
            similarly_scores.append([score.to("cpu").item(), idx])

        similarly_scores = sorted(similarly_scores,key= lambda x: x[0], reverse= True)

        window = min(len(similarly_scores), self.similarly_sentence_num)        
        found_similarly_summary = [summary_set[idx]["summary"] for _, idx in similarly_scores[0: window]]

        self.logger.debug(f"found_similarly_summary: {found_similarly_summary}")

        return found_similarly_summary


    
if __name__ == '__main__':

    s = summary(api_json="./doc/secret/openai.key", prompt_dir="./doc", prompt_output = True)
    for g in range(1,11):
        for i in range(0, 7):
            s.get_summary(file_name= f"./game{g}/game{g}_agent{i}.jsonl")
    
    # memory_stream_agent_script(api_json = "./doc/secret/openai.key", game_info_path = f"./doc/game_info/{game_name}/{game_time}_agent{i}.jsonl", agent_name = f"{role[i]}_{version}" , game_room = game_name)
    # s.find_similarly_summary(stage= "guess_role", game_info= "11_19_23_13_agent3.jsonl")
    