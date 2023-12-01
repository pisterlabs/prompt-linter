import requests
import threading
import logging
from openai import OpenAI , AzureOpenAI
import sys   
from pathlib import Path   
import time
import json
import numpy as np
import re
import time
import math
from sentence_transformers import SentenceTransformer, util

class long_memeory_stream():
    
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def __init__(self , prompt_dir , logger , client , openai_kwargs , summary=False , log_prompt = False):
        self.memory_stream = []
        self.openai_kwargs = openai_kwargs
        self.client : OpenAI | AzureOpenAI = client
        self.logger : logging.Logger = logger
        self.log_prompt = log_prompt
        self.max_fail_cnt = 3
        self.token_used = 0
        # used for the llm keyword translate
        self.chinese_to_english = {
            # importantance
            "分數" : "score",
            # reflection question
            "問題" : "question",
            # refection
            "見解" : "opinion",
            "參考見解" : "reference",
            # suspect role list
            "猜測身分" : "role",
            # vote
            "投票" : "vote",
            # dialogue
            "發言" : "dialogue",
            # importantance / suspect role list / vote
            "原因" : "reason",
        }
        self.role_to_chinese = {
            "seer" : "預言家",
            "witch" : "女巫",
            "village" : "村民",
            "werewolf" : "狼人",
            "hunter" : "獵人"
        }
        self.player_num = None
        self.role = None
        self.suspect_role_list : dict[int , str] = {}
        self.know_role_list : dict[int , str] = {}
        self.remain_player = []
        
        self.prompt_dir = prompt_dir
        self.prompt_template : dict[str , str] = None
        self.example : dict[str , str] = None
        
        self.day = 0
        # sample `update_stage` return type
        self.ret_format = {
            "stage_name" : None,
            "operation": None,
            "target" : None,
            "chat" : None
        }

        # gues roles updated flag for get info
        self.guess_roles_updated = 0
        # record the reflection 
        self.reflection_list = []
        # the use summary or not flag
        self.summary = summary
        # sumary data , if summary flag = false , set all to empty string
        self.summary_operation_data : list = ["" for i in range(5)]
        self.summary_guess_role_data : list = ["" for i in range(5)]
        self.dialogue_suggestion = "也請不要完全相信其他玩家的發言，小心掉入狼人陷阱"


    def update_game_info(self , player_id , player_name , role , roles_setting):
        """update the player name & init suspect_role_list"""
        self.player_num = len(player_name)
        self.player_id = int(player_id)
        self.player_name = player_name
        self.role = role
        self.roles_setting = roles_setting
        
        self.__load_prompt_and_example__(self.prompt_dir)
        # self.push('0' , len(self.memory_stream) , f"您為{self.player_id}號玩家({player_name[self.player_id]})" , default_importantance=10)
        self.push('0' , len(self.memory_stream) , f"您的身分為{self.role_to_chinese[role]}" , default_importantance=10)
        # self.push('0' , len(self.memory_stream) , f"{self.player_id}號玩家({player_name[self.player_id]})是{self.role_to_chinese[role]}" , default_importantance=10)

        self.suspect_role_list = {i:"未知" for i in range(self.player_num) if i != self.player_id}
        self.logger.debug(self.suspect_role_list)
        self.know_role_list[int(player_id)] = role
        # self.remain_player = [i for i in range(self.player_num)]

    def push(self , day , turn , observation , default_importantance = None):
        """push the observation in memeory stream"""
        if default_importantance == None:
            info = self.__cal_importantance__(observation)
        else:
            info = {"score" : default_importantance,"reason" : "default"}    

        full_observation = {
            "day" : day,
            "turn" : turn,
            "last_used" : turn,
            "observation" : observation ,
            "importantance" : info["score"],
            "impo_reason" : info['reason']
        }
        self.logger.debug(f"push observation {full_observation}")
        self.memory_stream.append(full_observation)

    def update_stage(self , data):
        # logging for test
        self.logger.info(f"---{data['stage']} {data['stage_description']}---\n")
        for i in self.memory_stream[-7:]:
            self.logger.debug(f"  {i['observation']}")
        self.logger.debug("")

        # if summary flag = true , save the summary
        if self.summary:
            self.summary_guess_role_data = [_.strip('\n') for _ in data['guess_summary']]
            if data['stage_summary'] != [None]:
                self.summary_operation_data = [_.strip('\n') for _ in data['stage_summary']]

        # a new day init
        # skip check_role stage
        if '-' in data['stage'] and self.day != data['stage'].split('-')[0]:
            self.day = data['stage'].split('-')[0]
            if self.day != "1":
                self.__day_init__()

        # if have vote info 
        if any(data["vote_info"].values()) :
            self.__push_vote_info__(data["vote_info"] , data["stage"])

        self.__process_announcement__(data)
        operations = self.__process_information__(data)
        for op in operations:
            op['stage_name'] = data['stage']

        return operations

    def get_long_memory_info(self):
        combine_guess_roles = {}
        for i in range(len(self.player_name)):
            if i in self.know_role_list.keys():
                combine_guess_roles[i] = self.know_role_list[i]
            else:
                combine_guess_roles[i] = self.suspect_role_list[i]

        ret = {
            "memory" : [self.__memory_to_str__(self.memory_stream[-10:])],
            "guess_roles" :[i for i in combine_guess_roles.values()],
            "token_used" : [str(self.token_used)],
            "updated" : [str(self.guess_roles_updated)]
        }
        self.guess_roles_updated = 0

        return ret

    def __process_announcement__(self , data):
        """add announcement to memory stream"""
        announcement = data['announcement']
        chat_flag = False
        for anno in announcement:
            observation = ""
            # player (except this agent) last stage chat 
            if anno["operation"] == "chat" and anno['user'][0] != self.player_id:
                observation = f"{anno['user'][0]}號玩家({self.player_name[anno['user'][0]]})說「{anno['description']}」"    
                chat_flag = True
                self.push(self.day , len(self.memory_stream) , observation)
            # player died
            elif anno["operation"] == "died":
                observation = f"{anno['user'][0]}號玩家({self.player_name[anno['user'][0]]})死了"    
                # self.remain_player.remove(int(anno['user'][0]))
                self.push(self.day , len(self.memory_stream) , observation , default_importantance=5)

    def __process_information__(self , data) -> list[dict]:
        
        informations = data["information"]
        
        operation = []
    
        for info in informations:
            # generate dialouge operation
            if info['operation'] == "dialogue":
                self.__reflection__(self.day , len(self.memory_stream))
                self.__gen_suspect_role_list__(self.day , len(self.memory_stream))
                operation.append(self.__gen_dialogue__(self.day , len(self.memory_stream)))
            # generate vote operation
            elif info['operation'] == "vote_or_not" and "vote" in data["stage"]:
                self.__reflection__(self.day , len(self.memory_stream))
                self.__gen_suspect_role_list__(self.day , len(self.memory_stream))
                operation.append(self.__gen_vote__(self.day , len(self.memory_stream) , info['target']))

        return operation
    
    def __retrieval__(self , day , turn , query , pick_num = 5 , threshold = 0.5):
        """
        the retrieval process , will call importantance,recency,relevance func
        and return the top {pick_num} memory sored by score.
        """
        self.logger.debug(f"start retrieval")
        self.logger.debug(f"  query : {query}")
        importantance_score = [ob['importantance'] for ob in self.memory_stream] 
        recency_score = self.__cal_recency__(day , turn) 
        ori_relevance_score = self.__cal_relevance__(query) 
        
        # normalize
        recency_score /= np.linalg.norm(recency_score)
        importantance_score /= np.linalg.norm(importantance_score)
        relevance_score = ori_relevance_score / np.linalg.norm(ori_relevance_score)

        importantance_factor = 1
        relevance_factor = 1
        recency_factor = 1

        # calulate score
        score = recency_score * recency_factor + importantance_score * importantance_factor + relevance_score * relevance_factor
        sorted_memory_streams = self.memory_stream.copy()
        delete_idx = []

        # delete the relevance_score < threshold
        for idx in range(len(sorted_memory_streams)):
            sorted_memory_streams[idx]["score"] = score[idx]
            if ori_relevance_score[idx] < threshold:
                delete_idx.append(idx)


        # delete score < threshold memory
        for idx in reversed(delete_idx):
            sorted_memory_streams.pop(idx)

        sorted_memory_streams.sort(key=lambda element: element['score'] , reverse=True)
        
        # logger with 1.5 * pick_num memory score
        self.logger.debug(f"  sum   | importantance | recency | relevance |  Memory | ori_rele")
        for order_mem in sorted_memory_streams[:int(1.5*pick_num)]:
            sum_score = order_mem["score"]
            ori_idx = order_mem["turn"]
            memory = order_mem['observation'].strip('\n')
            self.logger.debug(f"  {sum_score:.3f} | {importantance_score[ori_idx]:.11f} | {recency_score[ori_idx]:.5f} | {relevance_score[ori_idx]:.7f} |  {memory} | {ori_relevance_score[ori_idx]}")
        

        # updated last uesd
        for idx in range(min(pick_num , len(sorted_memory_streams))):
            self.memory_stream[sorted_memory_streams[idx]['turn']]['last_used'] = turn
            

    
        return sorted_memory_streams[:pick_num]
    
    def __reflection__(self , day , turn):
        """
        the relection func , first will gen question from recent observation
        second , use the question as retrieval query search the memory
        third , refection by the memory and push the new refection to memory
        """
            
        info = self.__reflection_question__(day , turn)
        question = info['question'].strip('\n')
        memory = self.__retrieval__(day , turn , question)
        info = self.__reflection_opinion__(memory)

        self.push(day , turn , info['opinion'])
        self.logger.info(f"reflection : {info['opinion']}")
        self.reflection_list.append(info)
    
    def __gen_suspect_role_list__(self , day , turn):
        """iterate the {suspect_role_list} and gen the new suspect role """
        for player , role in self.suspect_role_list.items():
            if player in self.know_role_list.keys(): continue
            if player == self.player_id : continue

            memory = self.__retrieval__(day , turn , f"{player}號玩家({self.player_name[player]})是什麼身分?")
            replace_order = {
                "%m" : self.__memory_to_str__(memory),
                "%e" : self.example['suspect_role_list'],
                "%t" :  f"{player}號玩家({self.player_name[player]})",
                "%rs" : self.__roles_setting_to_str__(),
                "%ar" : f"{self.player_id}號玩家({self.player_name[self.player_id]})，身分為{self.role_to_chinese[self.role]}",
                "%s" : self.__summary_to_str__(1)
            }
            final_prompt = self.prompt_template['suspect_role_list']
            for key , item in replace_order.items() : final_prompt = final_prompt.replace(key , item)
            
            info = {
                "role" : "村民",
                "reason" : "test"
            }
            info = self.__process_LLM_output__(final_prompt , {"role" : str , "reason" : str} , info , "guess roles")
            self.suspect_role_list[player] = info["role"]

        self.logger.info(f"update suspect role list : {self.suspect_role_list}")
        self.guess_roles_updated = 1
    
    def __gen_vote__(self , day , turn , target):
        """gen the vote player num & get the reason"""
        # memory = self.__retrieval__(day , turn , "幾號玩家是大家懷疑對象")
        # memory_str = self.__memory_to_str__(memory)
        replace_order = {
            "%m" : self.__memory_to_str__(self.memory_stream[-10:]),
            "%l" : self.__role_list_to_str__()[0],
            "%e" : self.example['vote'],
            "%t" : "、".join([str(_) for _ in target if _ != self.player_id]),
            "%ar" : f"{self.player_id}號玩家({self.player_name[self.player_id]})，身分為{self.role_to_chinese[self.role]}",
            "%rs" : self.__roles_setting_to_str__(),
            "%s" : self.__summary_to_str__()
        }
        final_prompt = self.prompt_template['vote']
        for key , item in replace_order.items() : final_prompt = final_prompt.replace(key , item)
        
        info = {
            "vote" : 4,
            "reason" : "test"
        }
        info = self.__process_LLM_output__(final_prompt , {"vote" : int , "reason" : str} , info , "vote")

        ret = self.ret_format.copy()
        ret['operation'] = "vote_or_not"
        ret['target'] = info["vote"]
        ret['chat'] = ""

        return ret
    
    def __gen_dialogue__(self , day ,turn):
        """gen the dialogue"""
        query = self.__reflection_question__(day , turn)['question']
        memory = self.__retrieval__(day , turn , query)
        # memory_str = self.__memory_to_str__(self.memory_stream[-5:])
        replace_order = {
            "%m" : self.__memory_to_str__(memory),
            "%l" : self.__role_list_to_str__()[0],
            "%e" : self.example['dialogue'],
            "%rs" : self.__roles_setting_to_str__(),
            "%su" : self.dialogue_suggestion,
            "%ar" : f"{self.player_id}號玩家({self.player_name[self.player_id]})，身分為{self.role_to_chinese[self.role]}",
            "%s" : self.__summary_to_str__()
        }
        final_prompt = self.prompt_template['dialogue']
        for key , item in replace_order.items() : final_prompt = final_prompt.replace(key , item)
        # final_prompt = self.prompt_template['dialogue'].replace("%m" , memory_str).replace("%e" , self.example['dialogue']).replace("%l" , sus_role_str).replace("%s" , summary)
        
        info = {
            "dialogue" : "test",
        }
        info = self.__process_LLM_output__(final_prompt , {"dialogue" : str} , info , "dialogue")

        ret = self.ret_format.copy()
        ret['operation'] = "dialogue"
        ret['target'] = -1
        ret['chat'] = info['dialogue']

        return ret
    
    def __role_list_to_str__(self):
        """
        export the {suspect_role_list} and {know_role_list} to string like
        1號玩家(Yui1)可能是女巫 
        or
        1號玩家(Yui1)是女巫 
        """
        sus_role_str = '\n'.join([f"{player}號玩家({self.player_name[player]})可能是{role}。" for player , role in self.suspect_role_list.items()])
        know_role_str = '\n'.join([f"{player}號玩家({self.player_name[player]})是{role}。" for player , role in self.know_role_list.items()])

        return sus_role_str , know_role_str
    
    def __roles_setting_to_str__(self):
        return ','.join([f"{values}個{self.role_to_chinese[key]}" for key , values in self.roles_setting.items()]) 
        
    
    def __summary_to_str__(self , summary_type=0 , pick_num = 1):
        # summary_type 0 => operation
        # summary_type 1 => guess roles
        if self.summary == False:
            return ""
        
        self.prompt_template['summary']
        summary_data_str = ""
        if summary_type == 0:
            summary_data_str = '\n'.join([f"{idx+1}. {_}" for idx , _ in enumerate(self.summary_operation_data[:pick_num])])
        else:
            summary_data_str = '\n'.join([f"{idx+1}. {_}" for idx , _ in enumerate(self.summary_guess_role_data[:pick_num])])

        return f"{self.prompt_template['summary']}\n{summary_data_str}"


    def __cal_importantance__(self , observation):
        """cal the importantance score"""
        replace_order = {
            "%m" : observation,
            "%e" : self.example['importantance'],
        }
        final_prompt = self.prompt_template['importantance']
        for key , item in replace_order.items() : final_prompt = final_prompt.replace(key , item)

        info = {
            "score" : 0,
            "reason" : "test"
        }

        info = self.__process_LLM_output__(final_prompt, {"score" : int, "reason" : str} , info , "importantance")
    
        return info

    def __cal_recency__(self , day, turn) :
        """cal the recency score"""
        initial_value = 1.0
        decay_factor = 0.90

        score = [0 for i in range(len(self.memory_stream))]

        for idx , observation in enumerate(self.memory_stream):

            time = (turn-observation['last_used'])
            score[idx] = initial_value * math.pow(decay_factor, time)
        
        score = np.array(score)
        return score
    
    def __cal_relevance__(self , query : str):
        """cal the relevance score"""
        query_embedding = self.sentence_model.encode([query] , convert_to_tensor=True)
        score = [0 for i in range(len(self.memory_stream))]
        text = [i['observation'] for i in self.memory_stream]
        embeddings = self.sentence_model.encode(text, convert_to_tensor=True)

        for idx in range(embeddings.shape[0]):
            score[idx] = util.pytorch_cos_sim(query_embedding, embeddings[idx]).to("cpu").item()
        
        score = np.array(score)
        return score
    
    def __reflection_question__(self , day , turn , pick_num = 8):
        """one of reflection process , get the question used for reflection."""
        self.logger.debug('reflection_question')

        replace_order = {
            "%m" : self.__memory_to_str__(self.memory_stream[-pick_num:]),
            "%l" : self.__role_list_to_str__()[0],
            "%e" : self.example['reflection_question'],
            "%ar" : f"{self.player_id}號玩家({self.player_name[self.player_id]})，身分為{self.role_to_chinese[self.role]}",
            "%rs" : self.__roles_setting_to_str__(),
        }
        final_prompt = self.prompt_template['reflection_question']
        for key , item in replace_order.items() : final_prompt = final_prompt.replace(key , item)

        info = {
            "question" : "test",
        }

        info = self.__process_LLM_output__(final_prompt, {"question" : str} , info , "reflection question")

        
        return info
    
    def __reflection_opinion__(self , memory):
        """one of reflection process , get the opinion as new observation."""
        self.logger.debug('reflection_opinion')
        memory_str = self.__memory_to_str__(memory)
        sus_role_str , know_role_str = self.__role_list_to_str__()

        replace_order = {
            "%m" : self.__memory_to_str__(memory),
            "%l" : self.__role_list_to_str__()[0],
            "%e" : self.example['reflection'],
            "%rs" : self.__roles_setting_to_str__(),
            "%ar" : f"{self.player_id}號玩家({self.player_name[self.player_id]})，身分為{self.role_to_chinese[self.role]}",
        }
        final_prompt = self.prompt_template['reflection']
        for key , item in replace_order.items() : final_prompt = final_prompt.replace(key , item)

        # final_prompt = self.prompt_template['reflection'].replace('%m' , memory_str).replace("%e" , self.example['reflection']).replace("%l" , sus_role_str)
        info = {
            "opinion" : "test",
            "reference" : "test",
        }
        info = self.__process_LLM_output__(final_prompt, {"opinion" : str , "reference" : str , "reason" : str} , info , "reflection opinion")
        # process reference to real memory idx
        try:
            reference_memory = info["reference"].strip('\n').split('、')
            real_reference_idx = [memory[int(idx)-1]["turn"] for idx in reference_memory]
            info["reference"] = real_reference_idx
        except Exception as e:
            self.logger.warning(f"__reflection_opinion__ fail with reference , {e}")
            
        return info
        
    def __push_vote_info__(self , vote_info : dict , stage):

        prefix = "狼人投票殺人階段:" if stage.split('-')[-1] == "seer" else "玩家票人出去階段:"

        for player , voted in vote_info.items():
            player = int(player)
            if voted != -1:
                ob = f"{player}號玩家({self.player_name[player]})投給{voted}號玩家({self.player_name[voted]})"
            else:
                ob = f"{player}號玩家({self.player_name[player]})棄票"

            self.push(self.day , len(self.memory_stream) , ob , default_importantance=5)

    def __day_init__(self):
        self.__reflection__(self.day , len(self.memory_stream))
        self.__gen_suspect_role_list__(self.day , len(self.memory_stream))
        # pass

    def __process_LLM_output__(self , prompt , keyword_dict : dict, sample_output , task_name):
        """
        communication with LLM , repeat {max_fail_cnt} util find the {keyword_list} in LLM response .
        return the {keyword_list} dict , if fail get {keyword_list} in LLM response , return {sample_output}.
        """
        success_get_keyword = False
        fail_idx = 0

        self.logger.debug(f"Start Task : {task_name}")
        self.logger.debug(f"  LLM keyword : {keyword_dict}")
        if self.log_prompt:
            self.logger.debug(f"{prompt}")
        info = {}

        while not success_get_keyword and fail_idx < self.max_fail_cnt:

            self.logger.debug(f"  {fail_idx} response generate...")
            info = {}
            result = self.__openai_send__(prompt)

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

            # check the keyword is in keyword_list and the type is satisfy require
            if info.keys() == keyword_dict.keys() and all(_.strip('\n').strip('-').isnumeric() for keyword , _ in info.items() if keyword_dict[keyword] == int):
                success_get_keyword = True
                # change data type & remove the '\n'
                for keyword , _ in info.items() : 
                    if keyword_dict[keyword] == int :
                        info[keyword] = int(_.strip('\n'))
                    else:
                        info[keyword] = _.strip('\n')
            else : 
                fail_idx+=1
                self.logger.warning(f"  {fail_idx} failed \n{result}")
        

        if fail_idx >= self.max_fail_cnt: 
            info = sample_output
            self.logger.debug(f"  failure cnt exceed {self.max_fail_cnt}")

        self.logger.debug(f"Task output : {info}")
        return info
    
    def __memory_to_str__(self , memory , add_idx=True):
        """
        export the memory dict to str like
        1. {observation[1]}
        2. {observation[2]}
        or
        {observation[1]}
        {observation[2]}
        """
        if add_idx:
            return '\n'.join([f"{idx+1}. {i['observation']}" for idx , i in enumerate(memory)])
        else:
            return '\n'.join([f"{i['observation']}" for idx , i in enumerate(memory)])


    def __openai_send__(self , prompt):
        """openai api send prompt , can override this."""
        response = self.client.chat.completions.create(
            **self.openai_kwargs,
            messages = [
                {"role":"system","content":"您為狼人殺遊戲的玩家，請根據狼人殺遊戲相關規則與經驗，回答相關的內容。"},
                {"role":"user","content":prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        
        response = response.model_dump()
        
        self.token_used += response["usage"]["total_tokens"]
        
        if response['choices'][0]["finish_reason"] == "content_filter":
            self.logger.debug("Block By Openai")
            return None

        
        return response['choices'][0]['message']['content']
    
    def __len__(self):
        return len(self.memory_stream)
    
    def __load_prompt_and_example__(self , prompt_dir):
        """load prompt json to dict"""
        self.logger.debug("load common json")
        with open(prompt_dir / "common_prompt.json" , encoding="utf-8") as json_file: self.prompt_template = json.load(json_file)
        with open(prompt_dir / "common_example.json" , encoding="utf-8") as json_file: self.example = json.load(json_file)

        
        with open(prompt_dir / "summary_addition_prompt.json" , encoding="utf-8") as json_file: self.summary_template = json.load(json_file)

        for key , prompt_li in self.prompt_template.items():
            self.prompt_template[key] = '\n'.join(prompt_li)
        for key , prompt_li in self.example.items():
            self.example[key] = '\n'.join(prompt_li)
        for key , prompt_li in self.summary_template.items():
            # load the summary addtional prompt only the `summary` flag is true
            if self.summary:
                self.prompt_template[key] = '\n'.join(prompt_li)
            else:
                self.prompt_template[key] = ""
    
    def __register_keywords__(self , keywords:dict[str,str]):
        self.logger.debug(f"Register new keyword : {keywords}")
        self.chinese_to_english.update(keywords)



