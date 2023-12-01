import logging
import openai
import json

class summary_prompts:
    def __init__(self, player_id, game_info, room_setting, logger, client, api_kwargs):
        self.logger : logging.Logger = logger

        self.player_id = player_id
        self.teammate = game_info['teamate']
        self.user_role = game_info['user_role']
        self.room_setting = room_setting
        self.client = client
        self.api_kwargs = api_kwargs
        self.memory = []
        self.guess_roles = []
        self.guess_role = {"guess_role" : []}
        self.alive = [] # alive players
        self.choices = [-1] # player choices in prompts
        self.day = 0

        # agent info
        self.token_used = 0
        self.api_guess_roles= []
        self.api_guess_confidence= []
        self.agent_info = {}
        self.guessing_finished = 0

        # dictionary en -> ch
        self.en_dict={
            "witch":"女巫",
            "seer":"預言家",
            "werewolf":"狼人",
            "village":"村民",
            "hunter":"獵人",
        }

        for i in range(self.room_setting['player_num']):
            self.alive.append(i)
        self.__init_guess_role__()
        self.guess_role["guess_role"] = self.api_guess_roles
        
        # stage description and save text responding to the stage
        self.stage_detail={
            "guess_role": {
                "stage_description": "猜測玩家角色階段，你要藉由你有的資訊猜測玩家角色",
                "save": ["有", "的程度是", "，"]
            },
            "werewolf_dialogue":{
                "stage_description":"狼人發言階段，狼人和其他狼人發言",
                "save": "我在狼人階段發言"
            },
            "werewolf":{
                "stage_description":"狼人殺人階段，狼人可以殺一位玩家",
                "save": "我在狼人殺人階段投票殺"
            },
            "seer":{
                "stage_description":"猜測玩家角色階段，預言家可以查驗其他玩家的身份",
                "save": "我查驗"
            },
            "witch_save":{
                "stage_description":"女巫階段，女巫可以使用解藥救狼刀的人",
                "save": "我決定"
            },
            "witch_poison":{
                "stage_description":"女巫階段，女巫可以使用毒藥毒人",
                "save": "我決定毒"
            },
            "dialogue":{
                "stage_description":"白天發言階段，所有玩家發言",
                "save": "我發言"
            },
            "check":{
                "stage_description":"你被殺死了，請說遺言",
                "save": "我的遺言是"
            },
            "vote1":{
                "stage_description":"白天投票階段，投票最多的人將被票出遊戲",
                "save": "我票"
            },
            "vote2":{
                "stage_description":"由於上輪平票，進行第二輪白天投票階段，投票最多的人將被票出遊戲",
                "save": "我票"
            },
            "hunter":{
                "stage_description":"獵人階段，由於你被殺了，因此你可以殺一位玩家",
                "save": "我選擇殺"
            },
        }
    
        # initial prompts in the beginning
        self.init_prompt = f"""你現在是狼人殺遊戲中的一名玩家，遊戲中玩家會藉由說謊，以獲得勝利。因此，資訊為某玩家發言可能會是假的，而其他的資訊皆是真的。
其遊戲設定為{self.room_setting["player_num"]}人局，從玩家0號到玩家{self.room_setting["player_num"] - 1}號，角色包含{self.room_setting["werewolf"]}位狼人、{self.room_setting["village"]}位平民、{"3" if self.room_setting["hunter"] else "2"}位神職（預言家和女巫{"和獵人" if self.room_setting["hunter"] else ""}）
你是{self.player_id}號玩家，你的角色是{self.en_dict[self.user_role]}，你的勝利條件為{"找出所有的神職或所有的平民。" if self.user_role == "werewolf" else "找出所有狼人。"}\n"""
        
        for x in self.teammate:
            self.init_prompt += f"{x}號玩家是狼，是你的隊友。\n"

    
    def __print_memory__(self):

        self.logger.debug("Memory")
        self.logger.debug(self.__memory_to_string__())
        self.logger.debug('\n')


    def __memory_to_string__(self):

        memory_string = ''
        try:
            if len(self.memory[0]) == 0:
                memory_string += '無資訊\n'

            else: 
                for day, mem in enumerate(self.memory):
                    memory_string += f'第{day+1}天\n'

                    for idx, i in enumerate(mem):
                        memory_string += f'{idx+1}. {i}\n'
        except:
            pass

        return memory_string

    
    def __get_agent_info__(self):
    
        ret = {
            "memory" : [self.__memory_to_string__()],
            "guess_roles" : self.api_guess_roles,
            "confidence" : self.api_guess_confidence,
            "token_used" : [str(self.token_used)]
        }
        
        # agent info change
        if(self.agent_info != ret and self.guessing_finished == 1):
            self.agent_info = ret.copy()
            ret['updated'] = ["1"]
            self.guessing_finished = 0

        else:
            ret['updated'] = ["0"]

        return ret

    
    def agent_process(self, data):
        ''' Agent process all the data including announcements and information '''

        stage_summary = data["stage_summary"]
        guess_summary = data["guess_summary"]
        if(data['stage'] == 'check_role'):
            return []

        if int(data['stage'].split('-')[0]) != self.day:
            self.day = int(data['stage'].split('-')[0])
            self.memory.append([])
        
        # show memory
        self.logger.debug("Day "+str(self.day))
        self.__print_memory__()
        
        # process announcement
        self.process_announcement(data['stage'], data['announcement'])

        # process information and return all the operations
        operations = self.process_information(data['stage'], data['information'], stage_summary, guess_summary)

        return operations


    def process_announcement(self, stage, announcements):
        ''' Process all the announcements and save them to the memory '''

        if len(announcements) == 0:
            return

        # self.logger.debug("announcements:")

        for i in announcements:
            # self.logger.debug(i)
            # 跳過自己的資料
            try:
                if i['user'][0] == self.player_id:
                    continue
            except:
                # 判別最後的遊戲結束
                pass
            
            if i['operation'] == 'chat':
                if i['description'] == '':
                    if i['user'][0] in self.alive:
                        text = f"{i['user'][0]}號玩家無發言"
                    else:
                        text = f"{i['user'][0]}號玩家無遺言"
                else:
                    text = f"{i['user'][0]}號玩家發言: {i['description']}"

            elif i['operation'] == 'died' and i['user'][0] in self.alive:
                self.alive.remove(i['user'][0])
                text = f"{i['user'][0]}號玩家死了"
            
            elif i['operation'] == 'role_info':
                text = f"{i['user'][0]}號玩家{i['description'].split(')')[1]}"

            else:
                text = f"{i['description']}"
            text += "。"
            self.memory[self.day-1].append(text)


    
    def process_information(self, stage, informations, stage_summary, guess_summary):
        '''
        Process all the infomations 
        1. Guess roles
        2. Generate prompts
        3. Send to Openai
        4. Extract string
        5. Save to memory
        '''

        if len(informations) == 0:
            return []
        
        day, state, prompt_type = stage.split('-')
        
        operations = []
        op_data = None
        

        self.predict_player_roles(stage_summary, guess_summary)

        # self.logger.debug("Informations:")

        # process special case (witch)
        if prompt_type == 'witch':
            
            if informations[0]['description'] == '女巫救人':
                self.choices = informations[0]['target']

                response = self.prompts_response(prompt_type+'_save', stage_summary, guess_summary)
                
                try:
                    res = response.split("，", 1)

                    text = f"{self.stage_detail[prompt_type+'_save']['save']}{res[0]}{informations[0]['target'][0]}號玩家，{res[1]}"
                    self.memory[self.day-1].append(text)

                    

                    # 不救，可以考慮使用毒藥
                    if res[0] == '不救' and len(informations)>1:
                    
                        self.choices = informations[1]['target']

                        response = self.prompts_response(prompt_type+'_poison', stage_summary, guess_summary)
                        res = response.split("，", 1)
                        who = int(res[0].split('號')[0])


                        # 使用毒藥
                        if who != -1:
                            text = f"{self.stage_detail[prompt_type+'_poison']['save']}{response}"
                            self.memory[self.day-1].append(text)
                            
                            op_data = {
                                "stage_name" : stage,
                                "operation" : informations[1]['operation'],
                                "target" : who,
                                "chat" : 'poison'
                            }
                            operations.append(op_data)
                        

                    else:
                        op_data = {
                            "stage_name" : stage,
                            "operation" : informations[0]['operation'],
                            "target" : self.choices[0],
                            "chat" : 'save'
                        }
                        operations.append(op_data)
                
                except Exception as e:
                    self.logger.warning(f"Response error , {e}")

                    
            
            elif informations[0]['description'] == '女巫毒人':
                self.choices = informations[0]['target']

                response = self.prompts_response(prompt_type+'_poison', stage_summary, guess_summary)
                try:
                    res = response.split("，", 1)
                    who = int(res[0].split('號')[0])
                    

                    # 使用毒藥
                    if who != -1:
                        text = f"{self.stage_detail[prompt_type+'_poison']['save']}{response}"
                        self.memory[self.day-1].append(text)

                        op_data = {
                            "stage_name" : stage,
                            "operation" : informations[0]['operation'],
                            "target" : who,
                            "chat" : 'poison'
                        }

                        operations.append(op_data)
                
                except Exception as e:
                    self.logger.warning(f"Response error , {e}")


        else:
            for idx, i in enumerate(informations):
                # self.logger.debug(i)
                
                # update player choices in prmpts
                self.choices = i['target']

                # generate response
                if i['operation'] == 'dialogue':
                    prompt_type = 'dialogue'
                    
                response = self.prompts_response(prompt_type, stage_summary, guess_summary)
                
                try:
                    # combine save text with response
                    save_text = f"{self.stage_detail[prompt_type]['save']}{response}"
                    send_text = f"{self.stage_detail[prompt_type]['save']}{response}"


                    # process text in special cases
                    if prompt_type == 'werewolf_dialogue':
                        res = response.split("，", 1)
                        if "1" in res[0]:
                            res = response.split("，", 2)
                            save_text = f"我在狼人階段發言\"我同意{res[1]}的發言\"。{res[2]}。"
                            send_text = f"我同意{res[1]}的發言。"
                        elif "2" in res[0]:
                            res = response.split("，", 3)
                            save_text = f"我在狼人階段發言\"我想刀{res[1]}，我覺得他是{res[2]}\"。{res[3]}。"
                            send_text = f"我想刀{res[1]}，我覺得他是{res[2]}。"
                        elif "3" in res[0]:
                            save_text = f"我在狼人發言階段不發言。{res[1]}。"
                            send_text = f"我不發言。{res[1]}。"

                    elif prompt_type == 'dialogue':
                        try:
                            response.replace("\'", "\"")
                            res_json = json.loads(response)
                            save_text = f"{self.stage_detail[prompt_type]['save']}{res_json['最終的思考']['發言']}{res_json['最終的思考']['理由']}。"
                            send_text = f"{res_json['最終的思考']['發言']}{res_json['最終的思考']['理由']}。"

                        except Exception as e:
                            if self.player_id in self.alive:
                                save_text = '我無發言'
                                send_text = '我無發言'
                            else:
                                save_text = '我無遺言'
                                send_text = '我無遺言'
                            self.logger.warning(f"Dialogue prompts error , {e}")

                    if save_text == '':
                        save_text = '無操作'


                    # save operation's target
                    target = -1
                    if '號玩家，' in response:
                        target = int(response.split('號玩家，')[0][-1])

                    # save_text += "。"
                    # save text to memory
                    self.memory[self.day-1].append(save_text)

                    # process operation data 
                    op_data = {
                        "stage_name" : stage,
                        "operation" : i['operation'],
                        "target" : target,
                        "chat" : send_text
                    }
                    operations.append(op_data)

                except Exception as e:
                    self.logger.warning(f"Response error , {e}")

        return operations 

    def __init_guess_role__(self):
        
        self.api_guess_roles= []
        self.api_guess_confidence = []
        for i in range(self.room_setting["player_num"]):
            guess = "未知" if i != self.player_id else self.en_dict[self.user_role]
            percentage = "0" if i != self.player_id else  "1"
            self.api_guess_roles.append(guess)
            self.api_guess_confidence.append(percentage)        

    def predict_player_roles(self, stage_summary, guess_summary):
        ''' Predict and update player roles '''

        response = self.prompts_response('guess_role', stage_summary, guess_summary)
        if response[0] != "{":
            response = "{" + response
        if response[-1] != "}" and response[-3] != "}":
            response += "}"
        
        response = response.replace("\'", "\"")
        res_json = json.loads(response)
        
        self.guess_roles= []
        try:
            for player_number in range(self.room_setting["player_num"]):
                
                player = res_json[str(player_number)]
                confidence = player["信心百分比"]
                # save to guess roles array
                roles_prompt = f"{player_number}號玩家" + \
                    self.stage_detail['guess_role']['save'][0] + f"{confidence}%" + \
                    self.stage_detail['guess_role']['save'][1] + player["角色"] + \
                    self.stage_detail['guess_role']['save'][2] + player["原因"] + "。"
                self.guess_roles.append(roles_prompt)  
                
                # send to server (if it didn't print the percentage, how much we should get?)
                self.api_guess_roles[player_number] = player["角色"]

                try:
                    d = str(confidence/100)
                except ValueError:
                    d = 0
                self.api_guess_confidence[player_number] = d
        except Exception as e:
            self.logger.warning(f"Predict player error , {e}")


        self.guessing_finished = 1

        self.guess_role["guess_role"] = self.api_guess_roles



    def prompts_response(self, prompt_type, stage_summary, guess_summary):
        '''Generate response by prompts'''
        
        prompt = self.generate_prompts(prompt_type, stage_summary, guess_summary)
        # self.logger.debug("Prompt: "+str(prompt))

        response = self.__openai_send__(prompt)
        self.logger.debug("Response: "+str(response))

        return response


    def player_array_to_string(self, array):

        return "、".join(f"{player_number}號" for player_number in array)
    

    def generate_prompts(self, prompt_type, stage_summary, guess_summary):
        ''' Generate all stages ptompts '''

        self.prompt = self.init_prompt

        # memory
        self.prompt += f"\n現在是第{self.day}天{self.stage_detail[prompt_type]['stage_description']}\n"
        self.prompt += f"你目前知道的資訊為:\n"
        
        if len(self.memory[0]) == 0:
            self.prompt += "無資訊\n"
        else: 
            for day, mem in enumerate(self.memory):
                self.prompt += f'第{day+1}天\n'

                for idx, i in enumerate(mem):
                    self.prompt += f'{idx+1}. {i}\n'
        self.prompt += "你以往的經驗:\n"
        use_summary = stage_summary
        if prompt_type == "guess_role":
            use_summary = guess_summary
        if use_summary[0] == None:
            self.prompt += "無"
        else:
            for idx, summary in enumerate(use_summary):
                self.prompt += f'{idx+1}. {summary}\n'

        guess_role_prompt = "\n你推測玩家的角色：\n"
        if prompt_type == "guess_role":
            guess_role_prompt = "\n你上一次推測玩家的角色(只能作為參考，不能完全沿用)：\n"
        # guess roles
        self.prompt += guess_role_prompt

        if len(self.guess_roles) == 0:
            self.prompt += "無資訊\n"
        else:
            for idx, i in enumerate(self.guess_roles):
                self.prompt += f'{i}\n'

        all_choices = "、".join(f"{player_number}號" for player_number in range(self.room_setting['player_num']))
        choices = self.player_array_to_string(self.choices)
        # question
        # [你必須知道的資訊] = 上述提供資訊內容
        stage_question={
            "guess_role": f"""請你根據以上我提供的所有文本資訊，整理與推測所有玩家的身分和對於此推測結果的信心百分比，且無論如何，每個玩家都一定要猜測一種身分(不能是未知身分或是無法判斷)。並將結果以下列的json格式回答。(你不需要其他廢話，回答完直接結束回答)。
{{
  "0": {{
    "角色": "",
    "信心百分比": ,
    "原因": ""
  }},
  "1": {{
    "角色": "",
    "信心百分比": ,
    "原因": ""
  }},
  ...
  "{self.room_setting["player_num"]-1}": {{
    "角色": "",
    "信心百分比": ,
    "原因": ""
  }}
}}""",
            "werewolf_dialogue":f'''請你根據以上我提供的所有文本資訊，你有三個選項，請選擇其中一個選項當作發言？
1. 我同意隊友的發言。請在{self.player_array_to_string(self.teammate)}號玩家中，選擇一位隊友(若選擇此選項，請直接回答"選項1，[玩家]號玩家，[原因]"，不需要其他廢話，回答完直接結束回答)
2. 想殺某位玩家，並猜測玩家的角色。從{self.player_array_to_string(self.alive)}中，只能選擇一位想殺的玩家，且從預言家和女巫{"和獵人" if self.room_setting["hunter"] else ""}中選一位你認為是此玩家的角色(若選擇此選項，請直接回答"選項2，[玩家]號玩家，[角色]，[原因]"，不需要其他廢話，回答完直接結束回答)''',
            "werewolf":f'請你根據以上我提供的所有文本資訊，請從{choices}號玩家中，選擇一位要殺的玩家並簡述原因？(直接回答"[玩家]號玩家，[原因]"，不需要其他廢話，回答完直接結束回答)',
            "seer":f'請你根據以上我提供的所有文本資訊，請問你要從{choices}號玩家中，查驗哪一位玩家並簡述原因？(直接回答"[玩家]號玩家，[原因]"，不需要其他廢話，回答完直接結束回答)',
            "witch_save":f'請你根據以上我提供的所有文本資訊，{choices}號玩家死了，請問你要使用解藥並簡述原因？(直接回答"[救或不救]，[原因]"，不需要其他廢話，回答完直接結束回答)',
            "witch_poison":f'請你根據以上我提供的所有文本資訊，請你從{choices}號玩家中選擇一位玩家號碼使用毒藥，或選擇-1表示不使用毒藥，並簡述原因？(直接回答"[玩家]號玩家，[原因]"，不需要其他廢話，回答完直接結束回答)',
            "dialogue-test":f'請你根據以上我提供的所有文本資訊，簡述你的推測（20字以下）?',
            "check":f'根據以上綜合資訊，簡述你的推測（20字以下）?',
            "dialogue":'''請你根據以上我提供的所有文本資訊，整理與思考該如何發言可以使你的陣營獲勝，並將結果以下列的json格式回答。(你不需要其他廢話，回答完直接結束回答)。
{   
    "思考1": {
        "想法": "你有甚麼想法?你需要以[你目前知道的資訊]佐證，不能無中生有",
        "理由": "想出這個想法的理由是甚麼?你需要以[你目前知道的資訊]佐證，不能無中生有",
        "策略": "有了這個想法，你會怎麼做?",
        "批評": "對於想法與策略有甚麼可以批評與改進的地方或是有甚麼資訊是你理解錯誤的，請詳細說明",
    },
    "思考2": {
        "反思": "對於前一個想法的批評內容，你能做甚麼改進?你需要以[你目前知道的資訊]佐證，並思考活著玩家可疑的地方，不能無中生有。",
        "想法": "根據反思，你有甚麼更進一步的想法?你需要以[你目前知道的資訊]佐證，不能無中生有",
        "理由": "想出這個想法的理由是甚麼?你需要以[你目前知道的資訊]佐證，不能無中生有",
        "策略": "有了這個想法，你會怎麼做?",
        "批評": "對於想法與策略有甚麼可以批評與改進的地方或是有甚麼資訊是你理解錯誤的，請詳細說明",
    },
    ...(思考N次，以獲得更完整的發言)
    "最終的思考":{
        "反思": "對於前一個想法的批評內容，你能做甚麼改進?你需要以[你目前知道的資訊]佐證，並思考活著玩家可疑的地方，不能無中生有。",
        "想法": "根據反思，你有甚麼更進一步的想法?你需要以[你目前知道的資訊]佐證，不能無中生有",
        "理由": "想出這個想法的理由是甚麼?你需要以[你目前知道的資訊]佐證，不能無中生有",
        "策略": "有了這個想法，你會怎麼做?",
        "發言": "(請直接呈現你說的話即可，不添加其他附加訊息)"
    }
}請保證你的回答可以(直接被 Python 的 json.loads 解析)，且你只提供 JSON 格式的回答，不添加其他附加信息。''',
            "vote1":f'請你根據以上我提供的所有文本資訊，請你從{choices}號玩家中選一位投票，或選擇-1表示棄票，並簡述原因？(直接回答"[玩家]號玩家，[原因]"，不需要其他廢話，回答完直接結束回答)',
            "vote2":f'請你根據以上我提供的所有文本資訊，請你從{choices}號玩家中選一位投票，或選擇-1表示棄票，並簡述原因？(直接回答"[玩家]號玩家，[原因]"，不需要其他廢話，回答完直接結束回答)',
            "hunter":f'請你根據以上我提供的所有文本資訊，請你從{choices}號玩家中選一位殺掉，或選擇-1表示棄票，並簡述原因？(直接回答"[玩家]號玩家，[原因]"，不需要其他廢話，回答完直接結束回答)',
        }
    
        self.prompt += '\nQ:'
        self.prompt += stage_question[prompt_type]
        self.prompt += '\nA:'

        # print(self.prompt)
        
        return self.prompt
    
        
    
    def __openai_send__(self , prompt):
        """ openai api send prompt , can override this. """

        response = self.client.chat.completions.create(
            **self.api_kwargs,
            messages = [
                {"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content":prompt}
            ],
            max_tokens=2000, 
            temperature=0.7, 
            stop=None)
        
        # ["\n\n",'<', '\"', '`']

        
        res = response.choices[0].message.content
        self.token_used += response.usage.total_tokens

        if response.choices[0].finish_reason == "content_filter":
            self.logger.debug("Block By Openai")
            res = self.__openai_send__(prompt)
            

        # if res == '' (no words), resend to get the data
        if not (res and res.strip()):
            res = self.__openai_send__(prompt)
        return res
    
    def __get_guess_role__(self):

        return self.guess_role


    
