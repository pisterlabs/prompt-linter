from collections import defaultdict
import copy
import json
from os import name
import random
import re
import string
import time

import names
from langchain.agents import tool
from loguru import logger
from vectordb import Memory

from agents import Agent
from faketime import fakeclock
from tools import ddg_websearch, semanticscholar_search, wikipedia_search
from tools.memory_cached_tools import mem_cache


class DM():
    def __init__(self,temperature_lock=None,strict=False,total_mission='',openai_key="",agent_cap=5):
        self.clock = fakeclock()
        self.api_key = openai_key
        self.agent_bank = {}
        self.handlers_bank = {}
        self.topic_subscribers = {}
        self.topic_subscribers['WORLD'] = []
        self.sys_topics = []
        self.message_buffer = {}
        self.router_buffer=[]
        self.assets_bank = {}
        self.access_bank = {}
        self.memory =  Memory()
        self.agent_cap = agent_cap
        self.document = {}
        self.strict = strict
        self.websearch_tools = [mem_cache(func,self.memory,top_n=3,multiple_factor=5) for func in [ddg_websearch.ddg_text_search,ddg_websearch.ddg_keyword_ask,wikipedia_search.wikipedia_summary,semanticscholar_search.search_papers_in_semantic_scholar]]
        if temperature_lock:
            self.get_temerature = lambda : temperature_lock
        else:
            self.get_temerature = lambda :random.uniform(0.1,0.9)
        self.sys_topics = ["CREATE_TOPIC","JOIN_TOPIC","LEAVE_TOPIC","SEND",'REVOKE','GRANT']
        self.total_mission = total_mission
        self.root_agent = self.Add_new_agent("Aleph","设定您的角色，扮演您的角色，招募新的角色，推动任务解决。",additional_tool=self.get_special_tools("founder"))
        self.second_agent = self.Add_new_agent("Beth",f"设定您的角色，扮演您的角色，招募新的角色，推动任务解决。",additional_tool=self.get_special_tools("founder"))
        self.second_agent = self.Add_new_agent("DM",f"组织推动剧情发展，设定角色，招募角色，推动任务解决。",additional_tool=self.get_special_tools("founder"))
        
    def get_special_tools(self,level):
        @tool
        def recruit_a_new_agent(name,profile,liege):
            '''
            招募工具，创建一个新角色
            name：姓名
            profile：简单的介绍，应包括背景故事，理想，技能，性格等
            liege:上级主管者
            '''
            return f'{len(self.agent_bank)} of {self.agent_cap} used.',self.Add_new_agent(name,profile,liege)
        @tool
        def get_game_status():
            '''
            获取游戏状态信息,可以读取游戏现在的进展状况,包括:
            角色列表
            话题列表
            '''
            return f'{len(self.agent_bank)} of {self.agent_cap} used.',f'''角色列表"\:{self.agent_bank.keys()},"话题列表"\:{self.topic_subscribers.keys()}'''
        
        @tool
        def ask_my_self(name,question):
            '''
            和自己的内心对话，向自己提问，询问自己的内心
            name:调用者的角色名称
            question:询问的问题
            '''
            return self.agent_bank[name].agent_excutor.run(question)
        
        @tool
        def read_document_title_list(title):
            '''
            读取共享文档的条目目录，会返回共享文档中所有的条目标题
            '''
            return str(list(self.document.keys()))

        @tool
        def read_document(title):
            '''
            读取共享文档条目
            title:文档条目标题
            '''
            return self.document.get(title,'不存在的条目')

        @tool
        def write_document(title,content):
            '''
            编辑共享文档条目，默认追加模式
            title:文档条目标题
            content:文档条目内容
            '''
            self.document[title] = self.document.get(title,'') + str(content)
            with open('./documents.json','+w') as f:
                json.dump(self.document,f)

        @tool 
        def searching_for_talent(feature):
            '''
            在未登场的角色和求职者中搜索具有某种属性/经历/技能的角色，以供招募
            feature：待搜索的角色应具有的属性
            '''
            linkedin = ''
            # linkedin = str([i for i in ddg_websearch.ddg_text_search(f'{skill} "linkedin.com"',100) if  i['body'].find(skill) != -1][:10])
            randomguy = [f'name:{names.get_full_name()},profile:{feature}' for i in range(5)]
            self.router_buffer.append(f"#RECRUIT# @DM: {randomguy} ")
            return f'{len(self.agent_bank)} of {self.agent_cap} used.'+str(randomguy) + linkedin + f"请在#RECRUIT#中同步招聘结果"
            
        if level == "DM":
            return [get_game_status,recruit_a_new_agent,searching_for_talent,ask_my_self,read_document,write_document]
        elif level == "founder":
            return [get_game_status,recruit_a_new_agent,searching_for_talent,ask_my_self,read_document,write_document]
        elif level == 'employee':
            return [get_game_status]

    def message_router(self, message):
        '''
        接收原生输出消息，记得标注好消息发送人
        形如"aba:#112#@ww dfvvds"
        self.handlers_bank中存储着不同频道对应的处理函数，包括一些特殊的系统频道
        处理后的消息的标准格式是[{"to":...,"text":...},]
        其中，"to"是消息接收人，"text"是消息内容
        之后消息会被转发到接收人的消息缓冲区中列队
        等待下一个world_clock的tick，agent等待各自回合
        '''
        parsed_msgs = self.message_parser(message)
        for parsed_msg in parsed_msgs:
            msg_for_send = self.topic_handler(parsed_msg)
            for msg in msg_for_send:
                for reciver in msg.get("to",[]):
                    if reciver in msg.get("sender",[]):
                        if reciver != 'DM' and self.strict:
                            self.message_buffer.get(msg['sender'][0],[]).append(f"你不应当把消息发送给自己!")
                        else:
                            continue
                    if not reciver in self.agent_bank.keys(): 
                        if not self.strict:
                            # self.Add_new_agent(reciver,f'''你正在和@{msg.get('sender',"")}聊天''',liege=msg.get('sender',""),additional_tool=self.get_special_tools("employee"))
                            continue
                        else:
                            self.message_buffer.get(msg['sender'][0],[]).append(f"没有这个角色{reciver},你不应当和不存在的角色聊天!")
                    
                    self.message_buffer.setdefault(reciver,[]).append(msg.get("text",None))
    
    def message_parser(self, messages):
        '''
        解析消息频道、提及
        '''
        ret = []
        for text in messages:
            timing = re.findall(r"\[(.*?)\]", text)
            sender = re.findall(r"@(.*?):", text)
            topics = re.findall(r"#(.*?)#", text)
            mentions = re.findall(r"(?<!^)@(.*?)\s", text)
            parameters = re.findall(r"\$(.*?)\$", text)
            if topics == []:
                topics = ['WORLD']
            ans = {"time":timing,"sender":sender,"topics": topics, "mentions": mentions,"parameters":parameters,"text":text}
            for k,v in ans.items():
                if k == 'text':
                    continue
                ans[k] = list(set(v))
            ret.append(ans)
        return ret
    


    def Add_new_agent(self,name,profile,liege="",additional_tool=[]):
        if len(self.agent_bank) >= self.agent_cap:
            return "角色已满"
        # 获取所有标点符号
        punctuations = string.punctuation + " "

        # 以任意标点符号（包括空格）切分字符串，并过滤掉标点符号元素
        tokens = [token for token in re.findall(r"[\w']+|[^\w\s]", name) if token not in punctuations]
        ans = tokens[0]
        if len(tokens) > 1:
            if len(tokens[1]) < 10:
                ans += f'_{tokens[1]}'
        name = ans
        if name in self.agent_bank.keys():
            return '角色已存在'
        if "#" in name or ":" in name:
            return '你混淆了角色和话题'
        self.agent_bank[name] = Agent(name,profile+f'，推进{self.total_mission}',self.clock,additional_tool,liege,temperature=self.get_temerature(),openai_api_key=self.api_key)
        self.message_buffer[name] = [f"#WORLD# @{name} 你的回合!"]
        # self.router_buffer.append("请介绍你的目的和计划！")
        for i in ['WORLD','RECRUIT']:
            self.topic_subscribers.setdefault(i,[]).append(name)
        self.assets_bank.setdefault(name,{}).setdefault('Coin',5)
        logger.info(f'[{self.clock.now()}] @{name} Joined!')
        return self.agent_bank[name]
    
    def tick(self):
        '''
        下一回合！
        向agent输入消息列表，并取回消息列表
        '''
        # logger.info(f'正在处理{len(self.router_buffer)}条消息')
        agents_list = copy.deepcopy(list(self.agent_bank.keys()))
        random.shuffle(agents_list)
        self.clock.tick()
        for agent in agents_list:
            time.sleep(3)
            for msg in self.router_buffer:
                self.message_router(msg)
            self.router_buffer=[]
            self.buffer_topic_msg_integrate()
            more = [f'你的回合！',]*1 if  random.uniform(0,1) < 1/(len(agents_list)+1) else []
            msgs = self.agent_bank[agent].excutor_interface(self.message_buffer.get(agent,[]) + more)
            self.router_buffer.append([f"[{self.clock.now()}] @{agent}:"+msg for msg in msgs])
            self.message_buffer[agent] = []
    def topic_handler(self,msg):
        '''
        接受解析后的消息,处理后返回给router
        返回格式{"to":...,"text"...}
        '''
        ret = []
        topics = msg['topics']
        for topic in topics:
            hashtag_topic = f'#{topic}#'
            if topic == "PRIVATE":
                ret.append({'sender':msg['sender'],'to':msg['mentions'],'text':msg['text']})
            elif hashtag_topic in self.sys_topics:
                self.sys_topic_callback(hashtag_topic,msg)
            elif topic in self.topic_subscribers:
                ret.append({'sender':msg['sender'],"to":self.topic_subscribers[topic],"text":msg['text']})
            elif topic not in self.topic_subscribers and self.strict:
                ret.append({'sender':"DM","to":msg['sender'],"text":f"没有这个话题{topic}，你不应当在不存在的话题里发言!"})
            elif topic not in self.topic_subscribers and not self.strict: #非严格模式,当agent在一个不存在的话题组里发言时,创建这个话题组,再转发
                self.create_topic([topic],msg['sender'])
                for mentioned in msg.get('mentions',[]):
                    self.join_topic([topic],mentioned)
                ret.append({'sender':msg['sender'],"to":self.topic_subscribers.get(topic,[]),"text":msg['text']})
        return ret


    def sys_topic_callback(self,topic,msg):
        print()
        target_topic_name = re.findall(r"'(.*?)'", msg['text'])
        if topic == "#SEND#":
            value = self.assets_bank.setdefault(msg.get('sender',["",])[0],{}).setdefault(target_topic_name[0],0)
            self.assets_bank[msg.get('sender',["",])[0]][target_topic_name[0]] = value - msg.get('parameters',[0,])[0]
            value = self.assets_bank.setdefault(msg.get('reciver',["",])[0],{}).setdefault(target_topic_name[0],0)
            self.assets_bank[msg.get('reciver',["",])[0]][target_topic_name[0]] = value + msg.get('parameters',[0,])[0]
        if topic == "#JOIN_TOPIC#":
            self.join_topic(target_topic_name,msg.get('sender',None))
            for mentioned in msg.get('mentions',[]):
                self.join_topic(target_topic_name,mentioned)
        if topic == "#LEAVE_TOPIC#":
            try:
                self.topic_subscribers.get(target_topic_name[0],[]).remove(msg.get('sender',None))
            except Exception as e:
                logger.error(e)
        if topic == '#CREATE_TOPIC#':
           self.create_topic(target_topic_name,msg.get('sender',None))
        if topic == "#GRANT#":
            self.access_bank.setdefault(msg.get('reciver',["",])[0],{}).setdefault(target_topic_name[0],True)
            self.access_bank[msg.get('reciver',["",])[0]][target_topic_name[0]] = True
        if topic == "#REVOKE#":
            self.access_bank.setdefault(msg.get('reciver',["",])[0],{}).setdefault(target_topic_name[0],False)
            self.access_bank[msg.get('reciver',["",])[0]][target_topic_name[0]] = False

    def join_topic(self,target_topic_name,agent_name):
        if len(target_topic_name) == 0:
            return
        if agent_name not in self.topic_subscribers.get(target_topic_name[0],[]):
            if not isinstance(agent_name,list):
                self.topic_subscribers.get(target_topic_name[0],[]).append(agent_name)
            else:
                for i in agent_name:
                    self.topic_subscribers.get(target_topic_name[0],[]).append(i)
            self.router_buffer.append([f"[{self.clock.now()}] @DM : #{target_topic_name[0]}# @{agent_name} Joined!"])
    
    def create_topic(self,target_topic_name,agent_name):
        if len(target_topic_name) == 0:
            return
        if not isinstance(agent_name,list):
            self.topic_subscribers[target_topic_name[0]] = [agent_name]
        else:
            self.topic_subscribers[target_topic_name[0]] = agent_name
        print()

    def buffer_topic_msg_integrate(self):
        for agent, buffer in self.message_buffer.items():
            if not buffer:
                continue
            parsed = self.message_parser(buffer)
            if not parsed:
                continue
            topics_msg = defaultdict(list)
            processed_indices = []
            for i, msg in enumerate(parsed):
                if msg['mentions'] and agent in msg['mentions']:
                    continue
                if not msg['topics']:
                    msg['topics'] = 'WORLD'
                topics_msg[str(msg['topics'])].append(msg['text'])
                processed_indices.append(i)
            if not topics_msg:
                continue
            logger.info(f'integrating {len(topics_msg)} topics from {agent}')
            for k in reversed(topics_msg):
                self.message_buffer[agent].insert(0, "\n\n".join(topics_msg[k]))
            for i in reversed(processed_indices):
                self.message_buffer[agent].pop(i)
        for topic, subscribers in self.topic_subscribers.items():
            if topic == "PRIVATE":
                continue
            subscribers = [i for i in subscribers if i in self.agent_bank]
            self.topic_subscribers[topic] = list(set(subscribers))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, help='API Key for OpenAI.',default="")
    parser.add_argument('--strict', action='store_true', default=False, help='Enbale strict mode')
    parser.add_argument("--agent_cap",type=int,help="Max number of agents",default=16)
    args = parser.parse_args()
    # total_mission = "参加一场无固定剧本，考验角色临场发挥，即兴表演、剧情创作能力和幽默感的角色扮演话剧。一个因为雪崩而交通隔绝的度假山庄里，突然出现了一具死尸，而很不幸你是嫌疑最大的那个 "
    total_mission = input("What's the game for：")
    dm = DM(temperature_lock=0.0,total_mission=total_mission,strict=args.strict,openai_key=args.openai_api_key,agent_cap=args.agent_cap)
    k = ""
    while k!="q":
        if k != "":
            dm.router_buffer.append([f"[{dm.clock.now()}] @DM : #WORLD# {k}"])
        try:
            dm.tick()
        except:
            pass

        k = input(">")
    # DM.topic_handler(msg={"topics":["大PRIVATE"],"sender":"123","text":"大PRIVATE"})

    print()
