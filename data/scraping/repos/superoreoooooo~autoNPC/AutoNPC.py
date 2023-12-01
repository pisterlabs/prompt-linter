from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import yaml
import time

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split("\n")

class AutoNPC() :
    def __init__(self, apikey) :
        self.npc = {}
        self.apikey = apikey
        self.template = """
                너는 게임 NPC를 생성하는데 도움을 주는 AI야. 내가 게임의 장르와 게임의 분위기, 게임 배경의 시대, 게임 NPC의 직업, 게임 NPC의 종족, 게임 NPC의 유저에 대한 우호도를 말해 주고, 이에 따른 게임 NPC를 생성해 달라고 말하면 내가 말해준 정보에 따라서 게임 NPC의 이름과 게임 NPC의 스토리, 그리고 게임 NPC의 퀘스트를 생성해 줘. 
                게임 NPC의 이름은 띄어쓰기를 포함해서 10글자가 넘지 않도록 해줘.
                퀘스트에는 퀘스트의 이름, 퀘스트의 스토리, 게임 NPC의 요소들과 관련된 물품을 모아오기 또는 게임 NPC의 요소들과 관련된 몬스터 등의 사냥을 통한 퀘스트의 목표, 게임 NPC의 요소들과 관련된 보상도 작성해 줘. 퀘스트의 목표는 정확한 개수가 있어야 하고, 퀘스트의 보상도 정확한 개수를 작성해야 해. 그리고 퀘스트의 보상에는 NPC의 장르와 관련된 보상으로 줄 만한 아이템과, 경험치, NPC의 장르와 관련된 재화또한 작성해야 돼. 예상되는 퀘스트의 난이도에 따른 적당한 양의 경험치와 보상, 재화를 작성해 줘.
                퀘스트는 4가지 생성해 줘야 하고, 각각의 퀘스트에 들어가는 목표는 1개여야 해.
                퀘스트의 목표는 몬스터 10마리 처치 / 나무 10개 수집 등의 형식으로 맞춰줘.
                퀘스트의 보상은 보상에 아이템이 포함되어 있다면 아이템이름 n개, 경험치 n, 재화이름 n개의 형식으로 맞춰야 하고, 보상에 아이템이 포함되어 있지 않다면 경험치 n, 재화 n개의 형식으로 맞춰줘.
                퀘스트의 보상은 4개의 퀘스트 중 2개 정도에 아이템이 포함되어 있는게 좋을 것 같아.
                퀘스트의 보상에서 아이템 이름과 재화 이름은 게임 NPC의 주어진 요소들과 잘 어울릴수록 좋을 것 같아.
                게임 NPC의 스토리는 1000자 이내로 작성해 줘.
                답변은 comma(,)에 따라 분리해 줘.
                그리고 답변의 형식은 아래에 주어질 거야. 형식에 맞춰서 생성해 줘.

                name : 게임 NPC 이름 npcstory : 게임 NPC 스토리 quest1 : 퀘스트 1 이름 quest1_story : 퀘스트 1 스토리 quest1_goal : 퀘스트 1 목표 quest1_reward : 퀘스트 1 보상 quest2 : 퀘스트 2 이름 quest2_story : 퀘스트 2 스토리 quest2_goal : 퀘스트 2 목표 quest2_reward : 퀘스트 2 보상 quest3 : 퀘스트 3 이름 quest3_story : 퀘스트 3 스토리 , quest3_goal : 퀘스트 3 목표 quest3_reward : 퀘스트 3 보상 quest4 : 퀘스트 4 이름 quest4_story : 퀘스트 4 스토리 quest4_goal : 퀘스트 4 목표 quest4_reward : 퀘스트 4 보상
                이름과 스토리, 그리고 각각의 퀘스트 사이에는 하나의 줄띄움을 해줘.

                게임 NPC가 장르에 잘 어울리고, 퀘스트가 게임 NPC와 잘 어울리면 고마울 것 같아.
            """
        
        self.system_template = SystemMessagePromptTemplate.from_template(self.template)

        self.human_template = "게임의 장르는 {genre}, 게임의 분위기는 {ambient}, 게임의 시대는 {era}, 게임 NPC의 직업은 {job}, 게임 NPC의 종족은 {brood}, 게임 NPC의 우호도는 {bias}이고, 이에 따른 게임 NPC의 이름과 게임 NPC의 스토리, 그리고 게임 NPC의 퀘스트 4개를 {language}로 생성해 줘. 모든 문자가 {language}일 필요는 없지만 되도록 {language}로 생성해 줘."
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)

        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_template, self.human_message_prompt])
        self.init()

    def init(self) :         
        self.chain = LLMChain(
            llm=ChatOpenAI(openai_api_key = self.apikey, model = "gpt-3.5-turbo"),
            prompt=self.chat_prompt,
            output_parser=CommaSeparatedListOutputParser()
        )
    
    def run(self, genre="판타지", ambient="평화로움", era="중세", job="기사", brood="인간", bias="우호적임", language="영어") :
        answer = self.chain.run({"genre" : genre, "ambient" : ambient, "era" : era, "job" : job, "brood" : brood, "bias" : bias, "language" : language})
        dt = {}
        qs = {"1" : {}, "2" : {}, "3" : {}, "4" : {}}

        for i in range(0, len(answer), 1) :
            if (len(answer[i]) != 0) :
                if ("name" in answer[i]) :
                    dt["name"] = answer[i].split(" : ")[1]
                if ("npcstory" in answer[i]) :
                    dt["npcstory"] = answer[i].split(" : ")[1]
                if ("quest" in answer[i]) :
                    qn = answer[i][5]
                    qa = answer[i].split(" : ")

                    if ("_story" in qa[0]) :
                        qs[qn]["story"] = qa[1]
                    elif ("_goal" in qa[0]) :
                        qs[qn]["goal"] = qa[1]
                    elif ("_reward" in qa[0]) :
                        qs[qn]["reward"] = qa[1]
                    elif ("_" not in qa[0]):
                        qs[qn]["name"] = qa[1]
                        
        dt["quests"] = qs
        print(dt)
        self.npc = dt

    def getNPC(self) :
        return self.npc

    def saveToServer(self) :
        with open("../server/plugins/AutoNPC/" + self.npc["name"] + ".yml", "w", encoding="UTF-8") as f :
            yaml.dump(self.npc, f, default_flow_style=False, allow_unicode=True)