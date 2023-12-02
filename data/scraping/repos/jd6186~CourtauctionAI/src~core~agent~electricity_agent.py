from datetime import datetime, timedelta

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

from src.core.model.chatbot_model import OpenAiModel


# 환경변수에서 API 키를 가져온다.
class ElectricityAgent():
    tools = [
        Tool(
            name="Datetime",
            description="I check the time.",
            func=lambda x: f"""
            오늘 : {datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초")},
            내일 : {(datetime.now() + timedelta(days=1)).strftime("%Y년 %m월 %d일 %H시 %M분 %S초")},
            모레 : {(datetime.now() + timedelta(days=2)).strftime("%Y년 %m월 %d일 %H시 %M분 %S초")},
            모래 : {(datetime.now() + timedelta(days=2)).strftime("%Y년 %m월 %d일 %H시 %M분 %S초")},
            어제 : {(datetime.now() - timedelta(days=1)).strftime("%Y년 %m월 %d일 %H시 %M분 %S초")},
            엇그제 : {(datetime.now() - timedelta(days=2)).strftime("%Y년 %m월 %d일 %H시 %M분 %S초")},
            그제 : {(datetime.now() - timedelta(days=2)).strftime("%Y년 %m월 %d일 %H시 %M분 %S초")},
            그그제 : {(datetime.now() - timedelta(days=3)).strftime("%Y년 %m월 %d일 %H시 %M분 %S초")},
            ...
            """,
        ),
        Tool(
            name="Workplace",
            description="I check the workplace.",
            func=lambda x: f"""
            아난티: 1000000000
                코드: 1000000001
                부산: 1000000002
                서울: 1000000003
            코오롱: 2000000000
                마곡: 2000000001
                과천: 2000000002
            메가존: 3000000000
                과천: 3000000001
            """
        ),
        Tool(
            name="Electricity Data",
            description="I check the workplace.",
            func=lambda x: """
            1000000001: {'2023-11-22': '100w'}
            1000000002: {'2023-11-22': '200w'}
            1000000003: {'2023-11-22': '300w'}
            2000000001: {'2023-11-22': '400w'}
            2000000002: {'2023-11-22': '500w'}
            3000000001: {'2023-11-22': '600w'}
            """
        )
    ]

    def __init__(self, llm):
        self.agent = initialize_agent(
            self.tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

    def run(self, query: str, my_workplace_list: list):
        print("Check ######################################################")
        context = f"""
        The result for areas not found in my work_place_list is always "Not available for retrieval."\n
        my_workplace_list: {str(my_workplace_list)}\n
        Question: {query}
        """
        return self.agent.run(context)
