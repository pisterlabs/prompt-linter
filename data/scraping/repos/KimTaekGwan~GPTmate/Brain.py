import re
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv


class Brainstoming:
    def __init__(self):
        self.setting()

    def setting(self):
        load_dotenv()

    def set_chain(self, temperature: int = 0.5):
        chat = ChatOpenAI(
            temperature=temperature,
            # max_tokens=,
            # model_name='text-davinci-003',
            openai_api_key=os.getenv("openai")
        )
        system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You are an assistant who helps me brainstorm \
                          to find creative topics using the data given to me.",
                input_variables=[]
            )
        )
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="{res_prompt}",
                input_variables=["res_prompt"]
            )
        )
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        self.chain = LLMChain(llm=chat, prompt=chat_prompt)

    def prompt_text(self, data_prompt: str, field: str, purpose: str, num_topics: int) -> str:
        prompt = f"Generate {num_topics} topics in the {field} field for the following purpose: {purpose}\n\nData:\n"
        prompt += data_prompt
        prompt += "\nOutput:"
        for num in range(1, num_topics+1):
            prompt += f'\n{num}. '
        return prompt

    def generate_prompt(self, data_infos=dict) -> str:
        """
        데이터 정보가 담긴 dict를 받아서 해당 데이터에 대한 프롬프트를 생성합니다.

        Args:
            data_infos (dict): data info(dict)가 담긴 dict
                각 데이터의 id값을 key로 가지고 있습니다.
            data_info (dict): 데이터 정보가 담긴 dict
                - 'data_name' (str): 데이터명
                - 'data_description' (str): 데이터 설명
                - 'columns' (list of dict): 데이터 칼럼 정보가 담긴 dict의 리스트
                    - 'column_name' (str): 칼럼명
                    - 'column_description' (str): 칼럼 설명

        Returns:
            str: 해당 데이터에 대한 프롬프트 문자열입니다.
        """
        prompt = ""
        for n, (_, data) in enumerate(data_infos.items(), start=1):
            prompt += f"{n}. {data['data_name']}\n"
            prompt += f"- data description\n"
            prompt += f"{data['data_description']}\n"
            prompt += f"- columns info\n"
            for column in data['columns']:
                prompt += f"\t- {column['column_name']}: {column['column_description']}\n"
        return prompt

    def process_run(self, data_infos, field: str, purpose: str, num_topics: int):
        self.set_chain()
        data_prompt = self.generate_prompt(data_infos)
        res_prompt = self.prompt_text(data_prompt, field, purpose, num_topics)

        with get_openai_callback() as cb:
            res = self.chain.run(res_prompt=res_prompt)
            topics = re.findall(r"\d+\.\s(.+)", res)
            # print(cb)
            res_tokens = cb.total_tokens
            # print(res_tokens)
        topics = re.findall(r"\d+\.\s(.+)", res)
        return {'status': True,
                'msg': 'Brainstoming Idea',
                'data': topics,
                'total_tokens': res_tokens}


if __name__ == '__main__':

    data_infos = {
        12345: {
            'data_name': 'iris',
            'data_description': '붓꽃 데이터셋',
            'columns': [
                {'column_name': 'sepal_length', 'column_description': '꽃받침 길이'},
                {'column_name': 'sepal_width', 'column_description': '꽃받침 너비'},
                {'column_name': 'petal_length', 'column_description': '꽃잎 길이'},
                {'column_name': 'petal_width', 'column_description': '꽃잎 너비'},
                {'column_name': 'class', 'column_description': '붓꽃 종류'}
            ]
        },
        12346: {
            'data_name': 'titanic',
            'data_description': '타이타닉 호 생존자 데이터',
            'columns': [
                {'column_name': 'survived',
                 'column_description': '생존 여부 (0: 사망, 1: 생존)'},
                {'column_name': 'pclass',
                 'column_description': '선실 등급 (1, 2, 3 중 하나)'},
                {'column_name': 'sex', 'column_description': '성별'},
                {'column_name': 'age', 'column_description': '나이'},
                {'column_name': 'fare', 'column_description': '운임'},
                {'column_name': 'embarked',
                 'column_description': '승선 항구 (C = Cherbourg, Q = Queenstown, S = Southampton)'}
            ]
        }
    }
    field = '교육'
    purpose = '머신러닝 연습'
    num_topics = 5

    brain = Brainstoming()
    print(brain.process_run(data_infos, field, purpose, num_topics))
