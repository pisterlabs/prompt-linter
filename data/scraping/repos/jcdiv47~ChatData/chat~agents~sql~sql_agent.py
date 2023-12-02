import json
import re
import pandas as pd
from typing import Union, Optional
from openai import ChatCompletion
from pydantic import BaseModel
from chat.vector_store import Chroma
from chat.agents.sql.prompt import SQL_AGENT_FUNCTIONS, TOKENIZE_WARMUP_MESSAGES
from chat.common.config import DB_URI
from chat.common.log import logger
from chat.model.openai import ChatModel
from chat.agents.base import Agent
from chat.agents.chat import ChatAgent


class Entity(BaseModel):
    name: str
    type: str


class SQLAgent(ChatModel, Agent):
    """ agent for translating natural language to generating sql query """

    def __init__(
            self,
            return_sql: bool = False,
            return_direct: bool = False,
    ):
        super().__init__()
        self.db = Chroma()
        self.functions = SQL_AGENT_FUNCTIONS
        self.return_sql = return_sql
        self.return_direct = return_direct

    def reassign(
            self,
            agent: Agent,
            text: str,
    ):
        pass

    def generate_sql(self, text: str) -> str:
        """
        Generate sql query from input text.
        Only supports single round dialogue for now
        """
        text = self.replace_entities(text)
        logger.debug(text)
        messages = [
            {"role": "system", "content": "Answer user questions by generating SQL queries against the database."},
            {"role": "user", "content": text},
        ]
        response = self.chat_completion_request(
            messages=messages,
            functions=self.functions,
            # function_call={"name": "ask_database"},
        )
        assistant_message = response['choices'][0]['message']
        if assistant_message.get("function_call") is None:
            logger.debug(
                """
                The model does not utilize function_call as desired.
                Reassigning the task to `ChatAgent`.
                """.strip()
            )
            return ChatAgent().clean_up_mess("not-query")
        return json.loads(assistant_message["function_call"]["arguments"])["query"]

    def tokenize(self, text: str) -> list[Entity]:
        """ perform tokenization on the text, find out which entities are in the text """

        # >> TODO need some thorough parsing process to avoid as many errors as possible
        def json_parse(text: str):
            """ parse output text from tokenization and turn into json format(list of dicts) """
            pattern = re.compile(r'\{.*?\}')
            matches = re.findall(pattern, text.replace("\n", ""))
            entity_list = []
            for match in matches:
                try:
                    entity_list.append(json.loads(match))
                except json.JSONDecodeError:
                    logger.warn(f"skipping invalid dictionary: {match}")
            return entity_list

        messages = TOKENIZE_WARMUP_MESSAGES
        messages.append({"role": "user", "content": f"Text: {repr(text)}"})
        # response = self.chat_completion_request(messages=messages)
        # ret = response['choices'][0]['message']['content']
        response = ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        ret = response['choices'][0]['message']['content']
        logger.debug(ret)
        # >> TODO deal with potential pydantic.ValidationError
        return [Entity(**ent) for ent in json_parse(ret)]

    def replace_entities(self, text: str):
        """ replace entities in the text with correct spelling in our database """
        entity_sub = {}
        for entity in self.tokenize(text):
            if entity.type == 'mall':
                entity_sub[entity.name] = self.db.query('mall_name-with-city',
                                                        entity.name,
                                                        n_results=1)["metadatas"][0][0]['mall_name']
            elif entity.type == 'brand':
                entity_sub[entity.name] = self.db.query('brand',
                                                        entity.name,
                                                        n_results=1)["metadatas"][0][0]['brand_name']
            # >> TODO replace with more appropriate approach
            elif entity.type == 'city':
                entity_sub[entity.name] = entity.name.replace("市", "") + "市"
            else:
                logger.debug("unnecessary entity | %s | %s", entity.name, entity.type)
        logger.info("entity_sub: %s", entity_sub)

        for orig, sub in entity_sub.items():
            text = text.replace(orig, sub)
        return text

    @staticmethod
    def read_from_sql(
            sql: str,
            is_dataframe: bool = False
    ) -> Union[pd.DataFrame, list[dict]]:
        df = pd.read_sql(sql=sql, con=DB_URI)
        if is_dataframe:
            return df
        return df.to_dict(orient='records')

    def translate_result(self, input_text, sql, sql_result):
        messages = [
            {
                "role": "user",
                "content": f"Please write a MySQL query to answer the following question. Question: {input_text}"
            },
            {
                "role": "system",
                "content": f"The resulted sql query is:\n{sql}"
            },
            {
                "role": "user",
                "content": "Can you query it for me and fetch the result"
            },
            {
                "role": "system",
                "content": json.dumps(sql_result)
            },
            {
                "role": "user",
                "content": "Can you answer in Chinese and describe the result for me in layman's terms"
            }
        ]
        response = self.chat_completion_request(messages=messages)
        return response['choices'][0]['message']['content']

    def run(
            self,
            input_text: str,
            return_sql: Optional[bool] = None
    ) -> str:
        sql = self.generate_sql(input_text)
        logger.info("generated sql:\n%s", sql)
        if return_sql is None:
            return_sql = self.return_sql
        if return_sql:
            return sql
        sql_result = self.read_from_sql(sql)
        output_text = self.translate_result(input_text, sql, sql_result)
        logger.debug(output_text)
        return output_text


if __name__ == '__main__':
    agent = SQLAgent()
    # print(agent.tokenize("海蓝之谜在上海市长泰的门店数量"))
    # print(agent.db.query('brand', 'Starbucks', n_results=5))
    # print(agent.db.query('mall', ['上海长泰广场', '上海长泰', '上海市长泰'], n_results=1))
    # print(agent.tokenize("海蓝之谜在上海市的面积最大的5个门店所在的商场名称和地址"))
    print(agent.run("兴业太古汇里有多少家护肤化妆品的门店"))
