import os
import re
import time
import traceback
from typing import List, Optional, Union

import openai

from config import COLUMN_DESCRIPTION_FILE, DB_FILE, OPENAI_API_KEY
from sql_common import SQLExecutor

openai.api_key = OPENAI_API_KEY


class OpenAI:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def complete(self, prompt: str, temperature: float = 1.0, max_tokens: int = 300, stop: Optional[str] = None) -> str:
        for _ in range(10):
            try:
                query = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop)["choices"][0]["message"]["content"]
                cleaned_query = re.search("```sql.*```", query, re.DOTALL)
                if cleaned_query:
                    query = cleaned_query[0]
                    query = query.removeprefix("```sql").removesuffix("```")
                return query
            except Exception as e:
                traceback.print_exc()
                time.sleep(10)
        raise Exception("OpenAI API not working")


class SQLGenerator:
    """
    Simple prompt node implementation.
    There seem to be many considerations for the eventual version for v2, but this should be good enough for now.
    """
    def __init__(self, model, prompt: str, temperature: float = 1.0, max_tokens: int = 300, stop: Optional[str] = None) -> None:
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.model = model

    def generate(self, question: str = "", temperature: Optional[float] = None, max_tokens: Optional[int] = None, stop: Optional[Union[str, List[str]]] = None): # prepend and append have an Optional type. This is wrong! However, it is necessary as long as https://github.com/deepset-ai/canals/issues/105 isn't fixed.
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        stop = stop or self.stop
        return self.model.complete(self.prompt + question + "\nQuery: ", temperature=temperature, max_tokens=max_tokens, stop=stop)


def load_base_generator(
        schema: str,
        raw_desc_path: os.PathLike=None,
        use_few_shot: bool=False,
    ) -> SQLGenerator:

    raw_desc_prompt = ""
    if raw_desc_path is not None:
        with open(raw_desc_path, "r") as f:
            survey_table = f.read()
        raw_desc_prompt = f"The questions asked are described in the following table: \n{survey_table}"

    few_shot_prompt = ""
    if use_few_shot:
        few_shot_prompt = """
Also, for percentage calculations over responses use either the main table "responses" accounting for nulls or distinct values from the Associative table. Not full counts from the Associative Tables.
Some examples presented below:
### 
Question: What percentage of the respondents have worked with which of the cloud platforms?
Query: SELECT Platforms.Name AS CloudPlatform, COUNT(*) * 100.0 / (select count(distinct(Response_id)) from Response_PlatformHaveWorkedWith) as percentage
  FROM Response_PlatformHaveWorkedWith
  JOIN Responses ON Responses.id = Response_PlatformHaveWorkedWith.Response_id
  JOIN Platforms ON Platforms.id = Response_PlatformHaveWorkedWith.Platforms_id
  GROUP BY Platforms.Name
Question: What is the percentage breakdown of Education level among professional developers?
Query: SELECT EdLevel, count(ResponderDescription) * 100.0 / sum(count(ResponderDescription)) over () as pct from Responses where ResponderDescription = 'I am a developer by profession' group by EdLevel
###""" # noqa

    prompt = f"""A database was created based on survey results. {raw_desc_prompt}
Please return an SQLite query that answers the given question.
The following is the schema of the database containing the survey result tables with some example rows or field-wise distinct values:
{schema}
Please return an SQLite query that answers the following question. Account for NULL values. If you need to make assumptions, do not state them and do not explain your query in any other way. Please make sure to disregard null entries.
{few_shot_prompt}
Question: """ # noqa
    query_generator = SQLGenerator(model=OpenAI("gpt-4"), prompt=prompt, temperature=0)
    return query_generator


if __name__ == "__main__":
    executor = SQLExecutor(file=DB_FILE, descriptions_file=COLUMN_DESCRIPTION_FILE)
    schema = "\n".join(executor.get_table_reprs(num_distinct=20).values())
    generator = load_base_generator(
        schema,
        use_few_shot=True
        # raw_desc_path=RAW_DESCRIPTION_FILE
    )
    query = generator.generate("What is the percentage distribution on how the respondents learn to code?")
    results, _ = executor.execute(query)
    print(query, results)
