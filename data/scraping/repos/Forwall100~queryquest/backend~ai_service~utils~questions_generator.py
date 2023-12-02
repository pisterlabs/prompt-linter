from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
import time
import asyncio
import os


async def generate_question(schema, cmplx):
    start_time = time.time()

    template = """
  The bot is an artificial intelligence that is geared to only one task: coming up with test questions. It comes up with SQL questions and comes up with only those questions that imply a SELECT query as the answer. The bot receives a database table schema as input and comes up with a question based on it. Questions can be of different complexity, the complexity of the question is measured on a scale of 1 to 5, where 1 is the easiest select * query, and 5 is a query where you need to use several operators join and group by.

  User: Database schema: {schema}
  Come up with a difficulty {cmplx} question to make a sql select query on this database, write the question itself in plain language and the correct answer as a sql query.

  Bot:
  Question: """

    prompt = PromptTemplate(
        template=template, input_variables=["schema", "cmplx"])

    local_path = os.path.dirname(os.path.realpath(__file__)) + '/ggml-gpt4all-j-v1.3-groovy.bin'

    llm = GPT4All(model=local_path)

    schema = schema[:1024-len(template)] + "\n"

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = await asyncio.to_thread(llm_chain.run, schema=schema, cmplx=cmplx)

    return {
        "answer": answer.replace("\\", ""),
        "time": time.time() - start_time
    }
