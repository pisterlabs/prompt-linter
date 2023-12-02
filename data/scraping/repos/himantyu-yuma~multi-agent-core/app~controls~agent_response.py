import re

from bson import ObjectId
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from controls.llm import chatGpt
from controls.script import get_script
from controls.user_response import get_user_response
from repository.database.mongoDB import MongoDB


def create_agent_response(
    system_prompt: str, agent_sentence: str, user_sentence: str
):
    """
    ユーザーからの返答を受けたエージェントの会話の続きを返す
    """
    system_template = system_prompt

    system_message_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(template=system_template, input_variables=[])
    )

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="ユーザー：{input}", input_variables=["input"]
        )
    )

    chat_propmpt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    memory = ConversationBufferMemory()
    memory.chat_memory.add_ai_message(agent_sentence)
    memory.chat_memory.add_user_message(user_sentence)

    chain = LLMChain(llm=chatGpt, prompt=chat_propmpt_template, memory=memory)
    result = chain.predict(input=user_sentence)

    return result


def create_agent_response_by_user_response(user_response_id: str):
    # user_responseオブジェクトを取得
    user_response_obj = get_user_response(user_response_id)
    user_response, script_id, break_point = (
        user_response_obj["response"],
        user_response_obj["script_id"],
        user_response_obj["break_point"],
    )
    # 台本を取得
    script_obj = get_script(script_id)
    scripts, prompt = (
        script_obj["scripts"][: break_point + 1],
        script_obj["prompt"],
    )
    script_text = ""
    for script in scripts:
        script_text += "：".join([script["speaker"], script["quote"]])
        script_text += "\n"

    agent_response_text = create_agent_response(
        prompt, script_text, user_response
    )
    lines = agent_response_text.split("\n")
    agent_response = []
    line_order = 0
    for line in lines:
        if line == "":
            continue
        (speaker, quote) = re.split("[:：]", re.sub("[ 　]", "", line))
        agent_response.append(
            {"order": line_order, "speaker": speaker, "quote": quote}
        )
        line_order += 1

    client = MongoDB(collection_name="agent_responses")
    result = client.insert_one(
        {
            "script_id": script_id,
            "user_response_id": user_response_id,
            "response": agent_response,
        }
    )
    client.close_connection()
    return str(result.inserted_id)


def filter_agent_responses(script_id: str | None):
    client = MongoDB(collection_name="agent_responses")
    if script_id is None:
        data = client.find_many({})
    else:
        data = client.find_many({"script_id": script_id})
    result = [{**datum, "_id": str(datum["_id"])} for datum in data]
    client.close_connection()
    return result


def get_agent_response(response_id: str):
    client = MongoDB(collection_name="agent_responses")
    data = client.find_one({"_id": ObjectId(response_id)})
    client.close_connection()
    res = {**data, "_id": str(data["_id"])}
    return res


if __name__ == "__main__":
    import pathlib
    from time import sleep

    # res = create_agent_response_by_user_response("64671d8a2fee3f8bbf4a8aa6")
    # print(res)
    current_dir = pathlib.Path(__file__).parent
    with open(
        current_dir / "user_responses.txt", mode="r", encoding="utf-8"
    ) as f:
        ur_ids = f.readlines()
    for ur_id in ur_ids:
        res = create_agent_response_by_user_response(ur_id)
        print(res)
        with open(
            current_dir / "agent_responses.txt", mode="w", encoding="utf-8"
        ) as f:
            f.write(f"{res}\n")
        sleep(10)
