from bson import ObjectId
from langchain import LLMChain, PromptTemplate

from controls.llm import chatGpt
from controls.script import get_script
from repository.database.mongoDB import MongoDB
from settings.get_template import get_template


def create_user_response(sentence: str):
    """
    入力文から推測されるユーザーの返答を返す
    """
    template = get_template("CreateResponse")

    prompt = PromptTemplate(template=template, input_variables=["sentence"])

    chain = LLMChain(llm=chatGpt, prompt=prompt)
    result = chain.run({"sentence": sentence})

    return result


def create_user_response_by_script(script_id: str, break_point: int):
    """
    台本からユーザーの返答を生成し、DBに保存する
    Args:
      script_id: str : 返答を作成する対象の台本ID
      breal_point: int : どこまでのセリフから返答を生成するか
    Return:
      返答ID
    """
    script_obj = get_script(script_id)
    scripts = script_obj["scripts"][: break_point + 1]
    script_text = ""
    for script in scripts:
        script_text += "：".join([script["speaker"], script["quote"]])
        script_text += "\n"

    response = create_user_response(script_text)

    client = MongoDB(collection_name="user_responses")
    result = client.insert_one(
        {
            "script_id": script_id,
            "break_point": break_point,
            "response": response,
        }
    )
    client.close_connection()
    return str(result.inserted_id)


def filter_user_responses(script_id: str | None):
    client = MongoDB(collection_name="user_responses")
    if script_id is None:
        data = client.find_many({})
    else:
        data = client.find_many({"script_id": script_id})
    result = [{**datum, "_id": str(datum["_id"])} for datum in data]
    client.close_connection()
    return result


def get_user_response(response_id: str):
    client = MongoDB(collection_name="user_responses")
    data = client.find_one({"_id": ObjectId(response_id)})
    client.close_connection()
    res = {**data, "_id": str(data["_id"])}
    return res


if __name__ == "__main__":
    import pathlib
    from time import sleep

    # s_id = "646668927483c4e9fa4e9a86"
    # res = create_user_response_by_script(s_id, 2)
    # print(res)
    current_dir = pathlib.Path(__file__).parent
    with open(current_dir / "scripts.txt", mode="r", encoding="utf-8") as f:
        ids = f.readlines()
    for s_id in ids:
        res = create_user_response_by_script(s_id, 3)
        print(res)
        with open(
            current_dir / "user_responses.txt", mode="w", encoding="utf-8"
        ) as f:
            f.write(f"{res}\n")
        sleep(10)
