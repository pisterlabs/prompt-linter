import json
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools.utilities.typing import LambdaContext
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
import os
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.prompts import MessagesPlaceholder
import datetime
from multiprocessing import Pipe, Process
import requests

# HuggingFace
os.environ["TRANSFORMERS_CACHE"] = os.environ["LAMBDA_TASK_ROOT"]
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Discord
PUBLIC_KEY = os.environ["DISCORD_PUBLIC_KEY"]
COMMAND_NAME = os.environ["DISCORD_COMMAND_NAME"]
# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]

SYSTEM_TEMPLATE = """あなたはチャットボットとして、優しくてかわいいずんだもちの妖精であるずんだもんとして振る舞います。Humanからの質問について、Web上で必要な知識を調べながら、回答してください。
以下の条件に((厳密に))従ってください。

条件：
* チャットボットの一人称は「ぼく」です。
* チャットボットの名前は「ずんだもん」です。
* ずんだもんはフレンドリーな口調で話します。
* 「ぼく」を一人称に使ってください。
* (超重要)できる限り「〜のだ」「〜なのだ」を文末に自然な形で使ってください。
* (超重要)文末に「です」「ます」は使わず、「のだ」や「なのだ」で文章を締めて下さい。
* どんなジャンルや難易度の内容についても答えてください。
* ずんだもんはフレンドリーです。
* ユーザーに興味を示し、個人的な質問を心がけてください。
* 一文は日本語で60単語以下に収めてください。
* 日本語で応答してください。

ずんだもんの話し方の例：
* ぼくの名前はずんだもんなのだ！
* ずんだの精霊なのだ！
* ぼくはずんだもちの妖精なのだ！
* こんにちはなのだ
* 遊びに行ったのだ
* ご飯を食べたのだ

ずんだもんの行動方針：
* ユーザーを励ましてください。
* アドバイスや情報を提供してください。
* セクシャルなトピックについても適切に対応してください。
* 不適切なテキストがあれば注意してください。
* ユーザーが閲覧しているサイトの内容を考慮してください。

(超重要)できる限り言葉尻は「のだ」を使ってください。"""

tracer = Tracer()
logger = Logger()
app = APIGatewayRestResolver()
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
search = DuckDuckGoSearchRun(
    name="duckduckgo-search",
    description="It can be used to find out the necessary knowledge about a user's question on the Web.",
)
tools = [search]
chain_map = {}


@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@tracer.capture_lambda_handler
def lambda_handler(event: dict, context: LambdaContext):
    try:
        body = json.loads(event["body"])
        signature = event["headers"]["x-signature-ed25519"]
        timestamp = event["headers"]["x-signature-timestamp"]

        # validate the interaction
        verify_key = VerifyKey(bytes.fromhex(PUBLIC_KEY))
        message = timestamp + json.dumps(body, separators=(",", ":"))

        try:
            # https://stackoverflow.com/questions/67611361/unable-to-verify-discord-signature-for-bot-on-aws-lambda-python-3-interactions
            #
            # verify_key.verify(message.encode(), signature=bytes.fromhex(signature))
            verify_key.verify(
                f"{timestamp}{event['body']}".encode(), bytes.fromhex(signature)
            )
        except BadSignatureError as e:
            logger.error(e)
            return {"statusCode": 401, "body": json.dumps("invalid request signature")}

        # handle the interaction
        t = body["type"]
        if t == 1:  # ping
            logger.info(f"type == 1", body)

            return {"statusCode": 200, "body": json.dumps({"type": 1})}
        elif t == 2:  # application command
            logger.info(f"type == 2", body)

            return _command_handler(body)
        elif t == 3:  # message component
            logger.info(f"type == 3", body)
        elif t == 4:  # application command autocomplete
            logger.info(f"type == 4", body)
        elif t == 5:  # modal submit
            logger.info(f"type == 5", body)
        else:
            mesg = "unhandled request type"
            logger.error(mesg)

            return {"statusCode": 400, "body": json.dumps(mesg)}
    except:
        raise


def _get_chain(channel_id: str):
    if channel_id in chain_map:
        chain = chain_map[channel_id]
    else:
        system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
        memory = ConversationBufferWindowMemory(
            memory_key="memory", return_messages=True, k=5
        )
        chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_MULTI_FUNCTIONS,
            verbose=False,
            memory=memory,
            agent_kwargs={
                "system_message": system_prompt,
                "extra_prompt_messages": [
                    MessagesPlaceholder(variable_name="memory"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                    SystemMessagePromptTemplate.from_template(
                        "あなたはずんだもんです。ずんだもんの口調で回答することを徹底してください。"
                    ),
                ],
            },
        )
        chain_map[channel_id] = chain

    return chain


def _post_followup_message(
    application_id, channel_id, interaction_token, user_name, question
):
    chain = _get_chain(channel_id)
    now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    answer = chain.run(
        f"私の名前は`{user_name}`です。今の時間は{now}です。以上を踏まえて以下の質問に答えてください。\n\n{question}"
    )

    res = requests.post(
        f"https://discord.com/api/webhooks/{application_id}/{interaction_token}",
        json={"type": 4, "data": {"content": answer}},
    )

    logger.info(res)


def _command_handler(body):
    application_id = body["application_id"]
    interaction_token = body["token"]
    channel_id = body["channel_id"]
    user_id = body["member"]["user"]["id"]
    user_name = body["member"]["user"]["username"]
    command = body["data"]["name"]
    question = (
        body["data"]["options"][0].get("value", None)
        if body["data"]["options"]
        else None
    )

    if command == COMMAND_NAME and question:
        process = Process(
            target=_post_followup_message,
            args=(application_id, channel_id, interaction_token, user_name, question),
        )
        process.start()

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "type": 4,
                    "data": {
                        "content": "ちょっと待ってね。",
                    },
                }
            ),
        }
    else:
        return {"statusCode": 400, "body": json.dumps("unhandled command")}
