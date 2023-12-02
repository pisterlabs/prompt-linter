from ast import literal_eval
import openai
from pydantic import Field
from bullshit_bot_bot_bot.middleware import GenericMessage, with_messages_processed
from telegram import Update
from telegram.ext import ContextTypes

from langchain.chat_models import ChatOpenAI
from bullshit_bot_bot_bot.utils import print_messages
from middleware import telegram_updates_to_generic_thread
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from bullshit_bot_bot_bot import tools

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import SystemMessage


def get_claim_extraction_prompt(
    messages: list[GenericMessage], parser: PydanticOutputParser
):
    return f"""The following is a conversation taking place on a social media platform.
Please identify the most important and/or contentious factual claims made in the conversation.

Conversation:
{print_messages(messages)}

{parser.get_format_instructions()}
"""


def get_evidence(messages: list[GenericMessage]):
    class Parser(BaseModel):
        claims: list[str] = Field(
            ..., description="A list of claims made in the conversation"
        )

    parser = PydanticOutputParser(pydantic_object=Parser)
    prompt = get_claim_extraction_prompt(messages, parser)
    model = ChatOpenAI()
    output = model.call_as_llm(prompt)
    claims = parser.parse(output).claims
    return fact_check_claims(claims)


def fact_check_claims(claims: list[str]):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    tools_ = [
        Tool(
            name="google_search",
            description="Search google",
            func=tools.GoogleSearchAPIWrapperWithLinks().run,
        ),
        Tool(
            name="read_web_page",
            description="Get quotes that may be relevant for the claims from a web page",
            func=tools.WebPagePassageExtractor(claims=claims).run,
        ),
    ]
    agent_kwargs = {
        "system_message": SystemMessage(content="You are a claim verification bot.")
    }
    agent = initialize_agent(
        tools_,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )
    claim_section = "\n".join([f"Claim: {claim}" for claim in claims])
    return agent.run(
        f"Please find evidence that supports or refutes the following claims:\n{claim_section}. Hint: to find evidence that refutes a claim, try googling for its negation. "
        "Investigate a few of the most relevant-looking sources. When you are done, please write a succinct summary of your findings, "
        "citing your sources inline, using markdown."
    )


async def get_sources(update: Update, context: ContextTypes.DEFAULT_TYPE):
    messages = telegram_updates_to_generic_thread(context.chat_data.get("updates", []))

    response_text = get_topic_list(messages)

    # ... search google
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=response_text,
    )


def get_topic_list(messages: list[GenericMessage]):
    model = ChatOpenAI()
    prompt = get_topic_extraction_prompt(messages)
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        functions=[
            {
                "name": "verify_claims",
                "description": "Verify that the claims are factual given a list of sentences",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sentences": {
                            "type": "array",
                            "description": "A list of sentences to be verified. Maximum 5 sentences.",
                            "items": {
                                "type": "string",
                                "description": "A single sentence that summarizes one of the key points of the news article.",
                            },
                        }
                    },
                    "required": ["sentences"],
                },
            }
        ],
        function_call={
            "name": "verify_claims",
        },
    )
    print(chat_completion)
    arguments = chat_completion.choices[0].message.function_call.arguments
    # arguments is a string representation of {sentences: ["sentence1", "sentence2", ...]}
    arguments_as_dict = literal_eval(arguments)
    content = arguments
    return content


def get_topic_extraction_prompt(messages: list[GenericMessage]):
    message_prompt_section = print_messages(messages)

    return f"""The following is a conversation taking place on a social media platform.
It may contain at least one message that is a news article or a summary of a news article.
Your task is to summarize the latest news article that is being discussed in the conversation into a list of sentences split by newlines.

Conversation:
{message_prompt_section}
"""


@with_messages_processed
async def evidence(
    messages: list[GenericMessage], update: Update, context: ContextTypes.DEFAULT_TYPE
):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=get_evidence(messages)
    )
