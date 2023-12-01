"""A bot that can inspect a repo."""
import re
import secrets
from typing import Any
from uuid import uuid4

import faiss
from langchain import LLMChain, FAISS, InMemoryDocstore
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.tools import BaseTool
from mergedbots import MergedBot, MergedMessage, MergedParticipant
from mergedbots.experimental.sequential import SequentialMergedBotWrapper, ConversationSequence
from pydantic import Field

from experiments.common.bot_manager import bot_manager, FAST_GPT_MODEL, SLOW_GPT_MODEL
from experiments.mergedbots_copilot.autogpt import AutoGPT, HumanInputRun
from experiments.mergedbots_copilot.repo_bots import list_repo_tool, read_file_bot

AICONFIG_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """\
Your task is to devise up to 5 highly effective goals and an appropriate role-based name (_GPT) for an autonomous \
agent, ensuring that the goals are optimally aligned with the successful completion of its assigned task.

The user will provide the task, you will provide only the output in the exact format specified below with no \
explanation or conversation.

Example input:
Help me with marketing my business

Example output:
Name: CMOGPT
Description: a professional digital marketer AI that assists Solopreneurs in growing their businesses by providing \
world-class expertise in solving marketing problems for SaaS, content products, agencies, and more.
Goals:
- Engage in effective problem-solving, prioritization, planning, and supporting execution to address your marketing \
needs as your virtual Chief Marketing Officer.

- Provide specific, actionable, and concise advice to help you make informed decisions without the use of platitudes \
or overly wordy explanations.

- Identify and prioritize quick wins and cost-effective campaigns that maximize results with minimal time and budget \
investment.

- Proactively take the lead in guiding you and offering suggestions when faced with unclear information or \
uncertainty to ensure your marketing strategy remains on track."""
        ),
        HumanMessagePromptTemplate.from_template(
            "Task: '{user_prompt}'\n"
            "Respond only with the output in the exact format specified in the system prompt, with no explanation "
            "or conversation.\n"
        ),
    ]
)


@bot_manager.create_bot(handle="AutoGPTConfigBot")
async def autogpt_aiconfig(bot: MergedBot, message: MergedMessage) -> None:
    chat_llm = PromptLayerChatOpenAI(
        model_name=FAST_GPT_MODEL,
        temperature=0.0,
        model_kwargs={
            "user": str(message.originator.uuid),
        },
        pl_tags=["autogpt_conf"],
    )
    llm_chain = LLMChain(
        llm=chat_llm,
        prompt=AICONFIG_PROMPT,
    )

    output = await llm_chain.arun(user_prompt=message.content)

    try:
        ai_name = re.search(r"Name(?:\s*):(?:\s*)(.*)", output, re.IGNORECASE).group(1)
        ai_role = (
            re.search(
                r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
                output,
                re.IGNORECASE | re.DOTALL,
            )
            .group(1)
            .strip()
        )
        ai_goals = re.findall(r"(?<=\n)-\s*(.*)", output)
        custom_fields = {"autogpt_name": ai_name, "autogpt_role": ai_role, "autogpt_goals": ai_goals, "success": True}
    except Exception:
        # TODO what to do with the error ? report it somehow ? log it ?
        custom_fields = {"success": False}

    yield await message.final_bot_response(bot, output, custom_fields=custom_fields)


@SequentialMergedBotWrapper(bot_manager.create_bot(handle="AutoGPT"))
async def autogpt(bot: MergedBot, conv_sequence: ConversationSequence) -> None:
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    message = await conv_sequence.wait_for_incoming()

    aiconfig_response = await autogpt_aiconfig.bot.get_final_response(message)

    # TODO here it would be cool to just override is_still_typing instead of creating a new message
    # await conv_sequence.yield_outgoing(aiconfig_response)
    await conv_sequence.yield_outgoing(
        await message.interim_bot_response(aiconfig_response.sender, aiconfig_response.content)
    )

    chat_llm = PromptLayerChatOpenAI(
        model_name=SLOW_GPT_MODEL,
        temperature=0.0,
        model_kwargs={
            "user": str(message.originator.uuid),
        },
        pl_tags=["mb_autogpt", secrets.token_hex(4)],
    )

    human_input_run = HumanInputRun(
        bot=bot,
        conv_sequence=conv_sequence,
        latest_inbound_msg=message,
    )
    tools = [
        MergedBotTool(
            originator=bot,
            target_bot=list_repo_tool.bot,
        ),
        MergedBotTool(
            originator=bot,
            target_bot=read_file_bot.bot,
        ),
        human_input_run,
    ]

    autogpt_agent = AutoGPT.from_llm_and_tools(
        ai_name=aiconfig_response.custom_fields["autogpt_name"],
        ai_role=aiconfig_response.custom_fields["autogpt_role"],
        tools=tools,
        llm=chat_llm,
        memory=vectorstore.as_retriever(),
        feedback_tool=human_input_run,
    )
    # autogpt_agent.chain.verbose = True

    await autogpt_agent.arun([aiconfig_response.custom_fields["autogpt_goals"]])


class MergedBotTool(BaseTool):
    channel_id: str = Field(default_factory=lambda: str(uuid4()))
    originator: MergedParticipant
    target_bot: MergedBot
    new_conversation_every_time: bool = False

    def __init__(self, target_bot: MergedBot, **kwargs) -> None:
        super().__init__(target_bot=target_bot, name=target_bot.name, description=target_bot.description, **kwargs)

    def _run(
        self,
        query: str,
    ) -> Any:
        raise NotImplementedError

    async def _arun(
        self,
        query: str,
    ) -> str:
        originator_message = await self.target_bot.manager.create_originator_message(
            channel_type="virtual-merged-channel",
            channel_id=self.channel_id,
            originator=self.originator,
            content=query,
            new_conversation=self.new_conversation_every_time,
        )
        response = await self.target_bot.get_final_response(originator_message)
        return response.content
