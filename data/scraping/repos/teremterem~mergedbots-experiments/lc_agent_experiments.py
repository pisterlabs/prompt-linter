import secrets
from pathlib import Path

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import PromptLayerChatOpenAI
from mergedbots import MergedBot
from mergedbots.experimental.sequential import SequentialMergedBotWrapper, ConversationSequence

from experiments.common.bot_manager import SLOW_GPT_MODEL, bot_manager
from experiments.common.repo_access_utils import ListRepoTool, ReadFileTool, WriteFileTool


@SequentialMergedBotWrapper(bot_manager.create_bot(handle="LcAgentExperiments"))
async def lc_agent_experiments(bot: MergedBot, conv_sequence: ConversationSequence) -> None:
    root_dir = (Path(__file__).parents[3] / "mergedbots").as_posix()
    tools = [
        ListRepoTool(root_dir=root_dir),
        ReadFileTool(root_dir=root_dir),
        WriteFileTool(root_dir=root_dir),
    ]

    model_name = SLOW_GPT_MODEL
    message = await conv_sequence.wait_for_incoming()
    await conv_sequence.yield_outgoing(await message.service_followup_for_user(bot, f"`{model_name}`"))

    chat_llm = PromptLayerChatOpenAI(
        model_name=model_name,
        temperature=0,
        model_kwargs={
            "user": str(message.originator.uuid),
        },
        pl_tags=["lc_agent_exp", secrets.token_hex(4)],
    )
    react = initialize_agent(tools, chat_llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)

    while True:
        # TODO feed in the dialog history too ?
        await conv_sequence.yield_outgoing(await message.final_bot_response(bot, await react.arun(message.content)))
        message = await conv_sequence.wait_for_incoming()
