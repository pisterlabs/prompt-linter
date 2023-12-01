from typing import Callable, List

from langchain.agents.agent import AgentExecutor
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_contrib.prompts import ChainedPromptTemplate, Templatable

from zamm.actions.base import Action
from zamm.actions.edit_file import EditFile
from zamm.actions.finish import Finish
from zamm.actions.follow_tutorial import FollowTutorial
from zamm.actions.note import MakeNote
from zamm.actions.use_terminal import UseTerminal
from zamm.chains.general import ActionChain, LaxSequentialChain
from zamm.chains.general.choice.base import ChoicePickerChain
from zamm.chains.general.choice.prompt import ChoicePromptTemplate
from zamm.chains.general.llm import ZLLMChain


def default_action_chain(
    llm: BaseLLM,
    prefix: Templatable,
    terminal_chain: Chain,
    agent_creator: Callable[[], AgentExecutor],
    choice_prompt: str = "You have a few actions available to accomplish this: ",
):
    actions: List[Action] = [
        MakeNote.default(llm=llm, prefix=prefix),
        UseTerminal.default(llm=llm, prefix=prefix, terminal_chain=terminal_chain),
        EditFile.default(llm=llm, prefix=prefix),
        FollowTutorial.default(llm=llm, prefix=prefix, agent_creator=agent_creator),
        Finish.default(),
    ]

    action_choice_template = ChoicePromptTemplate(
        prefix=ChainedPromptTemplate([prefix, choice_prompt]),
        choices=[action.choice_text for action in actions],
    )
    action_choice = ChoicePickerChain(
        llm=llm,
        prompt=action_choice_template,
    )
    return ActionChain(choice_picker=action_choice, actions=actions)


def action_with_thought_chain(
    llm: BaseLLM,
    prefix: Templatable,
    terminal_chain: Chain,
    agent_creator: Callable[[], AgentExecutor],
    choice_prompt: str = "You have a few actions available to accomplish this: ",
):
    thought_chain_prompt = ChainedPromptTemplate(
        [
            prefix,
            PromptTemplate(
                input_variables=[],
                template="""
Write down the next step or command in the employee training manual as a single line, along with your reasoning:

> """.lstrip(),  # noqa
            ),
        ],
    )
    thought_chain = ZLLMChain(
        llm=llm,
        output_key="next_step",
        prompt=thought_chain_prompt,
        default_stop=["\n"],
    )

    final_prefix = ChainedPromptTemplate(
        [
            prefix,
            PromptTemplate(
                input_variables=["next_step"],
                template="""
Now, the next step in the employee training manual is (quoted below as a single line):

> {next_step}

""".lstrip(),
            ),
        ],
    )

    action_chain = default_action_chain(
        llm=llm,
        prefix=final_prefix,
        terminal_chain=terminal_chain,
        agent_creator=agent_creator,
        choice_prompt=choice_prompt,
    )

    return LaxSequentialChain(
        input_variables=thought_chain.input_keys,
        output_variables=action_chain.output_keys,
        chains=[
            thought_chain,
            action_chain,
        ],
    )
