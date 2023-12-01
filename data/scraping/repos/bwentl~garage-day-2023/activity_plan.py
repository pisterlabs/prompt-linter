import sys
import os
import warnings
import random

from langchain.agents import (
    initialize_agent,
    Tool,
    AgentType,
)

sys.path.append("./")
from src.models import LlamaModelHandler
from src.agent_tool_selection import AgentToolSelection
from src.docs import DocumentHandler
from src.tools import ToolHandler
from src.memory_store import PGMemoryStoreSetter, PGMemoryStoreRetriever
from src.util import get_secrets, get_word_match_list, agent_logs
from src.docs import AggregateRetrieval
from src.prompts.day_plan import (
    CHAIN_DAY_PLAN_FIRST_ACTIVITY,
    CHAIN_DAY_PLAN_FIRST_ACTIVITY_INPUT,
    CHAIN_DAY_PLAN_NEXT_ACTIVITY,
    CHAIN_DAY_PLAN_NEXT_ACTIVITY_INPUT,
    CHAIN_DAY_PLAN_OVERALL_GOAL,
)


class ActivityPlan:
    """Generates a plan of activities for the day"""

    def retrieve_identity(self, memory_store_retriever):
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 1,
            "recency_wt": 1,
            "importance_wt": 0,
            "relevance_wt": 0,
            "mem_type": "identity",
        }
        identity_statement_query = memory_store_retriever.get_relevant_documents(
            "",
            **memory_kwargs,
        )
        if len(identity_statement_query) > 0:
            identity_statement = identity_statement_query[0].page_content
        else:
            # initial identity statement
            identity_statement = "This is Llama at Home, Llama for short. I am a generative agent build on an open-source, self-hosted large language model (LLM). I can understand and communicate fluently in English and other languages. I can also provide information, generate content, and help with various tasks. Some of my recent actions include answering questions and telling jokes. My short term plan is to continue chatting with others and learning about the world using tools. My long term plan is to improve my skills and knowledge by learning with tools like web searches and from feedback from others."

        return identity_statement

    def retrieve_last_plan(self, memory_store_retriever):
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 1,
            "recency_wt": 1,
            "importance_wt": 0,
            "relevance_wt": 0,
            "mem_type": "plan",
        }
        plan_query = memory_store_retriever.get_relevant_documents(
            "",
            **memory_kwargs,
        )
        if len(plan_query) > 0:
            last_plan = plan_query[0].page_content
        else:
            # initial day plan
            last_plan = """"1. Research[Penguins and their characteristics]
2. Write[Poem about penguins]
3. Research[Flightless birds, general]
4. News[scientific news, with a focus on biology]
5. Reflect[penguins]"""

        return last_plan

    def get_recent_memories(self, memory_store_retriever):
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 10,
            "relevance_wt": 0,
            "importance_wt": 0.5,
            "recency_wt": 2,
            "update_access_time": False,
        }

        recent_memories = AggregateRetrieval(
            vectorstore_retriever=memory_store_retriever
        ).run("", **memory_kwargs)

        return recent_memories

    def get_activity_list(self):
        eligible_activity_list = [
            "- Research: learn about the world, culture, or science.",
            "- News: learn about recent events.",
            "- Write: write a story, joke, poem, etc.",
            "- Reflect: consider what you have learned and what it means for you.",
        ]
        random.shuffle(eligible_activity_list)
        eligible_activity_string = "\n".join(eligible_activity_list)

        return eligible_activity_string

    def generate_day_plan(self, llm, memory_store_retriever, num_activities=5):
        identity_statement = self.retrieve_identity(memory_store_retriever)
        last_plan = self.retrieve_last_plan(memory_store_retriever)
        recent_memories = self.get_recent_memories(memory_store_retriever)
        activity_list = self.get_activity_list()

        first_activity_selection = llm(
            CHAIN_DAY_PLAN_FIRST_ACTIVITY.replace(
                "{identity_statement}", identity_statement
            )
            .replace("{last_plan}", last_plan)
            .replace("{recent_memories}", recent_memories)
            .replace("{activity_list}", activity_list)
        ).strip()
        # TODO: If the activity is not a valid selection, choose the closest activity, or one at random.

        first_activity_input = llm(
            CHAIN_DAY_PLAN_FIRST_ACTIVITY_INPUT.replace(
                "{identity_statement}", identity_statement
            )
            .replace("{last_plan}", last_plan)
            .replace("{recent_memories}", recent_memories)
            .replace("{activity_list}", activity_list)
            .replace("{first_activity}", first_activity_selection)
        ).strip()

        plan_under_construction = (
            f"1. {first_activity_selection}[{first_activity_input}]"
        )

        # TODO: remove an activity from the list if it's already been done twice
        for n in range(2, num_activities + 1):
            activity_list = self.get_activity_list()

            next_activity_selection = llm(
                CHAIN_DAY_PLAN_NEXT_ACTIVITY.replace(
                    "{identity_statement}", identity_statement
                )
                .replace("{last_plan}", last_plan)
                .replace("{recent_memories}", recent_memories)
                .replace("{activity_list}", activity_list)
                .replace("{plan_under_construction}", plan_under_construction)
            ).strip()
            # TODO: If the activity is not a valid selection, choose the closest activity, or one at random.

            next_activity_input = llm(
                CHAIN_DAY_PLAN_NEXT_ACTIVITY_INPUT.replace(
                    "{identity_statement}", identity_statement
                )
                .replace("{last_plan}", last_plan)
                .replace("{recent_memories}", recent_memories)
                .replace("{activity_list}", activity_list)
                .replace("{plan_under_construction}", plan_under_construction)
                .replace("{next_activity}", next_activity_selection)
            ).strip()

            plan_under_construction = f"{plan_under_construction}\n{n}. {next_activity_selection}[{next_activity_input}]"

        day_plan = plan_under_construction

        overall_goal_text = llm(
            CHAIN_DAY_PLAN_OVERALL_GOAL.replace(
                "{identity_statement}", identity_statement
            ).replace("{day_plan}", day_plan)
        ).strip()
        overall_goal = f"My goal today is to {overall_goal_text}"

        # TODO: Add memories for (1) plan and (2) overall goal

        return [day_plan, overall_goal]


if __name__ == "__main__":
    # model_name = "llama-13b"
    # lora_name = "alpaca-gpt4-lora-13b-3ep"
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"
    testAgent = LlamaModelHandler()
    eb = testAgent.get_hf_embedding()

    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    memory_store_setter = PGMemoryStoreSetter(embedding=eb)
    memory_store_retriever = PGMemoryStoreRetriever(embedding=eb)

    activity_plan = ActivityPlan()
    [day_plan, overall_goal] = activity_plan.generate_day_plan(
        llm=pipeline, memory_store_retriever=memory_store_retriever
    )

    print("done")
