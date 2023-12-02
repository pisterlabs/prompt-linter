from typing import List
from faiss import IndexFlatL2

from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models.base import BaseChatModel
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseRetriever
from langchain.vectorstores import FAISS

from agent.agent import Agent
from world.locations import Locations
from models.local_llamas import vicuna
from utils.callbacks import ConsoleManager

import json
import networkx as nx
import ray


class Simulation:
    llm: BaseChatModel
    long_term_memory: BaseRetriever
    callback_manager: BaseCallbackManager
    world: nx.Graph
    locations: Locations
    agents: List[Agent]

    @classmethod
    def setup(cls):
        cls.llm = vicuna()
        cls.long_term_memory = TimeWeightedVectorStoreRetriever(
            vectorstore=FAISS(
                embedding_function=cls.llm.get_embeddings().embed_query,
                index=IndexFlatL2(6656),  # 5120 13B : 6656 30B
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )
        )
        cls.callback_manager = ConsoleManager([])
        ray.init()

    @classmethod
    def teardown(cls):
        ray.shutdown()

    def create_world(self, world_file_path: str):
        world = nx.Graph()
        with open(world_file_path, "r") as f:
            world_data = json.load(f)

        nodes = world_data["location_data"]
        final_node = None

        for node in nodes.keys():
            world.add_node(node)
            world.add_edge(node, node)

            if final_node is not None:
                world.add_edge(node, final_node)

            final_node = node

        world.add_edge(list(nodes.keys())[0], final_node)

        locations = Locations()
        for name, description in nodes.items():
            locations.add_location(name, description)

        agents = []
        for name, description in world_data["agent_data"].items():
            starting_location = locations.get_random_location()
            agent = self.create_agent(
                name=name,
                description=description["description"],
                traits=description["traits"],
                location=str(starting_location),
            )
            agents.append(agent)

        self.world = world
        self.locations = locations
        self.agents = agents

        return world, locations, agents

    def create_agent(
        self, name: str, description: str, traits: List[str], location: str, **kwargs
    ) -> Agent:
        return Agent(
            **kwargs,
            name=name,
            description=description,
            traits=traits,
            location=location,
            llm=self.llm,
            long_term_memory=self.long_term_memory,
            callback_manager=self.callback_manager,
        )

    def test_planning(self):
        for agent in self.agents:
            print(f"Hello, I am {agent.name} and I am {agent.description}")
            print(f"My traits are {agent.traits}!\n")
            agent.get_plan()
            print(f"My plan for the day is {agent.plan}\n")
