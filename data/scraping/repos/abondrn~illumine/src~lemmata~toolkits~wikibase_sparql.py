import json
from typing import Any, Literal, Optional

from langchain.agents import Tool
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from pydantic import AnyHttpUrl, Field
import requests

from lemmata.langchain.agent_executor import AgentExecutor
from lemmata.langchain.initialize_agent import initialize_agent

PREFIX = """
Answer the following questions by running a sparql query against a wikibase where the p and q items are
completely unknown to you. You will need to discover the p and q items before you can generate the sparql.
Do not assume you know the p and q items for any concepts. Always use tools to find all p and q items.
After you generate the sparql, you should run it. The results will be returned in json.
Summarize the json results in natural language.

You may assume the following prefixes:
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>

When generating sparql:
* Try to avoid "count" and "filter" queries if possible
* Never enclose the sparql in back-quotes

You have access to the following tools:
"""


def get_nested_value(o: dict, path: list) -> Optional[Any]:
    current = o
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError):
            return None
    return current


ToolName = Literal["ItemLookup", "PropertyLookup", "SparqlQueryRunner"]


class SparqlToolkit(BaseToolkit):
    wikidata_user_agent: str
    sparql_endpoint: AnyHttpUrl = Field("https://query.wikidata.org/sparql")
    vocab_endpoint: AnyHttpUrl = Field("https://www.wikidata.org/w/api.php")
    selected_tools: list[ToolName] = Field(["ItemLookup", "PropertyLookup", "SparqlQueryRunner"])
    """If provided, only provide the selected tools. Defaults to all."""

    def run_sparql(
        self,
        query: str,
    ) -> str:
        headers = {"Accept": "application/json"}
        headers["User-Agent"] = self.wikidata_user_agent

        response = requests.get(self.sparql_endpoint, headers=headers, params={"query": query, "format": "json"})

        # TODO: better error reporting
        if response.status_code != 200:
            return "That query failed. Perhaps you could try a different one?"
        results = get_nested_value(response.json(), ["results", "bindings"])
        return json.dumps(results)

    def vocab_lookup(
        self,
        search: str,
        entity_type: Literal["item", "property"] = "item",
        srqiprofile: Optional[str] = None,
    ) -> Optional[str]:
        headers = {"Accept": "application/json"}
        headers["User-Agent"] = self.wikidata_user_agent

        if entity_type == "item":
            srnamespace = 0
            srqiprofile = "classic_noboostlinks" if srqiprofile is None else srqiprofile
        elif entity_type == "property":
            srnamespace = 120
            srqiprofile = "classic" if srqiprofile is None else srqiprofile
        else:
            raise ValueError("entity_type must be either 'property' or 'item'")

        params = {
            "action": "query",
            "list": "search",
            "srsearch": search,
            "srnamespace": srnamespace,
            "srlimit": 1,
            "srqiprofile": srqiprofile,
            "srwhat": "text",
            "format": "json",
        }

        response = requests.get(self.vocab_endpoint, headers=headers, params=params)

        # TODO: better error reporting
        if response.status_code == 200:
            title = get_nested_value(response.json(), ["query", "search", 0, "title"])
            if title is None:
                return f"I couldn't find any {entity_type} for '{search}'. Please rephrase your request and try again"
            # if there is a prefix, strip it off
            return title.split(":")[-1]
        else:
            return "Sorry, I got an error. Please try again."

    def get_tools(self) -> list[BaseTool]:
        allowed_tools = {
            "ItemLookup": Tool(
                name="ItemLookup",
                func=(lambda x: self.vocab_lookup(x, entity_type="item")),
                description="useful for when you need to know the q-number for an item",
            ),
            "PropertyLookup": Tool(
                name="PropertyLookup",
                func=(lambda x: self.vocab_lookup(x, entity_type="property")),
                description="useful for when you need to know the p-number for a property",
            ),
            "SparqlQueryRunner": Tool(
                name="SparqlQueryRunner",
                func=self.run_sparql,
                description="useful for getting results from a wikibase",
            ),
        }
        tools: list[BaseTool] = []
        for tool_name in self.selected_tools:
            tools.append(allowed_tools[tool_name])
        return tools

    def create_agent(
        self,
        llm: BaseLanguageModel,
        agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix: Optional[str] = PREFIX,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Construct a SPARQL agent from an LLM and dataframe."""
        return initialize_agent(llm=llm, agent=agent_type, prefix=prefix, tools=self.get_tools(), **kwargs)
