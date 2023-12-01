from enum import auto, Enum

from langchain import PromptTemplate, LLMChain

from hack_zurich_app.rag import llm_provider


class QueryType(Enum):
    COVERAGE = auto()
    POLICY_INFO = auto()
    UNKNOWN = auto()


# RoutingAgent determines which agent can answer the query:
# - If the query has the format of a coverage check, route to the claim agent
# - Else if the user has a general question related to policies, route to the policy info agent
class RoutingAgent:
    def __init__(self):
        print("Initializing RoutingAgent...")
        self.llm = llm_provider.openai_llm()

    def determine_query_type(self, query: str) -> QueryType:
        if self._is_coverage_query(query):
            return QueryType.COVERAGE
        elif self._is_policy_info_query(query):
            return QueryType.POLICY_INFO
        else:
            return QueryType.UNKNOWN

    def _is_coverage_query(self, query: str):
        return self._check_if_query_matches(
            query,
            "Is the QUERY about checking whether a given situation is covered by some insurance policy?"
        )

    def _is_policy_info_query(self, query: str):
        return self._check_if_query_matches(query, "Is the QUERY a question about insurance policy documents?")

    def _check_if_query_matches(self, query: str, pattern: str, debug_mode=False):
        debug_query = "Explain yourself." if debug_mode else ""

        prompt_text = f"""
            Suppose you work at an insurance company and receive a QUERY from a user.
            You have to answer the question "{pattern}".
            You answer only with YES, NO, INSUFFICIENT_CONTEXT. {debug_query}
            QUERY: '{{query}}'
            """

        prompt = PromptTemplate(
            input_variables=["query"],
            template=prompt_text,
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(query=query).strip()

        if debug_mode:
            print(result)

        return "YES" in result


if __name__ == "__main__":
    support_center = RoutingAgent()

    default_query = "What is house hold insurance?"

    while True:
        input_query = input(f"Enter the query [default:'{default_query}']") or default_query
        result = support_center.determine_query_type(input_query)
        print(result)
