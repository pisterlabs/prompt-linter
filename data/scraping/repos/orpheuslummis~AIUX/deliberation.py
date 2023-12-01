# TODO text doesn't disappear upon change
# TODO proper logigng

from dataclasses import dataclass, field
import openai
import streamlit as st

from utils import RequestParams, new_logger, get_params_from_env, request

log = new_logger("deliberation")

TEMPERATURE = 0.6
MAX_TOKENS = 300
N = 4


@dataclass
class Params:
    prompt: str


class Pipeline:
    name: str

    def run(self, _: Params):
        pass


@dataclass
class PipelineResults:
    text: str
    intermediate_text: list[str] = field(default_factory=list)


class Dummy(Pipeline):
    name = "dummy"


class Critic(Pipeline):
    """
    Pipeline that critics a prompt in various ways (expansion), then aggregates them into a short summary.
    """

    name = "critic"
    default_n = 5
    default_max_tokens = 300
    default_temperature = 0.777

    critics = {
        "critic1": """
        Provide criticism for the following idea:

        {idea}
        """,
        "critic2": """
        List points of potential lack of clarity, robustness, coherence, etc. in the following idea:
        
        {idea}
        """,
    }

    @staticmethod
    def aggregate(results) -> str:
        sep = "–––"
        joined_results = sep.join(results)

        aggregation_prompt = """
        The following are the results of the critics:

        {results}

        Represent clearly the given criticism as bullet points, referring text for each.
        """

        p = aggregation_prompt.format(results=joined_results)
        r = request(
            RequestParams(
                prompt=p,
                n=1,
                max_tokens=500,
                temperature=0.5,
            )
        )
        return r[0]

    def run(self, params: Params) -> PipelineResults:
        results_from_critics = []

        for critic in self.critics:
            prompt = self.critics[critic].format(idea=params.prompt)

            r = request(
                RequestParams(
                    prompt=prompt,
                    n=self.default_n,
                    max_tokens=self.default_max_tokens,
                    temperature=self.default_temperature,
                )
            )
            results_from_critics.extend(r)

        log.debug(f"{self.name}: results from critics: {results_from_critics}")
        aggregated_results = self.aggregate(results_from_critics)
        log.debug(f"{self.name}: aggregated results: {aggregated_results}")
        return PipelineResults(
            text=aggregated_results, intermediate_text=results_from_critics
        )


class Praise(Pipeline):
    """
    Pipeline that comes up with various aspects to praise of a prompt, (expansion),
    then aggregates them into a short summary.
    """

    name = "praise"
    default_n = 4
    default_max_tokens = 100
    default_temperature = 0.9

    praises = {
        "simple": """
        Provide praise for the following:

        {data}
        """,
        "list": """
        List aspects of this to be praised:

        {data}
        """,
    }

    def run(self, params: Params):
        results = {}
        for k in self.praises:
            prompt = self.praises[k].format(data=params.prompt)

            r = request(
                RequestParams(
                    prompt=prompt,
                    n=self.default_n,
                    max_tokens=self.default_max_tokens,
                    temperature=self.default_temperature,
                )
            )

            results[k] = r

        aggregated_results = self.aggregate_results(results)
        return PipelineResults(text=aggregated_results)

    @staticmethod
    def aggregate_results(rd: dict[str, str]) -> str:
        sep = "\n–––\n"
        agg = ""
        for k in rd:
            for v in rd[k]:
                agg += v.strip() + sep

        aggregation_prompt = """
        The following are the many praises:

        {praises}

        Represent clearly the given praises as bullet points.
        """

        p = aggregation_prompt.format(praises=agg)
        result = request(
            RequestParams(
                prompt=p,
                n=1,
                max_tokens=500,
                temperature=0.5,
            )
        )
        return result[0]


def flatten_and_join(v: list[list[str]]) -> str:
    flattened = [item for sublist in v for item in sublist]
    s = "\n–––\n".join(flattened)
    return s


class Improver(Pipeline):
    """
    Identify useful improvements, then rewrite the initial prompt,
    integrating the suggested improvements.
    """

    name = "improver"

    pass


def run_pipeline_set(
    pipelines: list[Pipeline], params: Params
) -> dict[str, PipelineResults]:
    # TBD async
    results = {}
    for p in pipelines:
        results[p.name] = p.run(params)
    return results


def update_prompt():
    params = Params(
        prompt=st.session_state.prompt,
    )
    log.debug(f"params: {params}")
    pipelines = [all_pipelines[p] for p in st.session_state.pipelines]
    results = run_pipeline_set(pipelines, params)

    for pname in results:
        container_bottom.header(pname)
        if st.session_state.show_intermediate_outputs:
            for r in results[pname].intermediate_text:
                container_bottom.markdown("–––")
                container_bottom.markdown(r)
            container_bottom.markdown("–––")
        container_bottom.markdown(results[pname].text)


if __name__ == "__main__":
    params = get_params_from_env()
    if params["apikey"] is None:
        st.error("Please set OPENAI_API_KEY environment variable.")
    openai.api_key = params["apikey"]

    all_pipelines = {p.name: p for p in [Critic(), Praise(), Improver(), Dummy()]}

    container_top = st.container()
    container_bottom = st.container()

    with container_top:
        st.header("Deliberation system")
        st.text_area("Prompt", key="prompt")
        st.multiselect(
            "Select pipelines",
            [p for p in all_pipelines],
            key="pipelines",
            default=["critic"],
        )
        if st.button("Submit"):
            update_prompt()

        with st.expander("Advanced"):
            st.checkbox("Show intermediate outputs", key="show_intermediate_outputs")
