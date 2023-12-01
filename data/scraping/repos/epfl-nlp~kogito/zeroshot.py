import openai

from kogito.core.model import KnowledgeModel
from kogito.core.knowledge import KnowledgeGraph
from kogito.core.utils import get_uuid


class GPT3Zeroshot(KnowledgeModel):
    """Zeroshot knowledge model based on GPT-3"""

    def __init__(self, api_key: str, model_name: str = "text-davinci-002") -> None:
        """Initialize a GPT-3 model

        Args:
            api_key (str): OpenAI API Key for GPT-3 model
            model_name (str, optional): Type of GPT-3 model. Defaults to "text-davinci-002".
        """
        self.api_key = api_key
        self.model_name = model_name

    def train(self):
        raise ValueError("GPT-3 Zeroshot model is not trainable")

    def save_pretrained(self, save_model_path: str):
        raise ValueError("GPT-3 Zeroshot cannot be saved")

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        raise ValueError("GPT-3 only supports API-based access")

    def generate(
        self,
        input_graph: KnowledgeGraph,
        num_samples: int = 10,
        include_task_prompt: bool = True,
        debug: bool = False,
        **kwargs,
    ) -> KnowledgeGraph:
        """Generate inferences from GPT-3 model

        Args:
            input_graph (KnowledgeGraph): Input dataset
            num_samples (int, optional): Number of samples to use. Defaults to 10.
            include_task_prompt (bool, optional): Whether to include task prompt. Defaults to True.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
            kwargs: Additional arguments to pass to the OpenAI.Completion API

        Returns:
            KnowledgeGraph: Completed knowledge graph
        """
        rel_kg_map = {}
        outputs = []

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 16

        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.9

        if "top_p" not in kwargs:
            kwargs["top_p"] = 0.9

        if "n" not in kwargs:
            kwargs["n"] = 1

        if "logprobs" not in kwargs:
            kwargs["logprobs"] = None

        if "stop" not in kwargs:
            kwargs["stop"] = None

        for input_kg in input_graph:
            if input_kg.relation not in rel_kg_map:
                rel_kg_map[input_kg.relation] = {"samples": [], "targets": []}
            if input_kg.tails:
                rel_kg_map[input_kg.relation]["samples"].append(input_kg)
            else:
                rel_kg_map[input_kg.relation]["targets"].append(input_kg)

        for relation, kg_map in rel_kg_map.items():
            samples = kg_map["samples"][:num_samples]
            targets = kg_map["targets"]

            if targets:
                prompts = []
                sample_prompts = []

                for index, sample_kg in enumerate(samples):
                    sample_prompt = sample_kg.to_prompt(
                        include_tail=True, index=index + 1
                    )
                    sample_prompts.append(sample_prompt)

                final_sample_prompt = "\n\n".join(sample_prompts)

                for target in targets:
                    target_prompt = target.to_prompt(index=len(samples) + 1)
                    final_prompt = f"{final_sample_prompt}\n\n{target_prompt}"

                    if include_task_prompt and relation.prompt:
                        final_prompt = f"{relation.prompt}\n\n{final_prompt}"
                    prompts.append(final_prompt)

                response = complete_gpt3(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    prompt=prompts,
                    max_tokens=kwargs["max_tokens"],
                    temperature=kwargs["temperature"],
                    top_p=kwargs["top_p"],
                    logprobs=kwargs["logprobs"],
                    n=kwargs["n"],
                    stop=kwargs["stop"],
                    debug=debug,
                )

                rel_outputs = []
                for target in targets:
                    rel_outputs.append(target.copy())

                for result in response.choices:
                    output_kg = rel_outputs[result["index"] // kwargs["n"]]
                    output_kg.tails.append(result["text"])

                outputs.extend(rel_outputs)

        return KnowledgeGraph(outputs)


def complete_gpt3(
    api_key,
    model_name,
    prompt,
    max_tokens=16,
    temperature=1,
    top_p=1,
    logprobs=None,
    n=1,
    stop=None,
    debug=False,
):
    response = None
    openai.api_key = api_key

    if debug:
        with open(f"gpt3_prompt_{get_uuid()}.txt", "w") as f:
            f.write("\n\n".join(prompt))

    try:
        response = openai.Completion.create(
            engine=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_p=top_p,
            echo=False,
            stop=stop,
            n=n,
        )
    except Exception as e:
        print("Something went wrong when querying GPT-3 API")
        raise e
    return response
