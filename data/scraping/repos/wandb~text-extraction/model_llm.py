import typing

import weave

import base_types
import cli


class PredictBasicOutput(typing.TypedDict):
    name: str
    shares: int


# Model
@weave.type()
class ModelLLM(base_types.Model):
    prompt_template: base_types.PromptTemplate

    @weave.op()
    def predict(self, example: str) -> PredictBasicOutput:
        import json
        from weave.monitoring import openai

        prompt = self.prompt_template.prompt.format(example=example)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        response_message = response["choices"][0]["message"]["content"]
        result = json.loads(response_message)
        result["shares"] = int(result["shares"])
        return result


if __name__ == "__main__":
    cli.weave_object_main(ModelLLM)
