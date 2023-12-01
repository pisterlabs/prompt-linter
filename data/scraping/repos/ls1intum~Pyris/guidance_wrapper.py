import guidance
import re

from app.config import LLMModelConfig
from app.services.guidance_functions import truncate


class GuidanceWrapper:
    """A wrapper service to all guidance package's methods."""

    def __new__(cls, *_, **__):
        return super(GuidanceWrapper, cls).__new__(cls)

    def __init__(
        self, model: LLMModelConfig, handlebars="", parameters=None
    ) -> None:
        if parameters is None:
            parameters = {}

        self.model = model
        self.handlebars = handlebars
        self.parameters = parameters

    def query(self) -> dict:
        """Get response from a chosen LLM model.

        Returns:
            Text content object with LLM's response.

        Raises:
            Reraises exception from guidance package
        """

        # Perform a regex search to find the names of the variables
        # being generated in the program. This regex matches strings like:
        #    {{gen 'response' temperature=0.0 max_tokens=500}}
        #    {{#geneach 'values' num_iterations=3}}
        #    {{set 'answer' (truncate response 3)}}
        # and extracts the variable names 'response', 'values', and 'answer'
        pattern = r'{{#?(?:gen|geneach|set) +[\'"]([^\'"]+)[\'"]'
        var_names = re.findall(pattern, self.handlebars)

        template = guidance(self.handlebars)
        result = template(
            llm=self._get_llm(),
            truncate=truncate,
            **self.parameters,
        )

        if isinstance(result._exception, Exception):
            raise result._exception

        generated_vars = {
            var_name: result[var_name]
            for var_name in var_names
            if var_name in result
        }

        return generated_vars

    def is_up(self) -> bool:
        """Check if the chosen LLM model is up.

        Returns:
            True if the model is up, False otherwise.
        """

        guidance.llms.OpenAI.cache.clear()
        handlebars = """
        {{#user~}}Say 1{{~/user}}
        {{#assistant~}}
            {{gen 'response' temperature=0.0 max_tokens=1}}
        {{~/assistant}}
        """
        content = (
            GuidanceWrapper(model=self.model, handlebars=handlebars)
            .query()
            .get("response")
        )
        return content == "1"

    def _get_llm(self):
        return self.model.get_instance()
