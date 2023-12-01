"""Prompt template that contains few shot examples."""
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Extra, root_validator

from langchain.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    BasePromptTemplate,
    check_valid_template,
)
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import BasePromptTemplate, PromptTemplate


class FewShotPromptTemplate(BasePromptTemplate, BaseModel):
    """Prompt template that contains few shot examples."""

    examples: Optional[List[dict]] = None
    """Examples to format into the prompt.
    Either this or example_selector should be provided."""

    example_selector: Optional[BaseExampleSelector] = None
    """ExampleSelector to choose the examples to format into the prompt.
    Either this or examples should be provided."""

    example_prompt: BasePromptTemplate
    """PromptTemplate used to format an individual example."""

    suffix: str
    """A prompt template string to put after the examples."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    example_separator: str = "\n\n"
    """String separator used to join the prefix, the examples, and suffix."""

    prefix: str = ""
    """A prompt template string to put before the examples."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    @root_validator(pre=True)
    def check_examples_and_selector(cls, values: Dict) -> Dict:
        """Check that one and only one of examples/example_selector are provided."""
        examples = values.get("examples", None)
        example_selector = values.get("example_selector", None)
        if examples and example_selector:
            raise ValueError(
                "Only one of 'examples' and 'example_selector' should be provided"
            )

        if examples is None and example_selector is None:
            raise ValueError(
                "One of 'examples' and 'example_selector' should be provided"
            )

        return values

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix and input variables are consistent."""
        check_valid_template(
            values["prefix"] + values["suffix"],
            values["template_format"],
            values["input_variables"],
        )
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _get_examples(self, **kwargs: Any) -> List[dict]:
        if self.examples is not None:
            return self.examples
        elif self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        else:
            raise ValueError

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        # Format the examples.
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
        # Create the overall template.
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    def _prompt_dict(self) -> Dict:
        """Return a dictionary of the prompt."""
        if self.example_selector:
            raise ValueError("Saving an example selector is not currently supported")

        prompt_dict = self.dict()
        prompt_dict["_type"] = "few_shot"
        return prompt_dict

class FewShotPromptTemplate2(BasePromptTemplate, BaseModel):
    """Prompt template that contains few shot examples."""

    prefix_template: Union[BasePromptTemplate, str] = ''
    """A prompt template string to put before the examples."""

    examples: Optional[List[dict]] = None
    """Examples to format into the prompt.
    Either this or example_selector should be provided."""

    example_selector: Optional[BaseExampleSelector] = None
    """ExampleSelector to choose the examples to format into the prompt.
    Either this or examples should be provided."""

    example_template: BasePromptTemplate
    """PromptTemplate used to format an individual example."""

    suffix_template: Union[BasePromptTemplate, str] = ''
    """A prompt template string to put after the examples."""

    example_separator: str = "\n\n"
    """String separator used to join the prefix, the examples, and suffix."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    max_len: int = -1
    subtract_gen_len: bool = False
    enc_len_fn: Any = None

    lm: str = 'EleutherAI/gpt-neo-2.7B'

    @root_validator(pre=True)
    def check_examples_and_selector(cls, values: Dict) -> Dict:
        """Check that one and only one of examples/example_selector are provided."""
        examples = values.get("examples", None)
        example_selector = values.get("example_selector", None)
        if examples and example_selector:
            raise ValueError(
                "Only one of 'examples' and 'example_selector' should be provided"
            )

        if examples is None and example_selector is None:
            raise ValueError(
                "One of 'examples' and 'example_selector' should be provided"
            )
        return values

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, test input, suffix, and input variables are consistent."""
        for part in ['prefix_template', 'suffix_template']:
            if isinstance(values[part], str):
                values[part] = PromptTemplate(template=values[part], input_variables=[], template_format=values['template_format'])
        assert isinstance(values['example_template'], BasePromptTemplate)
        values['input_variables'] = set()

        for part in ['prefix_template', 'example_template', 'suffix_template']:
            values['input_variables'].update(values[part].input_variables)
        values['input_variables'] = list(values['input_variables'])
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _get_examples(self, **kwargs: Any) -> List[dict]:
        if self.examples is not None:
            return self.examples
        elif self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        else:
            raise ValueError

    def make_prompt(self, prefix, example_strings, test_example_string, suffix):
        pieces = [
            prefix,
            *example_strings,
            test_example_string,
            suffix
        ]
        # Create the overall prompt.
        return self.example_separator.join([p for p in pieces if p])

    def format_from_examples(self, examples, return_demos=False, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            examples: A list of exemplars.
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        assert set(kwargs.keys()).issuperset(self.input_variables)
        # Format the examples.
        example_strings = [
            self.example_template.format(**ex, test=False) for ex in examples
        ]
        test_example_string = self.example_template.format(**kwargs, test=True)
        # Format the template with the input variables.
        prefix = self.prefix_template.format(**{k:kwargs[k] for k in kwargs
                if k in self.prefix_template.input_variables})
        suffix = self.suffix_template.format(**{k:kwargs[k] for k in kwargs
                if k in self.suffix_template.input_variables})
        max_len = self.max_len
        if max_len != -1:
            if not self.subtract_gen_len:
                while self.enc_len_fn(self.make_prompt(
                    prefix, example_strings, test_example_string, suffix)
                ) > max_len:
                    example_strings = example_strings[1:]
            else:
                test_example_string_completed = self.example_template.format(**kwargs)
                while self.enc_len_fn(self.make_prompt(
                    prefix, example_strings, test_example_string_completed, suffix)
                ) > max_len:
                    example_strings = example_strings[1:]
            # print(f'reduced examples from {len(examples)} to {len(example_strings)}')
        prompt = self.make_prompt(prefix, example_strings, test_example_string, suffix)
        if return_demos:
            return prompt, list(examples)[-len(example_strings):]
        else:
            return prompt

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        assert set(kwargs.keys()).issuperset(self.input_variables)
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        return self.format_from_examples(examples, **kwargs)

    def _prompt_dict(self) -> Dict:
        """Return a dictionary of the prompt."""
        if self.example_selector:
            raise ValueError("Saving an example selector is not currently supported")

        prompt_dict = self.dict()
        prompt_dict["_type"] = "few_shot"
        return prompt_dict

    def parse_output(self, lm_output: str, **kwargs) -> Union[str, List[str], Dict[str, str]]:
        if hasattr(self.example_template, 'parse_output'):
            return self.example_template.parse_output(lm_output, **kwargs)
        else:
            return super().parse_output(lm_output, **kwargs)

    def check_output(self, prediction, **kwargs) -> bool:
        if hasattr(self.example_template, 'check_output'):
            return self.example_template.check_output(prediction, **kwargs)
        else:
            super().check_output(prediction, **kwargs)