import inspect
from typing import Callable, Optional
import openai

from akashic_records._meta.compiler import compile_source


class DynamicFunction:
    def __init__(self, name, globals, source_code):
        self.name = name
        self.globals = globals
        self.source_code = source_code

    def __call__(self, *args, **kwargs):
        return self.globals[self.name](*args, **kwargs)


class CodeBuilder:
    def __init__(self, n=3, temperature=0.2, max_tokens=512):
        self.n = n
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.compiled_function = None

    # TODO: Can I do some kind of docopt style docstring parsing to get more structured data in here?
    #       lol, I could have people put JSON in the doc string I guess
    #       I could use something like numpydoc to parse a standard docstring foramt
    #         What would I actually use the parsed data for?
    def build_function(
        self,
        name: str,
        signature: inspect.Signature,
        docstring: Optional[str],
    ) -> list[Callable]:
        arg_spec = [str(param) for param in signature.parameters.values()]

        if isinstance(signature.return_annotation, str):
            ret = signature.return_annotation
        elif isinstance(signature.return_annotation, type):
            ret = signature.return_annotation.__name__
        else:
            ret = None

        def_line = f"def {name}({', '.join(arg_spec)}){' -> '+ret if ret else ''}:"
        prompt_text = def_line
        if docstring:
            prompt_text += '\n    """'
            for line in docstring.splitlines():
                prompt_text += f"\n    {line}"
            prompt_text += '\n    """'

        completions = openai.Completion.create(
            model="code-davinci-002",
            prompt=prompt_text,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            n=self.n,
        )

        # TODO: Sort choices by some measure of quality
        # finish_reason == "stop" is one criteria
        # if choice["finish_reason"] == "stop":
        #     completion_text = choice.text
        choices = completions.choices

        functions = []
        for choice in choices:
            try:
                source_code = prompt_text + choice.text
                code = compile_source(source_code)
                globals = {}
                exec(code, globals)
                functions.append(DynamicFunction(name, globals, source_code))
            except:
                # TODO: log
                pass
        return functions
