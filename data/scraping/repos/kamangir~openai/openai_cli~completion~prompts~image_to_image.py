from typing import List
from openai_cli.completion.prompts.structured import structured_prompt


class i2i_prompt(structured_prompt):
    def __init__(
        self,
        returns: str,
        *args,
        requirements: List[str] = None,
        **kwargs,
    ):
        super().__init__(
            inputs=["an image as a numpy array"],
            requirements=[
                "does not run a for loop on the pixels",
                "uses numpy vector functions",
                "imports all modules that are used in the code",
                "type-casts the output correctly",
            ]
            + ([] if requirements is None else requirements),
            returns=[f"{returns} as a numpy array"],
            *args,
            **kwargs,
        )
