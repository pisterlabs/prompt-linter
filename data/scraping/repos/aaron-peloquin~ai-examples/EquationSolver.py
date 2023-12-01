from typing import Optional

import numexpr
import re
import operator as op

from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class EquationSolver(BaseTool):
    """Tool that adds the capability to solve equations like 2 + 2."""

    name = "EQUATIONSOLVER"
    description = (
        "Used for when you want to know the answer to an equation to get a result "
        "The Action Input should be a single string in the style of mathematics, examples: `2 + 2`, `3 * 4`, `2 - 1`, `2 / 1`, etc. "
        "Does not perform math on variables, so `x + 3` is not acceptable input"
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the EquationSolver tool."""
        print("")
        print(f"==== EquationSolver qry: `{query}`")

        # Strip all characters that do not match the regular expression.
        prepared_equation = self.cleanup_equation(query)

        solution = numexpr.evaluate(prepared_equation)
        print(f"== Solution == ({prepared_equation}) = ({solution})")
        return solution

    def cleanup_equation(
        self,
        equation: str,
    ):
        """
        Strips all non-numeric characters or mathematical symbols from a string.

        Args:
            equation: The equation string to strip.

        Returns:
            The equation string with all non-numeric characters or mathematical symbols stripped.
        """
        equation_string = equation.replace("+", " + ").replace("-", " - ").replace("*", " * ").replace("/", " / ")
        equation_tokens = equation_string.split()
        equation_parsed = " ".join(equation_tokens)

        # Create a regular expression that matches all non-numeric characters or mathematical symbols.
        regex = re.compile(r'[^0-9\-+*/.]')

        # Replace all matches with an empty string.
        stripped_equation = regex.sub('', equation_parsed)

        return stripped_equation


    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the EquationSolver tool asynchronously."""
        raise NotImplementedError("EquationSolver does not support async")
