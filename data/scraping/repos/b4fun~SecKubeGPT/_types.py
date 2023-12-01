import dataclasses as dc
import typing as t
import traceback
import functools
import guidance


@dc.dataclass
class SpecResult:
    """SpecResult represents the result of a security check program check."""

    program_name: str
    has_issues: bool
    raw_response: str
    formatted_response: str


@dc.dataclass
class CheckPayload:
    """CheckPayload represents the security check payload."""

    openapi_key: str
    model: str
    spec: str


class SecurityCheckProgram(t.Protocol):
    """SecurityCheckProgram validates a Kubernetes spec for security issues."""

    @property
    def id(self) -> str:
        """The ID of the program. Should be globally unique."""
        ...

    @property
    def name(self) -> str:
        """The name of the program."""
        ...

    @property
    def help(self) -> str:
        """Return the help message for the program."""
        ...

    def check(self, payload: CheckPayload) -> SpecResult:
        """Run the security check program on the given spec."""
        ...

    def succeed(self, raw_response: str, formatted_response: str) -> SpecResult:
        return SpecResult(
            program_name=self.name,
            has_issues=False,
            raw_response=raw_response,
            formatted_response=formatted_response,
        )

    def failed(self, raw_response: str, formatted_response: str) -> SpecResult:
        return SpecResult(
            program_name=self.name,
            has_issues=True,
            raw_response=raw_response,
            formatted_response=formatted_response,
        )

    def create_llm(self, payload: CheckPayload) -> guidance.llms.LLM:
        """Create the LLM instance for the program."""
        # TODO: support more LLMs.
        return guidance.llms.OpenAI(model=payload.model, api_key=payload.openapi_key)

    def __str__(self) -> str:
        return f"SecurityCheckProgram(name={self.name})"

    def __repr__(self) -> str:
        return f"SecurityCheckProgram(name={self.name})"


def return_error_spec_on_failure(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            full_stack_trace = traceback.format_exc()

            return SpecResult(
                program_name=args[0].name,
                has_issues=True,
                raw_response=full_stack_trace,
                formatted_response=f"😱 {e}",
            )

    return wrapper
