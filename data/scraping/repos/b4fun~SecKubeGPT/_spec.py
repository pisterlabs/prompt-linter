import dataclasses as dc
import typing as t
import guidance
import json

from ._types import (
    SecurityCheckProgram,
    CheckPayload,
    SpecResult,
    return_error_spec_on_failure,
)


@dc.dataclass
class SecurityCheckProgramMetadata:
    id: str
    name: str
    help: str


@dc.dataclass
class SecurityCheckProgramPrompt:
    template: str
    input_variable_name: str
    static_variables: t.Mapping[str, t.Any] = dc.field(default_factory=dict)


@dc.dataclass
class SecurityCheckProgramResultSchema:
    response_variable_name: str
    # TODO: use enum type
    response_format: str = dc.field(default="json")
    # TODO: support openapi schema?
    property_names: t.List[str] = dc.field(default_factory=list)
    # TODO: template support
    succeed_message: str = dc.field(default="😊 no security issue detected!")


@dc.dataclass
class SecurityCheckProgramSpec:
    meta: SecurityCheckProgramMetadata
    prompt: SecurityCheckProgramPrompt
    result: SecurityCheckProgramResultSchema

    @classmethod
    def from_dict(cls, spec_dict: t.Dict[str, t.Any]) -> "SecurityCheckProgramSpec":
        import dacite

        return dacite.from_dict(
            data_class=cls,
            data=spec_dict,
        )


class SpecProgram(SecurityCheckProgram):
    @classmethod
    def from_yaml_spec(self, spec_source: str) -> "SpecProgram":
        import yaml

        print(yaml.safe_load(spec_source))
        spec = SecurityCheckProgramSpec.from_dict(yaml.safe_load(spec_source))
        return SpecProgram(spec)

    def __init__(self, spec: SecurityCheckProgramSpec):
        self._spec = spec

    @property
    def id(self) -> str:
        return self._spec.meta.id

    @property
    def name(self) -> str:
        return self._spec.meta.name

    @property
    def help(self) -> str:
        return self._spec.meta.help

    def create_program(self, llm: guidance.llms.LLM) -> guidance.Program:
        return guidance.Program(
            self._spec.prompt.template,
            llm=llm,
        )

    def create_succeed_result(
        self, payload: CheckPayload, response_content: str
    ) -> SpecResult:
        return self.succeed(response_content, self._spec.result.succeed_message)

    def parse_program_result(
        self, payload: CheckPayload, program_result: guidance.Program
    ) -> SpecResult:
        assert (
            self._spec.result.response_variable_name in program_result
        ), f"Expected response variable name {self._spec.result.response_variable_name} in program result, but got {program_result}"
        response_content = program_result[self._spec.result.response_variable_name]

        assert (
            self._spec.result.response_format == "json"
        ), "Only JSON response format is supported"

        data_dict = json.loads(response_content)
        if len(data_dict) < 1:
            return self.create_succeed_result(payload, response_content)

        table_rows = [
            # header
            "| " + " | ".join(self._spec.result.property_names) + " |",
            # separator
            "| " + " | ".join(["---"] * len(self._spec.result.property_names)) + " |",
        ]
        for item in data_dict:
            col_values = [
                str(item.get(property_name, ""))
                for property_name in self._spec.result.property_names
            ]
            table_rows.append("| " + " | ".join(col_values) + " |")

        return self.failed(response_content, "\n".join(table_rows))

    @return_error_spec_on_failure
    def check(self, payload: CheckPayload) -> SpecResult:
        program = self.create_program(self.create_llm(payload))

        variables = {**self._spec.prompt.static_variables}
        variables[self._spec.prompt.input_variable_name] = payload.spec
        program_result = program(**variables)

        return self.parse_program_result(payload, program_result)
