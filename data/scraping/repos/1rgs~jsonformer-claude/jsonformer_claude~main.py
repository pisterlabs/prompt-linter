import re
import anthropic
from typing import List, Union, Dict, Any
from jsonformer_claude.fields.base import FieldResponse
from jsonformer_claude.fields.bool import BoolField
from jsonformer_claude.fields.integer import IntField
from jsonformer_claude.fields.string import StrField
from termcolor import cprint
import json

FIELDS = {
    "number": IntField,
    "boolean": BoolField,
    "string": StrField
}


class JsonformerClaude:
    value: Dict[str, Any] = {}
    last_anthropic_response: str | None = None
    last_anthropic_response_finished: bool = False
    last_anthropic_stream = None
    llm_request_count = 0

    def __init__(
        self,
        anthropic_client: anthropic.Client,
        json_schema: Dict[str, Any],
        prompt: str,
        debug: bool = False,
        **claude_args,
    ):
        self.json_schema = json_schema
        self.prompt = prompt
        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.anthropic_client = anthropic_client
        self.claude_args = claude_args

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        if self.debug_on:
            if is_prompt:
                # cprint(caller, "green", end=" ")
                # cprint(value, "yellow")
                pass
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")

    async def _completion(self, prompt: str):
        self.debug("[completion] hitting anthropic", prompt)
        self.last_anthropic_response_finished = False
        stream = await self.anthropic_client.acompletion_stream(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            **self.claude_args,
        )
        self.llm_request_count += 1
        async for response in stream:
            self.last_anthropic_response = prompt + response["completion"]
            assistant_index = self.last_anthropic_response.find(anthropic.AI_PROMPT)
            if assistant_index > -1:
                self.last_anthropic_response = self.strip_json_spaces(
                    self.last_anthropic_response[
                        assistant_index + len(anthropic.AI_PROMPT) :
                    ]
                )
            yield self.last_anthropic_response
        self.last_anthropic_response_finished = True

    def completion(self, prompt: str):
        self.last_anthropic_stream = self._completion(prompt)
        return self.last_anthropic_stream

    async def prefix_matches(self, progress) -> bool:
        if self.last_anthropic_response is None:
            return False
        response = self.last_anthropic_response
        assert (
            len(progress) < len(response) or not self.last_anthropic_response_finished
        )
        while len(progress) >= len(response):
            await self.last_anthropic_stream.__anext__()
            response = self.last_anthropic_response

        result = response.startswith(progress)

        if not result:
            self.debug(
                "[prefix_matches]",
                f"Claude made a mistake",
            )

            cprint(progress, "red")
            cprint(response, "magenta")

        self.debug("[prefix_matches]", result)
        return result

    async def generate_object(
        self, properties: Dict[str, Any], obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            obj[key] = await self.generate_value(schema, obj, key)
        return obj

    def validate_ref(self, ref):
        if not ref.startswith('#/'):
            raise ValueError("Ref must start with #/")

    def get_definition_by_ref(self, ref) -> dict:
        self.validate_ref(ref)

        locations = ref.split('/')[1:]
        definition = self.json_schema
        for location in locations:
            definition = definition.get(location)

            if not definition:
                raise ValueError("Improper reference")

        return definition

    async def get_stream(self):
        progress = self.get_progress()
        prompt = self.get_prompt()

        self.debug("[debug-progress]", progress)
        self.debug("[debug-progress]", prompt)

        stream = self.last_anthropic_response

        if not await self.prefix_matches(progress) or stream is None:

            stream = self.completion(prompt)
        else:
            stream = self.last_anthropic_stream

        return stream

    async def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
        retries: int = 0
    ) -> Any:
        if retries > 5:
            self.debug("[completion] EXCEEDED RETRIES RETURNING NONE", str(retries))
            return None

        schema_type = schema.get("type")

        if schema_type in FIELDS:
            field = FIELDS[schema_type](
                schema=schema,
                obj=obj,
                key=key,
                generation_marker=self.generation_marker
            )
            field.insert_generation_marker()

            stream = await self.get_stream()

            async for completion in stream:
                progress = self.get_progress()
                completion = completion[len(progress):]
                self.debug("[completion]", completion)
                field_return = field.generate_value(completion)
                self.debug("[completion]", field_return)

                if field_return.value_valid:
                    return field_return.value
                elif field_return.value_found:
                    self.debug("[completion]", "retrying")
                    self.completion(self.get_prompt())
                    # Could do things like change temperature here
                    return await self.generate_value(
                        schema=schema,
                        obj=obj,
                        key=key,
                        retries=retries + 1
                    )

        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return await self.generate_array(schema["items"], new_array)

        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)

            return await self.generate_object(schema["properties"], new_obj)

        elif discriminator := schema.get("discriminator"):
            property_name = discriminator["propertyName"]
            mapping = discriminator["mapping"]

            property_name_schema = {
                "type": "string",
                "enum": [m for m in mapping]
            }

            new_obj = {}
            new_obj[property_name] = self.generation_marker

            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)

            property_enum_value = await self.generate_value(
                schema=property_name_schema,
                obj=new_obj,
                key=property_name
            )
            new_obj[property_name] = property_enum_value

            #new_obj.pop(property_name)

            self.debug("[discriminator]", property_enum_value)

            schema = self.get_definition_by_ref(mapping[property_enum_value])
            self.debug("[discriminator]", schema)
            return await self.generate_object(properties=schema["properties"], obj=new_obj)


        elif ref := schema.get("$ref"):
            definition = self.get_definition_by_ref(ref)
            return await self.generate_value(
                schema=definition,
                obj=obj,
                key=key,
            )

        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    async def generate_array(
        self, item_schema: Dict[str, Any], arr: List[Any]
    ) -> List[Any]:
        while True:
            if self.last_anthropic_response is None:
                # todo: below is untested since we do not support top level arrays yet
                stream = self.completion(self.get_prompt())
                async for response in stream:
                    completion = response[len(self.get_progress()) :]
                    if completion and completion[0] == ",":
                        self.last_anthropic_response = completion[1:]
                        break
            else:
                arr.append(self.generation_marker)
                progress = self.get_progress()
                arr.pop()
                progress = progress.rstrip(",")
                response = self.last_anthropic_response
                while len(progress) >= len(response):
                    await self.last_anthropic_stream.__anext__()
                    response = self.last_anthropic_response
                next_char = response[len(progress)]
                if next_char == "]":
                    return arr

            value = await self.generate_value(item_schema, arr)
            arr[-1] = value

    def strip_json_spaces(self, json_string: str) -> str:
        should_remove_spaces = True

        def is_unescaped_quote(index):
            return json_string[index] == '"' and (
                index < 1 or json_string[index - 1] != "\\"
            )

        index = 0
        while index < len(json_string):
            if is_unescaped_quote(index):
                should_remove_spaces = not should_remove_spaces
            elif json_string[index] in [" ", "\t", "\n"] and should_remove_spaces:
                json_string = json_string[:index] + json_string[index + 1 :]
                continue
            index += 1
        return json_string

    def get_progress(self):
        progress = json.dumps(self.value, separators=(",", ":"))
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")
        return self.strip_json_spaces(progress)

    def get_prompt(self):
        template = """{HUMAN}{prompt}\nOutput result in the following JSON schema format:\n{schema}{AI}{progress}"""
        progress = self.get_progress()
        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=progress,
            HUMAN=anthropic.HUMAN_PROMPT,
            AI=anthropic.AI_PROMPT,
        )
        return prompt.rstrip()

    async def __call__(self) -> Dict[str, Any]:
        self.llm_request_count = 0
        self.value = {}
        generated_data = await self.generate_object(
            self.json_schema["properties"], self.value
        )
        return generated_data
