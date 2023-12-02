import json
from dataclasses import asdict, dataclass

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

@dataclass
class Flashcard:
    input_expression: str
    input_language: str
    output_expression: str
    output_language: str
    example_usage: str

    @classmethod
    def from_dict(cls, data):
        return cls(
            input_expression=data.get("input_expression", None),
            input_language=data.get("input_language", None),
            output_expression=data.get("output_expression", None),
            output_language=data.get("output_language", None),
            example_usage=data.get("example_usage", None)
        )

@dataclass
class Flashcards:
    data: list[Flashcard]

    def as_json(self) -> dict:
        return {"flashcards": [asdict(card) for card in self.data]}

    @classmethod
    def import_from_json(cls, data):
        data = json.load(data)
        flashcard_objects = [Flashcard(**card) for card in data["flashcards"]]
        return cls(data=flashcard_objects)

    def __len__(self) -> int:
        return len(self.data)


class FlashcardGeneratorOpenAI:
    def __init__(self, api_key: str, llm_model: str = "gpt-3.5-turbo") -> None:
        self.chat = ChatOpenAI(temperature=0.0, model=llm_model, api_key=api_key)

        self.input_expression_schema = ResponseSchema(
            name="input_expression",
            type="str",
            description="Original expression entered by the user, refined to create translated_expression.",
        )
        self.input_language_schema = ResponseSchema(
            name="input_language", type="str", description="Language of the input expression."
        )
        self.output_expression_schema = ResponseSchema(
            name="output_expression",
            type="str",
            description="Translation of refined expression entered by the user.",
        )
        self.output_language_schema = ResponseSchema(
            name="output_language",
            type="str",
            description="Language of the output expression.",
        )
        self.example_usage_schema = ResponseSchema(
            name="example_usage",
            type="str",
            description="Example usage of input expression, used to give the user some example context where it could be used. Limited to one sentence.",
        )

        self.response_schemas = [
            self.input_expression_schema,
            self.input_language_schema,
            self.output_expression_schema,
            self.output_language_schema,
            self.example_usage_schema,
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.response_schemas
        )
        self.format_instructions = self.output_parser.get_format_instructions()

        self.flashcard_generator_template = """\
        For the following expression, extract the following information:

        input_expression: Original expression entered by the user, but refined to create translated_expression (for flashcard for language learning). If the expression is too long (more than 10 words), it should be shortened while keeping the sense.

        input_language: Language of the input expression

        output_expression: Refined input expression translated to {output_language} language. Provide 2 alternatives, separated with 'slash' sign (and space before & after the sign).

        example_usage: Example usage of input expression, used to give the user some example context where it could be used. Limited to one sentence.

        input_expression: {input_expression}
        input_language: {input_language}

        {format_instructions}
        """

        self.prompt = ChatPromptTemplate.from_template(
            template=self.flashcard_generator_template
        )

    def generate_flashcard(
        self, input_expression: str, input_language: str, output_language: str
    ) -> Flashcard:
        messages = self.prompt.format_messages(
            input_expression=input_expression,
            input_language=input_language,
            output_language=output_language,
            format_instructions=self.format_instructions,
        )
        response = self.chat(messages)
        flashcard_dict = self.output_parser.parse(response.content)
        return Flashcard.from_dict(flashcard_dict)

if __name__ == "__main__":
    
    from dotenv import load_dotenv, find_dotenv
    import os

    _ = load_dotenv(find_dotenv())  # Read local .env file

    generator = FlashcardGeneratorOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    input_expressions = [
        "cruel",
        "let someone off the hook",
        "it absorbed me",
        "get my thoughts in order",
        "crude",
        "pore over",
    ]
    input_language = "English"
    output_language = "Polish"

    flashcards = Flashcards()

    for input_expression in input_expressions:
        flashcard = generator.generate_flashcard(
            input_expression, input_language, output_language
        )
        print(flashcard)
        flashcards.flashcards.append(asdict(flashcard))

    flashcards.export_to_json("flashcards.json")
