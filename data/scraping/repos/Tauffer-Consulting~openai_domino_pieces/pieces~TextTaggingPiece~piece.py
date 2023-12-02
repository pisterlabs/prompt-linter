from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SecretsModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain


type_mapping = {
    "string": "string",
    "integer": "integer",
    "float": "number",
    "boolean": "boolean",
    "array": "array"
}


class TextTaggingPiece(BasePiece):

    def piece_function(self, input_data: InputModel, secrets_data: SecretsModel):
        # OpenAI settings
        if secrets_data.OPENAI_API_KEY is None:
            raise Exception("OPENAI_API_KEY not found in ENV vars. Please add it to the secrets section of the Piece.")

        model = input_data.openai_model
        input_text = input_data.input_text
        tags = input_data.tags
        temperature = input_data.temperature

        schema = dict(
            properties=dict(),
            required=list()
        )
        for t in tags:
            schema["properties"][t.name] = {
                "type": type_mapping[t.type.value],
                "description": t.description
            }
            if t.enum:
                schema["properties"][t.name]["enum"] = t.enum.split(",")
        schema["required"] = [t.name for t in tags]

        # Create and run chain
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=secrets_data.OPENAI_API_KEY
        )
        chain = create_tagging_chain(
            schema=schema,
            llm=llm
        )
        res = chain.run(input_text)

        # Format display result as markdown
        self.format_display_result(input_data, res)

        return OutputModel(**res)

    def format_display_result(self, input_data: InputModel, result: dict):
        md_text = """## Tags\n"""
        for k, v in result.items():
            md_text += f"""**{k}**: {v}\n"""
        md_text += "\n"
        md_text += f"""**Input text**: {input_data.input_text}\n"""
        file_path = f"{self.results_path}/display_result.md"
        with open(file_path, "w") as f:
            f.write(md_text)
        self.display_result = {
            "file_type": "md",
            "file_path": file_path
        }
