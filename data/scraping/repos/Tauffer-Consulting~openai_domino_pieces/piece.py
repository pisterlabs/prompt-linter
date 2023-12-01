from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SecretsModel
from openai import OpenAI


class TextGeneratorPiece(BasePiece):

    def piece_function(self, input_data: InputModel, secrets_data: SecretsModel):
        # OpenAI settings
        if secrets_data.OPENAI_API_KEY is None:
            raise Exception("OPENAI_API_KEY not found in ENV vars. Please add it to the secrets section of the Piece.")
        client = OpenAI(api_key=secrets_data.OPENAI_API_KEY)

        # Input arguments
        openai_model = input_data.openai_model
        completion_max_tokens = input_data.completion_max_tokens
        temperature = input_data.temperature
        template = input_data.template
        dict_args = {}
        for arg in input_data.prompt_args:
            dict_args[arg.arg_name] = str(arg.arg_value)
        prompt = template.format(**dict_args)

        # Generate text based on prompt
        self.logger.info("Running OpenAI completion request...")
        try:
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=completion_max_tokens,
            )
            string_generated_text = response.choices[0].message.content

        except Exception as e:
            self.logger.info(f"\nCompletion task failed: {e}")
            raise Exception(f"Completion task failed: {e}")

        # Display result in the Domino GUI
        self.format_display_result(input_data, string_generated_text, prompt)

        # Format output
        self.logger.info("Text generated!")
        if input_data.output_type == "string":
            self.logger.info("Returning generated text as a string")
            return OutputModel(
                string_generated_text=string_generated_text,
            )

        output_file_path = f"{self.results_path}/generated_text.txt"
        with open(output_file_path, "w") as f:
            f.write(string_generated_text)

        if input_data.output_type == "file":
            self.logger.info(f"Generated text saved at: {output_file_path}")
            return OutputModel(
                file_path_generated_text=output_file_path
            )

        self.logger.info(f"Returning generated text as string and file in {output_file_path}")
        return OutputModel(
            string_generated_text=string_generated_text,
            file_path_generated_text=output_file_path
        )

    def format_display_result(self, input_data: InputModel, string_generated_text: str, prompt: str):
        md_text = f"""
## Generated text
{string_generated_text}

## Args
**prompt**: {prompt}
**model**: {input_data.openai_model}
**temperature**: {input_data.temperature}
**max_tokens**: {input_data.completion_max_tokens}

"""
        file_path = f"{self.results_path}/display_result.md"
        with open(file_path, "w") as f:
            f.write(md_text)
        self.display_result = {
            "file_type": "md",
            "file_path": file_path
        }
