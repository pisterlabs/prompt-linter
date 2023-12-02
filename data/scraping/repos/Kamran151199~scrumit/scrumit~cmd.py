import openai
import typer
from promptify import OpenAI, Prompter

from scrumit.config import settings
from scrumit.entity.scrumit import Input
from scrumit.paraphraser.backends import ParaphraserOpenAI
from scrumit.recognizer.backends import RecognizerOpenAI
from scrumit.scrumer import Scrumer

app = typer.Typer()


@app.command()
def main(
    source: str = typer.Option(
        ...,
        "--source",
        "-s",
        help="Path to the source file where conversation transcripts are located",
        exists=True,
        file_okay=True,
    ),
    output: str = typer.Option(None, "--output", "-o", help="Path to the output file where the results will be saved"),
    domain: str = typer.Option(..., "--domain", "-d", help="Domain name"),
):
    """
    CLI application for processing files.
    """

    if not settings.openai_api_key:
        settings.openai_api_key = typer.prompt("OpenAI API key")

    openai.api_key = settings.openai_api_key
    model = OpenAI(settings.openai_api_key)

    prompter = Prompter(model)
    recognizer = RecognizerOpenAI(model, prompter)

    client = openai.Completion
    paraphraser = ParaphraserOpenAI(client)

    scrumer = Scrumer(recognizer, paraphraser)

    with open(source) as file:
        content = file.read()

    conversation = Input(
        text=content,
        domain=domain,
    )

    outputs = scrumer.convert(conversation)
    if output:
        with open(output, "w") as file:
            for index, story in enumerate(outputs.stories, start=1):
                file.write(f"{index}) {story.story}\n")
    else:
        typer.echo(outputs.dict())


app()
