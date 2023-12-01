import enum

import openai
import typer
from instructor.patch import wrap_chatcompletion
from pydantic import BaseModel

from prompting_techniques import AsyncTyper, format_prompt

client = openai.AsyncOpenAI()
app = AsyncTyper()

class SupportRequestLabel(str, enum.Enum):
    """The types of support requests."""
    hardware = "hardware"
    software = "software"
    network = "network"
    security = "security"
    access = "access"
    training = "training"
    other = "other"

class SupportRequestLabels(BaseModel):
    labels: list[SupportRequestLabel]

async def get_support_request_labels(text: str) -> SupportRequestLabels:    
    func = wrap_chatcompletion(client.chat.completions.create)
    result: SupportRequestLabels = await func(
        messages=[
            {
                "role": "user",
                "content": format_prompt(f"""
                You are an AI support request labeler. You have one goal: to classify a given support request into one or more labels.
                
                Classify the following support request: {text}
                """),
            },
        ],
        model="gpt-4",
        response_model=SupportRequestLabels,
        temperature=0,
        seed=256,
    )
    return SupportRequestLabels.model_validate(result)
    

@app.command()
async def label():
    """From a given message of text, classify it into one or more support request labels."""
    text: str = str(typer.prompt("Enter a support request", type=str))
    assert len(text) > 0, "Please provide some text."

    typer.echo("Labels:")
    labels = await get_support_request_labels(text)
    for label in labels.labels:
        typer.echo(f"  - {label.value}")
        
@app.command()
async def example():
    """Run an example support request label task."""
    
    text = "Help me, I think something is broken! I can't access my email."
    assert len(text) > 0, "Please provide some text."
    typer.echo(f"Support Request: {text}")

    typer.echo("Labels:")
    labels = await get_support_request_labels(text)
    for label in labels.labels:
        typer.echo(f"  - {label.value}")

if __name__ == "__main__":
    app()