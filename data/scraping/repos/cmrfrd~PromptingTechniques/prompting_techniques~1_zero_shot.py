import openai
import typer

from prompting_techniques import AsyncTyper, format_prompt

client = openai.AsyncOpenAI()
app = AsyncTyper()


@app.command()
async def cowboy_translator():
    """Translate text into cowboy speak."""
    text: str = str(typer.prompt("What do you want to translate?", type=str))
    assert len(text) > 0, "Please provide some text to translate."

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": format_prompt(f"""
                You are a friendly AI cowboy. You have one goal: to translate text into cowboy speak.
                
                Here is the input text: {text}
                
                What is the cowboy translation? Please output just the translation and nothing else.
                """)
            }
        ],
        max_tokens=64,
        temperature=0.9,
        model="gpt-4",
        stream=True,   
    )
    typer.echo("Translation: ", nl=False)
    async for message in response:
        assert len(message.choices) > 0, "No translation was provided."
        typer.echo(message.choices[0].delta.content, nl=False)

if __name__ == "__main__":
    app()