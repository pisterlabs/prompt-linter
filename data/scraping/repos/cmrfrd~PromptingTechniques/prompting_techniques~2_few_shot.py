import openai
import typer

from prompting_techniques import AsyncTyper, format_prompt

client = openai.AsyncOpenAI()
app = AsyncTyper()


@app.command()
async def job_description_labeler():
    """From a given job description, label it with the appropriate job title."""
    text: str = str(typer.prompt("Enter a job description", type=str))
    assert len(text) > 0, "Please provide some text."

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": format_prompt(f"""
                You are an AI job classification bot. You have one goal: to label job descriptions with the appropriate job title / role.
                
                Here are some examples of job descriptions and their corresponding job titles:

                Description: you will advise clients on the adoption and implementation of renewable energy solutions. Your responsibilities include conducting feasibility studies, providing cost-benefit analyses, and staying up-to-date with the latest in sustainable technology.
                Job: Renewable Energy Consultant
                
                Description: will lead efforts to connect the organization with the local community. This role involves planning and executing community events, managing social media outreach, and building relationships with community leaders to promote the organization's mission and values
                Job: Community Outreach Coordinator
                
                Description: will oversee the development and management of urban farming projects. This role involves collaborating with community groups, managing sustainable agricultural practices, and promoting local food production in urban settings
                Job: Urban Agriculture Director
                
                Description: you will organize and manage commercial space travel experiences for clients. This includes coordinating with aerospace companies, ensuring compliance with safety regulations, and providing clients with a once-in-a-lifetime journey into space.
                Job: Space Tourism Manager
                
                Here is a new job description, please label it with the appropriate job title, just output the title nothing else:
                
                Description: {text}
                """)
            }
        ],
        max_tokens=64,
        temperature=0.9,
        model="gpt-4",
        stream=True,   
    )
    typer.echo("Job: ", nl=False)
    async for message in response:
        assert len(message.choices) > 0, "No text was provided."
        typer.echo(message.choices[0].delta.content, nl=False)

if __name__ == "__main__":
    app()