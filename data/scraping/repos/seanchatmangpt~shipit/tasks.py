import typer
import openai

# Create Typer app instance
app = typer.Typer()


# Define Typer command to create daily tasks
@app.command()
def create_daily_tasks(
    project_schedule: str, start_date: str, end_date: str, num_days: int
):
    # Convert start and end dates to datetime objects
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate number of days between start and end date
    delta = end_date - start_date

    # Set up OpenAI API key
    openai.api_key = "your_api_key"

    # Generate daily tasks for each day between start and end date
    for i in range(delta.days + 1):
        # Get current date
        current_date = start_date + datetime.timedelta(days=i)

        # Generate daily task using OpenAI
        response = openai.Completion.create(
            engine="davinci",
            prompt=project_schedule + f" for {current_date.strftime('%Y-%m-%d')}.",
            max_tokens=50,
            stop=["\n"],
        )

        # Print generated daily task
        print(response.choices[0].text)

    # Print confirmation message
    print("Daily tasks successfully created!")


# Run Typer app
if __name__ == "__main__":
    app()
