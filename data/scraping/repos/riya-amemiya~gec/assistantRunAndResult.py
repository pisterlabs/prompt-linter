from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI()
thread = client.beta.threads.create()
try:
    while True:
        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=input(">>> ")
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id="asst_dE36bXQLqQOXy6UU9ceGcZLn",
        )
        status = run.status
        while status != "completed":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            status = run.status

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        print(messages.data[0].content[0].text.value)


except KeyboardInterrupt:
    print("Exiting...")
    exit()
