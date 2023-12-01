from openai import OpenAI
import json
import time
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()


class Beta(BaseModel):
    Strike: List[float] = Field(..., description="List of strike prices or levels for the product, corresponding to each underlying asset. Provide up to five values. Example: [120.5, 130.0]. Specific price levels set on the Strike Date for different underlying assets. This is not the number associated with the knock-out/in barrier. It is rather associated with the underlying indices.")
    Finalvalday: str = Field(None, description="The final valuation day, distinct from the maturity date, formatted as 'dd/mm/yyyy'. This is the date for the final assessment of the product's value before maturity. Example: '31/12/2022'.")
    Cap: Optional[int] = Field(None, description="Optional. The upper limit or cap of the product's return, expressed as a percentage. Example: 130. Leave blank if not applicable.")


def format_response_to_json(response_string : str, gpt4 : bool = True) -> dict:
    """
    Takes a string and formats it into a JSON object. This is used to parse the output of the previous model.
    """
    client = OpenAI()
    print("Formatting response to JSON")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106" if not gpt4 else "gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant specialized in financial data analysis and extraction. Your task is to meticulously process a structured product schema and accurately populate a form with relevant data extracted from a provided document. It is your job to to extract a solid JSON from the provided message. If any values are speculative or uncertain, you should not include them in the JSON. If anything is yet to be extracted, ignore it."
            },
            {
                "role": "user",
                "content": "This is the message you need to extract a JSON from: " + response_string
            },
            {
                "role": "user",
                "content": "The following are fields that need to be extracted from the document: " + Beta.schema_json(indent=2)
            },
            {
                "role": "user",
                "content": "Think carefully and think step by step. Take a deep breath and give me an accurate JSON. DO NOT create any new fields. If you are not sure about a value, leave it blank."
            }
        ],
        response_format={'type': "json_object"}
    )
    data = completion.choices[0].message.content
    print(data)
    parsed = json.loads(data)
    return parsed


# Upload a file with an "assistants" purpose
def get(path, qs):
    file = client.files.create(
        file=open(path, "rb"),
        purpose='assistants'
    )

    print("Uploaded")

    assistant = {
        'id': "asst_qWEY6O2sL5VnotNn5nikz4fC"
    }

    # make this asynchronous

    # Assuming 'client', 'qs', 'assistant', and 'file' are already defined

    # Create all threads
    threads = [client.beta.threads.create() for _ in range(len(qs))]
    messages = [client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=q,
        file_ids=[file.id]
    ) for thread,q in zip(threads, qs)]
    runs = {
        thread.id: 0 for thread in threads
    }
    from concurrent.futures import ThreadPoolExecutor
    # Function to queue a question

    def queue_question(m, thread):
        print("Queueing question")
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant['id']
        )
        runs[thread.id] = run

    # Queue all questions
    with ThreadPoolExecutor() as executor:
        executor.map(queue_question, messages, threads)
        while any(r == 0 for r in runs.values()):
            print(".", end="")
            time.sleep(0.5)
        print("Runs created")


    def get_status(thread):
        run = runs[thread.id]
        update = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        return update.status

    # Function to get response
    def get_response(thread):
        run = runs[thread.id]
        status = "queued"
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        return messages.data[0].content[0].text.value

    # Retrieve responses
    with ThreadPoolExecutor() as executor:
        res = executor.map(get_status, threads)
        print(res)
        res = list(res)
        while any(r != "completed" for r in res):
            time.sleep(0.5)
            res = executor.map(get_status, threads)
            res = list(res)
            print("Waiting for responses", len([r for r in res if r == "completed"]), "completed")


    res = [get_response(thread) for thread in threads]


    return res

def turn_path_to_json(path, qs):
    a=get(path,qs)
    slick="\n".join(a)
    print(format_response_to_json(slick))
    return format_response_to_json(slick)
