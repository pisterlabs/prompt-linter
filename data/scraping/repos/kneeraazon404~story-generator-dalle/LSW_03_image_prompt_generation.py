import openai
import os
import time
import logging


# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Load API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize OpenAI client
client = openai.Client()

assistant_id = "asst_le5WmT0Meinq1g66lIVZbro4"


def combine_data(book_data, visual_description):
    """Combine book data and visual data into a series of prompts."""
    combined_data = []
    for story, visual in zip(book_data, visual_description):
        prompt = f"Story: {story}\nVisual Description: {visual}"
        combined_data.append(prompt)
    return combined_data


def generate_image_prompts(book_data, visual_data):
    responses = []

    combined_prompts = combine_data(book_data, visual_data)

    for prompt in combined_prompts:
        thread = client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=prompt
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=assistant_id
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
            if run_status.status == "completed":
                break
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread.id).data
        # Concatenate all message contents into a single string
        assistant_response = " ".join(
            msg.content.text.value if hasattr(msg.content, "text") else str(msg.content)
            for msg in messages
            if msg.role == "assistant"
        )

        if assistant_response:
            # Log the assistant's response including generated prompts
            logging.info("Assistant's response with prompts:")

            return assistant_response

        else:
            logging.warning("No response for the prompt.")

    return responses
