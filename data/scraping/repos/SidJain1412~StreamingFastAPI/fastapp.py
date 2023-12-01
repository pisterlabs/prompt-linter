from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
import openai
import os
import sys

# Getting OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not len(OPENAI_API_KEY):
    print("Please set OPENAI_API_KEY environment variable. Exiting.")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

# Parameters for OpenAI
openai_model = "gpt-3.5-turbo"
max_responses = 1
temperature = 0.7
max_tokens = 512

# Defining the FastAPI app and metadata
app = FastAPI(
    title="Streaming API",
    description="""### API specifications\n
To test out the Streaming API `campaign`, fire a sample query, then use the Curl command in your terminal to see it stream in real time\n
This doc does not support streaming outputs, but curl does.
              """,
    version=1.0,
)

# Defining error in case of 503 from OpenAI
error503 = "OpenAI server is busy, try again later"


def get_response_openai(prompt):
    try:
        prompt = prompt
        response = openai.ChatCompletion.create(
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=max_responses,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {"role": "system", "content": "You are an expert creative marketer. Create a campaign for the brand the user enters."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
    except Exception as e:
        print("Error in creating campaigns from openAI:", str(e))
        raise HTTPException(503, error503)
    try:
        for chunk in response:
            current_content = chunk["choices"][0]["delta"].get("content", "")
            yield current_content
    except Exception as e:
        print("OpenAI Response (Streaming) Error: " + str(e))
        raise HTTPException(503, error503)


@app.get(
    "/campaign/",
    tags=["APIs"],
    response_model=str,
    responses={503: {"detail": error503}},
)
def campaign(prompt: str = Query(..., max_length=20)):
    """
    Create a marketing campaign plan for the brand name entered in the prompt
    """
    return StreamingResponse(get_response_openai(prompt), media_type="text/event-stream")
