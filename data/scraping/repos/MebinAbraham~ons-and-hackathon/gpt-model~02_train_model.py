"""
02 - Train model (AD-HOC)
Uses the parsed jsonl data to train an openAPI model
This runs once, to train a model
get_fine_tune can check if it's done yet
"""

import logging
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()
OPENAPI_API_KEY = os.getenv("OPENAI_API_KEY")
FINE_TUNE_ID = os.getenv("FINE_TUNE_ID")
FORMATTED_DATA_PATH = Path("datasets/formatted_data.jsonl")

logger = logging.getLogger("Hackathon")
logger.setLevel(logging.INFO)
openai.api_key = OPENAPI_API_KEY


def file_upload(file_path: Path) -> int:
    """
    Uploads the file with training data to openAI
    Needs to be a jsonl file with the correct prompt completion format
    """
    logger.info("Uploading file to openAI")
    response = openai.File.create(file=file_path.read_bytes(), purpose="fine-tune")
    file_id = response.id
    logger.info(f"{file_id=}")
    return file_id


def create_fine_tune(file_id: int):
    """
    Starts fine-tuning the model, this can take minutes or hours to complete
    """
    logger.info(f"Fine tuning model with {file_id=}")
    openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")


def get_fine_tune_id() -> str:
    """
    OpenAI doesn't allow users to choose this, so it must be found
    """
    response = openai.FineTuningJob.list()
    print(f"{response=}")
    if response.get("data", []):
        return response["data"][0]["id"]
    raise ValueError("No fine-tunes found")


def get_fine_tune(fine_tune_id: str) -> str | None:
    """
    Retrieves the status of the model from openAI
    """
    # Retrieve the state of fine-tune
    data = openai.FineTuningJob.retrieve(fine_tune_id)
    print(data)
    if data["fine_tuned_model"] is not None:
        return data["fine_tuned_model"]
    else:
        logger.warning("Model not yet ready")
        logger.warning(f"status={data['status']}")


def main():
    """
    Uploads file which generates the file_id
    uses the file_id to create the model
    once that is done, keep using get_fine_tune_id until it is - put it in the env file
    RUN ONCE PER MODEL TRAINING
    """
    file_id = file_upload(FORMATTED_DATA_PATH)
    create_fine_tune(file_id)
    get_fine_tune_id()


def check_model_status():
    response = openai.FineTuningJob.list()
    for item in response.get("data", []):
        print(f'model id={item["id"]}, model status={item["status"]}')


if __name__ == "__main__":
    check_model_status()
    # main()
