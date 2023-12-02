import time
import os
import io
import json
import pandas as pd
from .._dataset import current
from .._commandarg import parse
from .._print import _print


def _convert_tabular_format_to_openai_format(df) -> list:
    # Convert DataFrame to dictionary with 'records' orientation
    df_dict = df.to_dict(orient="records")

    # Use map to transform each record into the desired format
    records = list(
        map(
            lambda record: {
                "messages": [
                    {"role": role, "content": content}
                    for role, content in record.items()
                ]
            },
            df_dict,
        )
    )

    return records


def _write_records_to_stringio(records: list) -> io.StringIO:
    output = io.StringIO()
    for record in records:
        output.write(json.dumps(record))
        output.write("\n")
    output.seek(0)
    return output


class FineTuningJobHelper:
    def __init__(self, openai_ftjob_id: str) -> None:
        self.openai_ftjob_id = openai_ftjob_id

    def retrieve(self):
        import openai

        return openai.FineTuningJob.retrieve(self.openai_ftjob_id)

    def cancel(self):
        import openai

        return openai.FineTuningJob.cancel(self.openai_ftjob_id)

    def list_events(self, limit=10):
        import openai

        return openai.FineTuningJob.list_events(  # type: ignore
            id=self.openai_ftjob_id, limit=limit
        )

    def delete_model(self):
        import openai

        model_name = self.retrieve().to_dict()["fine_tuned_model"]
        return openai.Model.delete(model_name)

    def list_jobs(self):
        import openai

        jobs = pd.DataFrame(openai.FineTuningJob.list()["data"])  # type: ignore
        jobs["created_at"] = pd.to_datetime(jobs["created_at"], unit="s")
        jobs["finished_at"] = pd.to_datetime(jobs["finished_at"], unit="s")
        jobs = jobs.sort_values("created_at", ascending=False)
        return jobs

    def list_models(self):
        import openai

        models = pd.DataFrame(openai.Model.list()["data"])  # type: ignore
        models = models[models["id"].str.contains("personal")]
        models.reset_index(inplace=True, drop=True)
        models["created"] = pd.to_datetime(models["created"], unit="s")
        return models


def ftgpt(
    commandarg: str, model_name: str = "gpt-3.5-turbo", nonblocking=True
) -> FineTuningJobHelper:
    """
    1. creates a training file for OpenAI Fine-tuning API
    2. Upload a training file; https://platform.openai.com/docs/guides/fine-tuning/upload-a-training-file
    3. Create a fine-tuned model; https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model

    """
    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")

    _ = parse(commandarg, "varlist")
    assert len(_["varlist"].split()) == 3
    assistant_var = _["varlist"].split()[0]
    user_var = _["varlist"].split()[1]
    system_var = _["varlist"].split()[2]

    df = current.df.rename(
        columns={assistant_var: "assistant", user_var: "user", system_var: "system"}
    )
    df = df[["assistant", "user", "system"]]

    records = _convert_tabular_format_to_openai_format(df)
    stringio_object = _write_records_to_stringio(records)
    openai_file_object = openai.File.create(file=stringio_object, purpose="fine-tune")
    openai_file_id = openai_file_object.to_dict()["id"]  # type: ignore
    openai_ftjob_object = openai.FineTuningJob.create(
        training_file=openai_file_id, model=model_name
    )
    openai_ftjob_id = openai_ftjob_object.to_dict()["id"]  # type: ignore
    ftjob_helper = FineTuningJobHelper(openai_ftjob_id)
    current.last_openai_ftjob_id = openai_ftjob_id

    if not nonblocking:
        while ftjob_helper.retrieve().to_dict()["status"] in [
            "validating_files",
            "running",
        ]:
            time.sleep(10)
            print("Waiting for fine-tuning to complete...")

        print("Job halted (i.e., completed or queued)!")
        final_status = ftjob_helper.retrieve().to_dict()["status"]
        print(f"Final status: {final_status}")

    return ftjob_helper


def askgpt(user_content="Hello!", system_content="You are a helpful assistant."):
    import openai
    from openai.error import ServiceUnavailableError

    while True:
        try:
            ftjob_helper = FineTuningJobHelper(current.last_openai_ftjob_id)  # type: ignore
            try:
                model_name = ftjob_helper.retrieve().to_dict()["fine_tuned_model"]
                print(f"(Using fine tuned model {current.last_openai_ftjob_id}.)")
                print()
            except Exception as e:
                model_name = "gpt-3.5-turbo-0613"
                print(
                    f"(Fine tuned model not found.  Using base model {model_name} instead.)"
                )
                print()

            if not model_name:
                final_status = ftjob_helper.retrieve().to_dict()["status"]
                raise Exception(f"Model not found; Current job status: {final_status}")

            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )
            break
        except ServiceUnavailableError as e:
            pass
            time.sleep(10)

    return completion.choices[0].message["content"]
    # print(completion.choices[0].message["content"])  # type: ignore
