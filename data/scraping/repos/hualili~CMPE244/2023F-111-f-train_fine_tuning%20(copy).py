import openai, json, os, time, logging

workdir = os.path.abspath(os.path.dirname(__file__))
print(workdir)
with open(workdir + "/API_Key.json", "r") as f: # Please put your own API key into the API_Key.json first.
    keyDict = json.load(f)

MODEL = "gpt-3.5-turbo"
API_KEY = keyDict.get("personalTestKey")
openai.api_key = API_KEY


def configure_logging():
    logging.basicConfig(filename=workdir + 'output.log', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s')
    return logging.getLogger()


def upload_file(file_name):
    # Note: For a 400KB train_file, it takes about 1 minute to upload.
    file_upload = openai.File.create(file=open(file_name, "rb"), purpose="fine-tune")
    logger.info(f"Uploaded file with id: {file_upload.id}")

    while True:
        logger.info("Waiting for file to process...")
        file_handle = openai.File.retrieve(id=file_upload.id)

        if len(file_handle) and file_handle.status == "processed":
            logger.info("File processed")
            break
        time.sleep(60)

    return file_upload


if __name__ == '__main__':
    # Configure logger
    logger = configure_logging()

    file_name = workdir + "/train-cti-data.jsonl"
    uploaded_file = upload_file(file_name)

    logger.info(uploaded_file)
    job = openai.FineTuningJob.create(training_file=uploaded_file.id, model=MODEL)
    logger.info(f"Job created with id: {job.id}")

    # Note: If you forget the job id, you can use the following code to list all the models fine-tuned.
    # result = openai.FineTuningJob.list(limit=10)
    # print(result)