from dlm_matrix.utility.loader import DatasetLoader
from dlm_matrix.utils import log_handler
from openai import OpenAI
import os
import time
import random


class DataTuner:
    def __init__(
        self, dataset_loader: DatasetLoader, verbose: bool = False, api_key=None
    ):
        self.dataset_loader = dataset_loader
        self.verbose = verbose
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def wait_for_file_ready(self, file_id, timeout=600, poll_interval=10):
        """
        Polls OpenAI's servers for the file's status until it's ready or until the timeout.

        Args:
            file_id (str): The ID of the uploaded file.
            timeout (int, optional): Maximum waiting time in seconds. Defaults to 600 (10 minutes).
            poll_interval (int, optional): Time in seconds between each polling request. Defaults to 10.

        Returns:
            tuple: (bool indicating if the file is ready, status message)
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                file_status = self.client.files.retrieve(file_id)
                if file_status["status"] == "processed":
                    return True, "File is ready"
                elif file_status["status"] == "failed":
                    return False, "File processing failed"
            except Exception as e:
                return False, f"An error occurred: {e}"
            time.sleep(poll_interval)
        return False, "Timeout reached"

    def wait_for_fine_tuning_completion(self, job_id, initial_poll_interval=60):
        """
        Waits for the fine-tuning to complete by polling the server at intervals, with exponential backoff and jitter.

        Args:
            job_id (str): The ID of the fine-tuning job.
            initial_poll_interval (int): Initial time in seconds between each polling request.

        Returns:
            bool: True if the job completed successfully, False otherwise.
        """
        poll_interval = initial_poll_interval
        retry = 0
        start_time = time.time()
        alert_interval = 600  # Alert every 10 minutes
        next_alert = start_time + alert_interval  # Next alert time

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Check if it's time for an alert
            if current_time >= next_alert:
                log_handler(
                    f"Job {job_id} is taking longer than expected. Elapsed time: {elapsed_time:.2f}s",
                    level="warning",
                    verbose=self.verbose,
                )
                next_alert = current_time + alert_interval  # Reset next alert time

            try:
                job_status = self.client.fine_tunes.retrieve(job_id)

                if job_status["status"] == "completed":
                    log_handler(
                        f"Job {job_id} completed.", level="info", verbose=self.verbose
                    )
                    return True
            except Exception as e:
                log_handler(
                    f"An exception occurred while checking job {job_id}: {str(e)}.",
                    level="error",
                    verbose=self.verbose,
                )

            # Exponential backoff with jitter
            poll_interval = min(
                600,
                (2**retry) * initial_poll_interval
                + random.uniform(0, 1) * 0.1 * (2**retry),
            )

            log_handler(
                f"Job {job_id} not yet completed. Retrying in {poll_interval:.2f} seconds...",
                level="info",
                verbose=self.verbose,
            )

            time.sleep(poll_interval)

            retry += 1

    def process_and_fine_tune(
        self,
        output_filename,
        model_suffix,
        system_message_text="",
        target_model="gpt-3.5-turbo-1106",
        upload_purpose="fine-tune",
        openai_api_key=None,
        conversation_dataframe=None,
        retrieve_model_name=False,
    ):
        """
        Generates training examples from a DataFrame and fine-tunes a language model.
        """
        # Step 0: Ensure output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_handler(
                f"Created directory {output_dir} for output files.",
                level="info",
                verbose=self.verbose,
            )

        # Step 1: Generate Training Examples
        if not conversation_dataframe:
            conversation_dataframe = self.dataset_loader.data

        log_handler(
            "Generating training examples...", level="info", verbose=self.verbose
        )
        self.dataset_loader.generate_training_examples(
            conversation_dataframe, output_filename, system_message_text
        )

        jsonl_output_filename_train = f"{output_filename}.jsonl"
        log_handler(
            f"Training examples saved to {jsonl_output_filename_train}",
            level="info",
            verbose=self.verbose,
        )

        jsonl_output_filename_test = f"{output_filename}_test.jsonl"
        log_handler(
            f"Test examples saved to {jsonl_output_filename_test}",
            level="info",
            verbose=self.verbose,
        )

        # Step 2: File Upload for Fine-Tuning
        self.client.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        log_handler(
            "Uploading training file to OpenAI...", level="info", verbose=self.verbose
        )

        with open(jsonl_output_filename_train, "rb") as training_file:
            uploaded_file_id_train = self.client.files.create(
                file=training_file, purpose=upload_purpose
            ).id

        with open(jsonl_output_filename_test, "rb") as test_file:
            uploaded_file_id_test = self.client.files.create(
                file=test_file, purpose=upload_purpose
            ).id

        log_handler(
            f"Waiting for file {uploaded_file_id_train} to be processed...",
            level="info",
            verbose=self.verbose,
        )
        if not self.wait_for_file_ready(uploaded_file_id_train):
            log_handler(
                f"File {uploaded_file_id_train} was not ready after waiting.",
                level="error",
                verbose=self.verbose,
            )
            raise Exception(
                f"File {uploaded_file_id_train} was not ready after waiting."
            )

        # Step 3: Fine-Tuning the Model
        log_handler(
            f"Starting the fine-tuning process on model: {target_model}",
            level="info",
            verbose=self.verbose,
        )
        fine_tuning_job = self.client.fine_tuning.jobs.create(
            training_file=uploaded_file_id_train,
            validation_file=uploaded_file_id_test,
            model=target_model,
            suffix=model_suffix,
        )

        log_handler(
            f"Fine-tuning job started with ID: {fine_tuning_job.id}",
            level="info",
            verbose=self.verbose,
        )

        if retrieve_model_name:
            # Wait for fine-tuning to complete
            log_handler(
                f"Waiting for fine-tuning job {fine_tuning_job.id} to complete...",
                level="info",
                verbose=self.verbose,
            )
            if not self.wait_for_fine_tuning_completion(fine_tuning_job.id):
                log_handler(
                    f"Fine-tuning job {fine_tuning_job.id} was not completed after waiting.",
                    level="error",
                    verbose=self.verbose,
                )
                raise Exception(
                    f"Fine-tuning job {fine_tuning_job.id} was not completed after waiting."
                )

            # Step 4: Retrieve Fine-Tuned Model Name
            model_name_pre_object = self.client.fine_tunes.retrieve(fine_tuning_job.id)
            model_name = model_name_pre_object["fine_tuned_model"]
            log_handler(
                f"The fine-tuned model name is: {model_name}",
                level="info",
                verbose=self.verbose,
            )
            return model_name

        return fine_tuning_job.id  # Return job ID if retrieve_model_name is False
