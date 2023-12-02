from openai import OpenAI
import openai
from typing import Optional
from typing import Union
import os
from typing import List, Dict
import logging
import httpx 

# Read logging level from environment variable
logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()

# Configure logging with the level from the environment variable
logging.basicConfig(
    level=getattr(logging, logging_level, logging.WARNING),  # Default to WARNING if invalid level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a logger object
logger = logging.getLogger(__name__)

class FineTuningAPI:
    """
    A class to interact with the OpenAI API, specifically for fine-tuning operations.

    This class initializes a client for the OpenAI API, handling the API key validation and configuring timeout options for the API requests. It is designed to work with fine-tuning tasks, providing an interface to interact with OpenAI's fine-tuning capabilities.

    :param openai_key: The OpenAI API key. If not provided, it defaults to the value set in the environment variable 'OPENAI_API_KEY', optional
    :type openai_key: str, optional
    :param enable_timeouts: Flag to enable custom timeout settings for API requests. If False, default timeout settings are used, defaults to False
    :type enable_timeouts: bool, optional
    :param timeouts_options: A dictionary specifying custom timeout settings. Required if 'enable_timeouts' is True. It should contain keys 'total', 'read', 'write', and 'connect' with corresponding timeout values in seconds, optional
    :type timeouts_options: dict, optional
    :raises ValueError: If no valid OpenAI API key is provided or found in the environment variable
    """
    def __init__(self, openai_key=None, enable_timeouts= False, timeouts_options= None):
        self.openai_api_key = openai_key if openai_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key.")
        self.client = OpenAI(api_key=self.openai_api_key, max_retries=3)
        if enable_timeouts:
            if timeouts_options is None:
                timeouts_options = {"total": 120, "read": 60.0, "write": 60.0, "connect": 10.0}
                self.client = self.client.with_options(timeout=httpx.Timeout(120.0, read=60.0, write=60.0, connect=10.0))
            else:
                self.client = self.client.with_options(timeout=httpx.Timeout(timeouts_options["total"], timeouts_options["read"], timeouts_options["write"], timeouts_options["connect"]))
        

    def create_fine_tune_file(self, file_path: str, purpose: Optional[str] = 'fine-tune') -> str:
        """
        Uploads a specified file to OpenAI for fine-tuning purposes and returns the file's identifier.

        This method is integral for preparing datasets for language model fine-tuning on OpenAI's platform.
        It takes a local file path, uploads the file, and returns the unique identifier of the uploaded file.
        The method is robust, encapsulating error handling for file accessibility and API interaction issues.

        :param file_path: Absolute or relative path to the JSONL file designated for fine-tuning.
        :type file_path: str
        :param purpose: Intended use of the uploaded file, influencing how OpenAI processes the file. Defaults to 'fine-tune'.
        :type purpose: str, optional
        :return: Unique identifier of the uploaded file, typically used for subsequent API interactions.
        :rtype: str
        :raises FileNotFoundError: Raised when the specified file_path does not point to an existing file.
        :raises PermissionError: Raised when access to the specified file is restricted due to insufficient permissions.
        :raises Exception: Generic exception for capturing and signaling failures during the API upload process.

        :example:

        ::

        >>> api = FineTuningAPI(api_key="sk-your-api-key")
        >>> file_id = api.create_fine_tune_file("/path/to/your/dataset.jsonl")
        >>> print(file_id)
        >>> >'file-xxxxxxxxxxxxxxxxxxxxx'
        """
        try:
            with open(file_path, "rb") as file_data:
                config = self.client.files.create(
                    file=file_data,
                    purpose=purpose
                )
            return config.id
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at path {file_path} was not found.")
        except PermissionError:
            raise PermissionError(f"Permission denied when trying to open {file_path}.")
        except Exception as e:
            raise Exception(f"An error occurred with the OpenAI API: {e}")

    
    def create_fine_tuning_job(self, 
                           training_file: str, 
                           model: str, 
                           suffix: Optional[str] = None, 
                           batch_size: Optional[Union[str, int]] = 'auto', 
                           learning_rate_multiplier: Optional[Union[str, float]] = 'auto', 
                           n_epochs: Optional[Union[str, int]] = 'auto', 
                           validation_file: Optional[str] = None) -> dict:
        """
        Start a fine-tuning job using the OpenAI Python SDK.

        This method initiates a fine-tuning job with the specified model and training file. It allows customization of additional parameters such as batch size, learning rate multiplier, number of epochs, and the validation file.

        :method create_fine_tuning_job: Initiates a fine-tuning job for a model.
        :type create_fine_tuning_job: function

        :param training_file: The file ID of the training data uploaded to OpenAI API.
        :type training_file: str

        :param model: The name of the model to fine-tune.
        :type model: str

        :param suffix: A suffix to append to the fine-tuned model's name, optional.
        :type suffix: str, optional

        :param batch_size: Number of examples in each batch, can be a specific number or 'auto', optional.
        :type batch_size: str or int, optional

        :param learning_rate_multiplier: Scaling factor for the learning rate, can be a specific number or 'auto', optional.
        :type learning_rate_multiplier: str or float, optional

        :param n_epochs: The number of epochs to train the model for, can be a specific number or 'auto', optional.
        :type n_epochs: str or int, optional

        :param validation_file: The file ID of the validation data uploaded to OpenAI API, optional.
        :type validation_file: str, optional

        :return: A dictionary containing information about the fine-tuning job, including its ID.
        :rtype: dict

        :raises ValueError: If the training_file is not provided.
        :raises Exception: If an error occurs during the creation of the fine-tuning job.
 
        :example:

        ::


            >>> api = FineTuningAPI(api_key="your-api-key")
            >>> job_info = api.create_fine_tuning_job(training_file="file-abc123", 
                                                    model="gpt-3.5-turbo",
                                                    suffix="custom-model-name",
                                                    batch_size=4,
                                                    learning_rate_multiplier=0.1,
                                                    n_epochs=2,
                                                    validation_file="file-def456")
            >>> print(job_info)
            {'id': 'ft-xyz789', ...}
        """

        if not training_file:
            raise ValueError("A training_file must be provided to start a fine-tuning job.")

        hyperparameters = {
            'batch_size': batch_size,
            'learning_rate_multiplier': learning_rate_multiplier,
            'n_epochs': n_epochs,
        }

        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file,
                model=model,
                suffix=suffix,
                hyperparameters=hyperparameters,
                validation_file=validation_file
            )
            return response
        except Exception as e:
            raise Exception(f"An error occurred while creating the fine-tuning job: {e}")




    def list_fine_tuning_jobs(self, limit: int = 10) -> List[Dict]:
        """
        List the fine-tuning jobs with an option to limit the number of jobs returned.

        This method retrieves a list of fine-tuning jobs. An optional parameter 'limit' can be set to restrict the number of jobs returned. It interacts with the OpenAI API and processes the response to provide a concise list of fine-tuning jobs.

        :method list_fine_tuning_jobs: Retrieves a list of fine-tuning jobs.
        :type list_fine_tuning_jobs: function

        :param limit: The maximum number of fine-tuning jobs to return, defaults to 10.
        :type limit: int, optional

        :return: A list of dictionaries, each representing a fine-tuning job.
        :rtype: List[Dict]

        :raises openai.error.OpenAIError: If an error occurs with the OpenAI API request.

        :example:

        ::


            >>> api = FineTuningAPI(api_key="your-api-key")
            >>> jobs = api.list_fine_tuning_jobs(limit=5)
            >>> for job in jobs:
            >>>     print(job)
        """

        try:
            response = self.client.fine_tuning.jobs.list(limit=limit)
            return response.data
        except openai.error.OpenAIError as e:
            raise openai.error.OpenAIError(f"An error occurred while listing fine-tuning jobs: {e}")

    def retrieve_fine_tuning_job(self, job_id: str) -> Dict:
        """
        Retrieve the state of a specific fine-tuning job.

        This method is used to obtain detailed information about a specific fine-tuning job, identified by its job ID. It interacts with the OpenAI API to retrieve and present the state and other relevant details of the requested fine-tuning job.

        :method retrieve_fine_tuning_job: Retrieves details of a specific fine-tuning job.
        :type retrieve_fine_tuning_job: function

        :param job_id: The ID of the fine-tuning job to retrieve.
        :type job_id: str

        :return: A dictionary containing details about the fine-tuning job.
        :rtype: Dict

        :raises ValueError: If the job_id is not provided.
        :raises openai.error.OpenAIError: If an error occurs with the OpenAI API request.

        :example:

        ::


            >>> api = FineTuningAPI(api_key="your-api-key")
            >>> job_details = api.retrieve_fine_tuning_job(job_id="ft-xyz789")
            >>> print(job_details)
        """

        if not job_id:
            raise ValueError("A job_id must be provided.")

        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            return response
        
        except openai.APIConnectionError as e:                            
                logger.error(f"[retrieve_fine_tuning_job] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[retrieve_fine_tuning_job] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[retrieve_fine_tuning_job] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[retrieve_fine_tuning_job] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"An error occurred during model evaluation: {e}")


    def cancel_fine_tuning_job(self, job_id: str) -> Dict:
        """
        Cancel a specific fine-tuning job.

        This method allows for the cancellation of a fine-tuning job identified by its job ID. It interacts with the OpenAI API to send a cancellation request and handles various potential errors that might occur during this process.

        :method cancel_fine_tuning_job: Cancels a fine-tuning job.
        :type cancel_fine_tuning_job: function

        :param job_id: The ID of the fine-tuning job to cancel.
        :type job_id: str

        :return: Confirmation of the cancellation.
        :rtype: Dict

        :raises ValueError: If the job_id is not provided.
        :raises openai.APIConnectionError: If there's a connection error with the API.
        :raises openai.RateLimitError: If the request is rate-limited by the API.
        :raises openai.APIStatusError: If there's a status error from the API.
        :raises AttributeError: If an attribute error occurs during the process.
        :raises Exception: For any other exceptions that occur during the cancellation process.

        :example:

        ::


            >>> api = FineTuningAPI(api_key="your-api-key")
            >>> cancellation_result = api.cancel_fine_tuning_job(job_id="ft-xyz789")
            >>> print(cancellation_result)
        """

        if not job_id:
            raise ValueError("A job_id must be provided.")

        try:
            response = self.client.fine_tuning.jobs.cancel(job_id)
            return response
        
        except openai.APIConnectionError as e:                            
                logger.error(f"[cancel_fine_tuning_job] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[cancel_fine_tuning_job] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[cancel_fine_tuning_job] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[cancel_fine_tuning_job] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"[cancel_fine_tuning_job] An error occurred during model evaluation: {e}")


    
    def list_fine_tune_files(self) -> List[Dict]:
        """
        List files that have been uploaded to OpenAI for fine-tuning.

        This method allows the retrieval of a list of files uploaded to the OpenAI API, primarily for the purpose of fine-tuning models. The list includes comprehensive details such as file IDs, creation dates, and the purposes of the files.

        :method list_fine_tune_files: Retrieves a list of uploaded files for fine-tuning.
        :type list_fine_tune_files: function

        :return: A list of dictionaries, each containing details of an uploaded file.
        :rtype: List[Dict]

        :raises Exception: If an error occurs during the API request.

        :example:

        ::


            >>> api = FineTuningAPI(api_key="your-api-key")
            >>> files = api.list_fine_tune_files()
            >>> for file in files:
            >>>     print(file)
        """

        try:
            response = self.client.files.list()
            return response.data
        except Exception as e:
            raise Exception(f"An error occurred while listing uploaded files: {e}")


    def list_events_fine_tuning_job(self, fine_tuning_job_id: str, limit: int = 10) -> List[Dict]:
        """
        List up to a specified number of events from a fine-tuning job.

        This method retrieves a list of events associated with a specific fine-tuning job, identified by its job ID. It allows setting a limit on the number of events to be returned and handles various potential errors that might occur during the API interaction.

        :method list_events_fine_tuning_job: Retrieves a list of events from a specified fine-tuning job.
        :type list_events_fine_tuning_job: function

        :param fine_tuning_job_id: The ID of the fine-tuning job to list events from.
        :type fine_tuning_job_id: str

        :param limit: The maximum number of events to return, defaults to 10.
        :type limit: int, optional

        :return: A list of dictionaries, each representing an event from the fine-tuning job.
        :rtype: List[Dict]

        :raises ValueError: If the fine_tuning_job_id is not provided.  
        :raises openai.APIConnectionError: If there's a connection error with the API.
        :raises openai.RateLimitError: If the request is rate-limited by the API.
        :raises openai.APIStatusError: If there's a status error from the API.
        :raises AttributeError: If an attribute error occurs during the process.
        :raises Exception: For any other exceptions that occur during the process.

        :example:

        ::


            >>> api = FineTuningAPI(api_key="your-api-key")
            >>> events = api.list_events_fine_tuning_job(fine_tuning_job_id="ft-xyz789", limit=5)
            >>> for event in events:
            >>>     print(event)
        """

        if not fine_tuning_job_id:
            raise ValueError("A fine_tuning_job_id must be provided.")

        try:
            response = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=fine_tuning_job_id, limit=limit)
            return response
        except openai.APIConnectionError as e:                            
                logger.error(f"[list_events_fine_tuning_job] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[list_events_fine_tuning_job] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[list_events_fine_tuning_job] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[list_events_fine_tuning_job] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"[list_events_fine_tuning_job] An error occurred during model evaluation: {e}")

    
    
    def delete_fine_tuned_model(self, model_id: str) -> Dict:
        """
        Delete a fine-tuned model. The caller must be the owner of the organization the model was created in.

        This method facilitates the deletion of a fine-tuned model identified by its model ID. It manages the API interaction to delete the model and handles various potential errors that might occur during this process.

        :method delete_fine_tuned_model: Deletes a fine-tuned model.
        :type delete_fine_tuned_model: function

        :param model_id: The ID of the fine-tuned model to delete.
        :type model_id: str

        :return: Confirmation of the deletion.
        :rtype: Dict

        :raises ValueError: If the model_id is not provided.
        :raises openai.APIConnectionError: If there's a connection error with the API.
        :raises openai.RateLimitError: If the request is rate-limited by the API.
        :raises openai.APIStatusError: If there's a status error from the API.
        :raises AttributeError: If an attribute error occurs during the process.
        :raises Exception: For any other exceptions that occur during the process.

        :example:

        ::

            >>> api = FineTuningAPI(api_key="your-api-key")
            >>> deletion_result = api.delete_fine_tuned_model(model_id="ft-model-12345")
            >>> print(deletion_result)
        """

        if not model_id:
            raise ValueError("A model_id must be provided.")

        try:
            response = self.client.models.delete(model_id)
            return response
        except openai.APIConnectionError as e:                            
                logger.error(f"[delete_fine_tuned_model] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[delete_fine_tuned_model] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[delete_fine_tuned_model] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[delete_fine_tuned_model] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"[delete_fine_tuned_model] An error occurred during model evaluation: {e}")

    def use_fine_tuned_model(self, model_name: str, user_prompt:str, system_prompt="You are a helpful assistant." ) -> str:
        """
        This method enables interaction with a fine-tuned model to generate responses based on provided messages.

        :method use_fine_tuned_model: Uses a specified fine-tuned model to generate responses to messages.
        :type use_fine_tuned_model: function

        :param model_name: The name of the fine-tuned model used for generating responses.
        :type model_name: str

        :param user_prompt: The user's message prompt for the model.
        :type user_prompt: str

        :param system_prompt: A predefined system message prompt, defaulting to "You are a helpful assistant."
        :type system_prompt: str, optional

        :return: The response generated by the fine-tuned model.
        :rtype: str

        :raises Exception: If an error occurs during the API request or while processing the response.

        :example:

        ::


            >>> api = FineTuningAPI(api_key="your-api-key")
            >>> response = api.use_fine_tuned_model(
                "ft:gpt-3.5-turbo:my-org:custom_suffix:id", 
                user_prompt="Hello!",
                system_prompt="You are a helpful assistant."
            )
            >>> print(response)
            'Response from the model...'
        """


        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
            )
            return response.choices[0].message
        except openai.APIConnectionError as e:                            
                logger.error(f"[use_fine_tuned_model] APIConnectionError error:\n{e}")

        except openai.RateLimitError as e:
            # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
            logger.error(f"[use_fine_tuned_model] RateLimit Error {e}. Trying again in 0.5 seconds...")

        except openai.APIStatusError as e:
            logger.error(f"[use_fine_tuned_model] APIStatusError:\n{e}")
            # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter

        except AttributeError as e:            
            logger.error(f"[use_fine_tuned_model] AttributeError:\n{e}")
            # You can also add additional error handling code here if needed

        except Exception as e:
            raise Exception(f"[use_fine_tuned_model] An error occurred during model evaluation: {e}")

    def run_dashboard(self):
        """
        This method runs a dashboard for various fine-tuning operations related to a model.

        :method run_dashboard: Launches an interactive dashboard allowing the user to perform various operations related to fine-tuning a model.
        :type run_dashboard: function

        :choice: User's choice from the dashboard menu for different operations.
        :type choice: str

        :file_path: File path for creating a fine-tune file.
        :type file_path: str, optional

        :purpose: Purpose of the file, either for fine-tuning or other purposes.
        :type purpose: str, optional

        :training_file: ID of the training file used for creating a fine-tuning job.
        :type training_file: str, optional

        :model: Name of the model used for fine-tuning.
        :type model: str, optional

        :suffix: Suffix for the fine-tuned model name.
        :type suffix: str, optional

        :batch_size: Batch size for training, either automatic or a specific number.
        :type batch_size: str, optional

        :learning_rate_multiplier: Learning rate multiplier, either automatic or a specific number.
        :type learning_rate_multiplier: str, optional

        :n_epochs: Number of epochs for training, either automatic or a specific number.
        :type n_epochs: str, optional

        :validation_file: ID of the validation file, if provided.
        :type validation_file: str, optional

        :job_id: ID of the fine-tuning job for retrieving state, cancelling, or listing events.
        :type job_id: str, optional

        :model_name: Name of the fine-tuned model for usage.
        :type model_name: str, optional

        :user_prompt: User prompt for testing the fine-tuned model.
        :type user_prompt: str, optional

        :system_prompt: System prompt for testing the fine-tuned model.
        :type system_prompt: str, optional

        :model_id: ID of the fine-tuned model to be deleted.
        :type model_id: str, optional

        :return: None
        """

        while True:
            print("\nMenu:")
            print("1. Create a fine-tune file")
            print("2. Create a fine-tuning job")
            print("3. List of tune-tune files")
            print("4. List 10 fine-tuning jobs")
            print("5. Retrieve the state of a fine-tune")
            print("6. Cancel a job")
            print("7. List up to 10 events from a fine-tuning job")
            print("8. Use a fine-tuned model")
            print("9. Delete a fine-tuned model")
            print("10. Exit")

            choice = input("Enter your choice: ")

            if choice == "1":
                file_path = input("Enter the file path: ")
                purpose = input("Enter the purpose (fine-tune/other): ")
                print(self.create_fine_tune_file(file_path, purpose))

            elif choice == "2":
                training_file = input("Enter training file ID: ")
                model = input("Enter model name: ")
                suffix = input("Enter suffix (optional): ") or None
                batch_size = input("Enter batch size (auto/number): ") or 'auto'
                learning_rate_multiplier = input("Enter learning rate multiplier (auto/number): ") or 'auto'
                n_epochs = input("Enter number of epochs (auto/number): ") or 'auto'
                validation_file = input("Enter validation file ID (optional): ") or None
                print(self.create_fine_tuning_job(training_file, model, suffix, batch_size, 
                                                  learning_rate_multiplier, n_epochs, validation_file))

            elif choice == "3":
                print(self.list_fine_tune_files())
                print()
            
            elif choice == "4":
                print(self.list_fine_tuning_jobs())

            elif choice == "5":
                job_id = input("Enter fine-tuning job ID: ")
                print(self.retrieve_fine_tuning_job(job_id))

            elif choice == "6":
                job_id = input("Enter fine-tuning job ID to cancel: ")
                print(self.cancel_fine_tuning_job(job_id))

            elif choice == "7":
                job_id = input("Enter fine-tuning job ID for events: ")
                print(self.list_events_fine_tuning_job(job_id))

            elif choice == "8":
                model_name = input("Enter fine-tuned model name: ")
                user_prompt = input("Enter user prompt: ")
                system_prompt = "You are a helpful assistant."

                try:
                    response = self.use_fine_tuned_model(
                        model_name=model_name, 
                        user_prompt=user_prompt,
                        system_prompt=system_prompt
                    )
                    print(response)
                except Exception as e:
                    print(f"An error occurred: {e}")
            
            elif choice == "9":
                model_id = input("Enter fine-tuned model ID to delete: ")
                print(self.delete_fine_tuned_model(model_id))

            elif choice == "10":
                break

            else:
                print("Invalid choice. Please try again.")

