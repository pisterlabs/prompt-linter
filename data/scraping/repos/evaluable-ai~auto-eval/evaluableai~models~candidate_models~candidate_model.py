import logging
import os

from evaluableai.models.candidate_models.candidate_model_names import CandidateModelName
from evaluableai.models.candidate_models.hugging_face import HuggingFace
from evaluableai.models.candidate_models.open_ai import OpenAICandidate
from evaluableai.models.candidate_models.open_ai import OpenAiChatCandidate
from evaluableai.models.candidate_models.rest_candidate_model import CustomModelClass
from evaluableai.models.model import Model

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CandidateModel(Model):
    """This class represents a wrapper for different candidate AI models."""

    def __init__(self, model_name, model_version, api_key_env=None, api_endpoint=None, api_auth_token=None):
        """Initialize a candidate model based on the given parameters."""
        if model_name == CandidateModelName.CUSTOM:
            if not api_auth_token:
                raise ValueError("API auth token must be provided for custom models.")
            self._instance = CustomModelClass(model_name, model_version, api_endpoint, api_auth_token)
        else:
            if not api_key_env:
                raise ValueError("API key environment variable must be provided for non-custom models.")
            self._api_key = self._get_api_key(api_key_env)
            self._instance = self._create_instance(model_name, model_version, self._api_key)

    def _get_api_key(self, api_key_env):
        """Retrieve the API key from the environment variable."""
        api_key = os.getenv(api_key_env)
        if api_key is None:
            raise EnvironmentError(f"API key environment variable '{api_key_env}' not found.")
        return api_key

    def _create_instance(self, model_name, model_version, api_key):
        """Create an instance of the specified model."""
        try:
            if model_name == CandidateModelName.HUGGING_FACE:
                return HuggingFace(model_version, api_key)
            elif model_name == CandidateModelName.OPEN_AI:
                return OpenAICandidate(model_version, api_key)
            elif model_name == CandidateModelName.OPEN_AI_CHAT:
                return OpenAiChatCandidate(model_version, api_key)
            else:
                raise ValueError(f"Invalid model name: {model_name}")
        except (ValueError, EnvironmentError) as e:
            logging.error("Model instantiation failed: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred during model instantiation: %s", e)
            raise

    @property
    def model_name(self):
        """Model name property."""
        return self._instance.model_name;

    @property
    def api_key(self):
        """API key property."""
        return self._instance.api_key;

    @property
    def model_version(self):
        """Model version property."""
        return self._instance.model_version

    def generate_response(self, input_frame):
        """Generates a response from the model for the given input frame."""
        try:
            return self._instance.generate_response(input_frame)
        except Exception as e:
            logging.error("Error generating response: %s", e)
            raise

    @property
    def response_list(self):
        """Property to get the response list from the model instance."""
        try:
            return self._instance.response_list
        except Exception as e:
            logging.error("Error accessing response list: %s", e)
