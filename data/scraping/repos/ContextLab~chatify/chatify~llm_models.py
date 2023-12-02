import os
import warnings

with warnings.catch_warnings():  # catch warnings about accelerate library
    warnings.simplefilter("ignore")
    from langchain.llms import OpenAI, HuggingFacePipeline, LlamaCpp
    from langchain.llms.base import LLM
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError:
        hf_hub_download = None  # ignore missing library unless needed later

from .utils import FakeListLLM


class ModelsFactory:
    """A factory class for creating different models."""

    def __init__(self, *args) -> None:
        """
        Initializes the ModelsFactory instance.

        Parameters
        ----------
        *args : tuple
            Variable-length arguments.

        Returns
        -------
        None
        """

        return None

    def get_model(self, model_config):
        """Returns the initialized model based on the model configuration.

        Parameters
        ----------
        model_config : dict
            Configuration for the desired model.

        Returns
        -------
        model : object
            Initialized model based on the model configuration.

        Raises
        ------
        RuntimeError
            If the specified model is not supported.
        """
        model_ = model_config['model']

        # Collect all the models
        models = {
            'open_ai_model': OpenAIModel,
            'open_ai_chat_model': OpenAIChatModel,
            'fake_model': FakeLLMModel,
            'cached_model': CachedLLMModel,
            'huggingface_model': HuggingFaceModel,
            'llama_model': LlamaModel,
            'proxy': ProxyModel,
        }

        if model_ in models.keys():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if type(models[model_]) == str:
                    return models[model_]
                else:
                    return models[model_](model_config).init_model()
        else:
            raise RuntimeError(f"{model_} is not supported yet!")


class BaseLLMModel:
    """Base class for Language Model (LLM) models."""

    def __init__(self, model_config) -> None:
        """Initializes the BaseLLMModel instance.

        Parameters
        ----------
        model_config : dict
            Configuration for the model.

        Returns
        -------
        None
        """
        self.model_config = model_config
        self.llm_model = None

    def init_model(self, *args, **kwargs):
        """Initializes the LLM model (to be implemented by derived classes).

        Parameters
        ----------
        *args : tuple
            Variable-length arguments.
        **kwargs : dict
            Arbitrary keyword arguments.

        Raises
        ------
        NotImplementedError
            If not implemented by derived classes.
        """
        raise NotImplementedError


class OpenAIModel(BaseLLMModel):
    """Class representing an OpenAI Chat Model derived from BaseLLMModel."""

    def __init__(self, model_config) -> None:
        """Initializes the OpenAIChatModel instance.

        Parameters
        ----------
        model_config : dict
            Configuration for the model.

        Returns
        -------
        None
        """
        super().__init__(model_config)

    def init_model(self):
        """Initializes the OpenAI Chat Model.

        Returns
        -------
        llm_model : ChatOpenAI
            Initialized OpenAI Chat Model.
        """
        if self.model_config['open_ai_key'] is None:
            raise ValueError(f'openai_api_key value cannot be None')

        os.environ["OPENAI_API_KEY"] = self.model_config['open_ai_key']

        llm_model = OpenAI(
            temperature=0.85,
            openai_api_key=self.model_config['open_ai_key'],
            model_name=self.model_config['model_name'],
            presence_penalty=0.1,
            max_tokens=self.model_config['max_tokens'],
        )
        return llm_model


class OpenAIChatModel(BaseLLMModel):
    """Class representing an OpenAI Chat Model derived from BaseLLMModel."""

    def __init__(self, model_config) -> None:
        """Initializes the OpenAIChatModel instance.

        Parameters
        ----------
        model_config : dict
            Configuration for the model.

        Returns
        -------
        None
        """
        super().__init__(model_config)

    def init_model(self):
        """Initializes the OpenAI Chat Model.

        Returns
        -------
        llm_model : ChatOpenAI
            Initialized OpenAI Chat Model.
        """
        if self.model_config['open_ai_key'] is None:
            raise ValueError(f'openai_api_key value cannot be None')

        llm_model = ChatOpenAI(
            temperature=0.85,
            openai_api_key=self.model_config['open_ai_key'],
            model_name=self.model_config['model_name'],
            presence_penalty=0.1,
            max_tokens=self.model_config['max_tokens'],
        )
        return llm_model


class FakeLLMModel(BaseLLMModel):
    def __init__(self, model_config) -> None:
        """Initializes the FakeListLLM instance.

        Parameters
        ----------
        model_config : dict
            Configuration for the model.

        Returns
        -------
        None
        """
        super().__init__(model_config)

    def init_model(self):
        """Initializes the Fake Chat Model.

        Returns
        -------
        llm_model : FakeListLLM
            Initialized Fake Chat Model.
        """
        responses = [
            'The explanation you requested has not been included in Chatify\'s cache. You\'ll need to enable interactive mode to generate a response. Please see the [Chatify GitHub repository](https://github.com/ContextLab/chatify) for instructions.  Note that generating responses to uncached content will require an [OpenAI API Key](https://platform.openai.com/account/api-keys).'
        ]
        llm_model = FakeListLLM(responses=responses)
        return llm_model


class CachedLLMModel(BaseLLMModel):
    def __init__(self, model_config) -> None:
        """Initializes the FakeListLLM instance.

        Parameters
        ----------
        model_config : dict
            Configuration for the model.

        Returns
        -------
        None
        """
        super().__init__(model_config)

    def init_model(self):
        """Initializes the Fake Chat Model.

        Returns
        -------
        llm_model : FakeListLLM
            Initialized Fake Chat Model.
        """
        llm_model = FakeListLLM(
            responses=[
                f'The explanation you requested has not been included in Chatify\'s cache. You\'ll need to enable interactive mode to generate a response. Please see the [Chatify GitHub repository](https://github.com/ContextLab/chatify) for instructions.  Note that generating responses to uncached content will require an [OpenAI API Key](https://platform.openai.com/account/api-keys).'
            ]
        )
        return llm_model


class HuggingFaceModel(BaseLLMModel):
    def __init__(self, model_config) -> None:
        """Initializes the model instance.

        Parameters
        ----------
        model_config : dict
            Configuration for the model.

        Returns
        -------
        None
        """
        super().__init__(model_config)

    def init_model(self):
        """Initializes the OpenAI Chat Model.

        Returns
        -------
        llm_model : HuggingFaceModel
            Initialized Hugging Face Chat Model.
        """
        self.proxy = self.model_config['proxy']
        self.proxy_port = self.model_config['proxy_port']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                llm = HuggingFacePipeline.from_model_id(
                    model_id=self.model_config['model_name'],
                    task='text-generation',
                    device=0,
                    model_kwargs={'max_length': self.model_config['max_tokens']},
                )
            except:
                llm = HuggingFacePipeline.from_model_id(
                    model_id=self.model_config['model_name'],
                    task='text-generation',
                    model_kwargs={
                        'max_length': self.model_config['max_tokens'],
                        'temperature': 0.85,
                        'presence_penalty': 0.1,
                    },
                )
        return llm


class LlamaModel(BaseLLMModel):
    def __init__(self, model_config) -> None:
        """Initializes the model instance.

        Parameters
        ----------
        model_config : dict
            Configuration for the model.

        Returns
        -------
        None
        """
        super().__init__(model_config)

    def init_model(self):
        """Initializes the OpenAI Chat Model.

        Returns
        -------
        llm_model : HuggingFaceModel
            Initialized Hugging Face Chat Model.
        """
        self.model_path = hf_hub_download(
            repo_id=self.model_config['model_name'],
            filename=self.model_config['weights_fname'],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            try:
                llm = LlamaCpp(
                    model_path=self.model_path,
                    max_tokens=self.model_config['max_tokens'],
                    n_gpu_layers=self.model_config['n_gpu_layers'],
                    n_batch=self.model_config['n_batch'],
                    callback_manager=callback_manager,
                    verbose=True,
                )
            except:
                llm = LlamaCpp(
                    model_path=self.model_path,
                    max_tokens=self.model_config['max_tokens'],
                    n_batch=self.model_config['n_batch'],
                    callback_manager=callback_manager,
                    verbose=True,
                )
        return llm


class ProxyModel(BaseLLMModel):
    def __init__(self, model_config) -> None:
        """Initializes the model instance.

        Parameters
        ----------
        model_config : dict
            Configuration for the model.

        Returns
        -------
        None
        """
        super().__init__(model_config)

    def init_model(self):
        return None
