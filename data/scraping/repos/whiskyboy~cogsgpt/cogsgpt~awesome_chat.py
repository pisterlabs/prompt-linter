from __future__ import annotations

import json
import os
import pkg_resources
import re
from typing import Dict, List
import yaml

from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import PromptValue
from langchain.memory import ConversationBufferMemory

from cogsgpt.schema import ArgsType, LanguageType
from cogsgpt.llm import LLMManager
from cogsgpt.cogsmodel import *
from cogsgpt.logger import logger
from cogsgpt.utils import num_tokens_from_string


TaskMap = {
    "form-recognizer": FormReadModel,
    "form-layout": FormLayoutModel,
    "form-key-value": FormKeyValueModel,
    "w2-tax-form": W2TaxFormModel,
    "invoice-form": InvoiceFormModel,
    "receipt-form": ReceiptFormModel,
    "id-document-form": IDDocumentFormModel,
    "business-card-form": BusinessCardFormModel,
    "image-caption": ImageCaptionModel,
    "object-detection": ObjectDetectionModel,
    "image-tagging": ImageTaggingModel,
    "people-detection": PeopleDetectionModel,
    "smart-crop": SmartCropModel,
    "OCR": ImageTextModel,
    "image-categorize": ImageCategorizeModel,
    "brand-detection": BrandDetectionModel,
    "face-detection": FaceDetectionModel,
    "adult-content-detection": AdultContentDetectionModel,
    "color-scheme-detection": ColorSchemeDetectionModel,
    "image-type-detection": ImageTypeDetectionModel,
    "background-remover": BackgroundRemover,
    "text2speech": Text2SpeechModel,
    "speech2text": Speech2TextModel,
    "key-phrase": KeyPhraseModel,
    "entity-linking": EntityLinkingModel,
    "entity-recognition": EntityRecognitionModel,
    "personally-identifiable-information": PIIModel,
    "sentiment-analysis": SentimentAnalysisModel,
    "language-detection": LanguageDetectionModel,
    # "extract-summarize": ExtractSummarizeModel,
    # "abstract-summarize": AbstractSummarizeModel,
    # "text-translation": TextTranslationModel,
    "text-generation": TextGenerationModel,
    "text-summarization": GenerativeTextSummarizationModel,
    "text-translation": GenerativeTextTranslationModel,
}


class CogsGPT():
    def __init__(self,
                 model_name: str = "",
                 deployment_name: str = "",
                 deployment_version: str = "",
                 temperature: float = 0.7,
                 request_timeout: int = 60,
                 max_retries: int = 6,
                 max_tokens: int | None = None,
                 save_history: bool = True,
                 verbose: bool = False,
                 ) -> None:
        """
        The CogsGPT class is the main class for the CogsGPT library. It provides a simple interface to chat with Azure Cognitive Services.

        Args:
            model_name (str, optional): The name of the OpenAI model. Defaults to "". If not specified, it will be read from the environment variable OPENAI_MODEL_NAME.
            deployment_name (str, optional): The name of the Azure deployment. Defaults to "". If not specified, it will be read from the environment variable OPENAI_DEPLOYMENT_NAME.
            deployment_version (str, optional): The version of the Azure deployment. Defaults to "". If not specified, it will be read from the environment variable OPENAI_DEPLOYMENT_VERSION.
            temperature (float, optional): The temperature of the OpenAI model. Defaults to 0.7.
            request_timeout (int, optional): The timeout of the OpenAI model. Defaults to 60.
            max_retries (int, optional): The maximum number of retries of the OpenAI model. Defaults to 6.
            max_tokens (int | None, optional): The maximum number of tokens of the OpenAI model. Defaults to None.
            save_history (bool, optional): Whether to save the chat history. Defaults to True.
            verbose (bool, optional): Whether to print verbose logs. Defaults to False.

        Examples:
            >>> from cogsgpt import CogsGPT
            >>> chatbot = CogsGPT()
            >>> chatbot.chat("What's the content of this image?")
        """
        if model_name != "":
            os.environ["OPENAI_MODEL_NAME"] = model_name
        if deployment_name != "":
            os.environ["OPENAI_DEPLOYMENT_NAME"] = deployment_name
        if deployment_version != "":
            os.environ["OPENAI_DEPLOYMENT_VERSION"] = deployment_version

        self._prompts_config = yaml.load(
            open(pkg_resources.resource_filename('cogsgpt', 'metas/prompts.yaml'), "r"),
            Loader=yaml.FullLoader
            )

        self._task_metas = yaml.load(
            open(pkg_resources.resource_filename('cogsgpt', 'metas/task_metas.yaml'), "r"),
            Loader=yaml.FullLoader
            )

        self._parse_task_presteps = json.load(
            open(pkg_resources.resource_filename('cogsgpt', 'metas/parse_task_presteps.json'), 'r')
        )
        self._generate_response_presteps = json.load(
            open(pkg_resources.resource_filename('cogsgpt', 'metas/generate_response_presteps.json'), 'r')
        )

        self._parse_task_system_prompt = self._prompts_config["parse_task"]["system"]
        self._parse_task_user_prompt = self._prompts_config["parse_task"]["user"]
        self._generate_response_system_prompt = self._prompts_config["generate_response"]["system"]
        self._generate_response_user_prompt = self._prompts_config["generate_response"]["user"]

        self.parse_task_prompt = self._create_prompt(
            system_prompt=self._parse_task_system_prompt,
            system_prompt_vars=["task_metas", "args_type_list", "supported_language_list"],
            user_prompt=self._parse_task_user_prompt,
            user_prompt_vars=["history", "input"],
            presteps=self._parse_task_presteps,
        )
        self.generate_response_prompt = self._create_prompt(
            system_prompt=self._generate_response_system_prompt,
            user_prompt=self._generate_response_user_prompt,
            presteps=self._generate_response_presteps,
            presteps_vars=["input", "processes"],
        )

        self.memory = ConversationBufferMemory(memory_key='history')

        self.chatbot = LLMManager(
            model_name=model_name,
            deployment_name=deployment_name,
            deployment_version=deployment_version,
            temperature=temperature,
            request_timeout=request_timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
            verbose=verbose,
        ).LLM

        self._save_history = save_history

        if verbose:
            logger.setLevel("DEBUG")

    def _create_prompt(self, 
                       system_prompt: str = "", system_prompt_vars: List[str] = [],
                       user_prompt: str = "", user_prompt_vars: List[str] = [], 
                       presteps: List[Dict] = [], presteps_vars: List[str] = []) -> ChatPromptTemplate:
        system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=system_prompt,
                input_variables=system_prompt_vars,
                template_format="jinja2",
                validate_template=False
            )
        )

        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=user_prompt,
                input_variables=user_prompt_vars,
                template_format="jinja2",
                validate_template=False
            )
        )

        presteps_message_prompts = []
        for prestep in presteps:
            role_message_prompt_template = AIMessagePromptTemplate if prestep["role"] == "assistant" else HumanMessagePromptTemplate
            presteps_message_prompts.append(
                role_message_prompt_template(
                    prompt=PromptTemplate(
                        template=prestep["content"],
                        input_variables=presteps_vars,
                        template_format="jinja2",
                        validate_template=False
                    )
                )
            )

        return ChatPromptTemplate.from_messages([system_message_prompt, *presteps_message_prompts, human_message_prompt])

    def _format_parse_task_prompt(self, human_input: str) -> PromptValue:
        return self.parse_task_prompt.format_prompt(
            history=self.load_context(),
            input=human_input,
            task_metas=self._task_metas,
            args_type_list=[t.value for t in ArgsType],
            supported_language_list=[l.value for l in LanguageType]
        )

    def _format_generate_response_prompt(self, human_input: str, task_list: List[Dict]) -> PromptValue:
        return self.generate_response_prompt.format_prompt(
            input=human_input,
            processes=task_list
        )

    def _collect_deps_results(self, task_args: Dict, task_list: List[Dict]) -> Dict:
        """
        Collect the results of the dependencies tasks.
        """
        dep_pattern = r'<GENERATED>-(\d+)'
        for arg_name, arg_value in task_args.items():
            matches = re.findall(dep_pattern, arg_value)
            for match in matches:
                dep_id = int(match)
                dep_result = task_list[dep_id].get("result", "")
                task_args[arg_name] = task_args[arg_name].replace(f"<GENERATED>-{dep_id}", dep_result)
        return task_args

    def save_context(self, human_input: str, ai_response: str) -> None:
        """
        Save user input and AI response to memory.

        Args:
            human_input (str): The user input.
            ai_response (str): The AI response.
        """
        self.memory.chat_memory.add_user_message(human_input)
        self.memory.chat_memory.add_ai_message(ai_response)

    def load_context(self) -> str:
        """
        Load the conversation history from memory.

        Returns:
            str: The conversation history.
        """
        return self.memory.load_memory_variables({})['history']

    def parse_tasks(self, human_input: str) -> List[Dict]:
        """
        Parse user input into a series of Azure Cognitive Service tasks.

        Args:
            human_input (str): User input.

        Returns:
            List[Dict]: A list of Azure Cognitive Service tasks.
        """
        messages = self._format_parse_task_prompt(human_input)
        messages_tokens = num_tokens_from_string(messages.to_string())
        logger.debug(f"[Parse Task] Prompt: {messages.to_string()}")
        logger.debug(f"[Parse Task] Prompt Tokens: {messages_tokens}")

        response = self.chatbot(messages.to_messages()).content
        response_tokens = num_tokens_from_string(response)
        logger.info(f"[Parse Task] Result: {response}")
        logger.debug(f"[Parse Task] Result Tokens: {response_tokens}")

        return json.loads(response)

    def execute_tasks(self, task_list: List[Dict]) -> List[Dict]:
        """
        Execute a series of Azure Cognitive Service tasks.

        Args:
            task_list (List[Dict]): A list of Azure Cognitive Service tasks.

        Returns:
            List[Dict]: A list of Azure Cognitive Service tasks with results.
        """
        for i, task in enumerate(task_list):
            try:
                task_name = task["task"]
                logger.info(f"[Execute Task] Executing task{i}: {task_name}")
                if task_name not in TaskMap:
                    logger.warning(f"[Execute Task] Task{i} {task_name} not supported")
                else:
                    task_model = TaskMap[task_name]()
                    task["args"] = self._collect_deps_results(task["args"], task_list)
                    logger.debug(f"[Execute Task] Task{i} {task_name} args: {task['args']}")
                    task["result"] = task_model.run(**task["args"])
                    logger.info(f"[Execute Task] Task{i} {task_name} result: {task['result']}")
            except Exception as e:
                logger.error(f"[Execute Task] Task{i} {task_name} failed: {e}")
        return task_list

    def generate_response(self, human_input: str, task_result_list: List[Dict]) -> str:
        """
        Generate final response from a series of Azure Cognitive Service results.

        Args:
            human_input (str): User input.
            task_result_list (List[Dict]): A list of Azure Cognitive Service tasks with results.

        Returns:
            str: Final response.
        """
        messages = self._format_generate_response_prompt(human_input, task_result_list)
        messages_tokens = num_tokens_from_string(messages.to_string())
        logger.debug(f"[Generate Response] Prompt: {messages.to_string()}")
        logger.debug(f"[Generate Response] Prompt Tokens: {messages_tokens}")

        response = self.chatbot(messages.to_messages()).content
        response_tokens = num_tokens_from_string(response)
        logger.info(f"[Generate Response] Result: {response}")
        logger.debug(f"[Generate Response] Result Tokens: {response_tokens}")

        return response

    def chat(self, human_input: str) -> str:
        """
        This is the main method of CogsGPT class. It calls the following methods in order to generate a final response:
        1. parse_tasks: parse user input into a series of Azure Cognitive Service tasks
        2. execute_tasks: execute these Azure Cognitive Service tasks
        3. generate_response: generate final response from the results of Azure Cognitive Service tasks
        4. save_context: save user input and AI response to memory

        Args:
            human_input (str): User input.

        Returns:
            str: Final response.
        """
        # 1. parse tasks
        logger.info("[CogsGPT] Parsing user input...")
        task_list = self.parse_tasks(human_input)

        # 2. execute tasks
        logger.info("[CogsGPT] Executing tasks...")
        task_result_list = self.execute_tasks(task_list)

        # 3. generate response
        logger.info("[CogsGPT] Generating final response...")
        response = self.generate_response(human_input, task_result_list)

        # 4. save human_input/response into memory
        if self._save_history:
            logger.info("[CogsGPT] Saving context...")
            self.save_context(human_input, response)

        return response