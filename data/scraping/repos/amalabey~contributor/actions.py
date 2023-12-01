import os
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from platform_integration.azure_devops.api import AzureDevOpsApiClient
from platform_integration.pull_requests import (
    PullRequestDataProvider,
    PullRequestDecoratorService,
)
from core.code_review.changeset import ChangesetProvider
from core.code_review.comments import ReviewCommentProvider
from core.code_review.lang import LanguageDetector
from core.code_review.reviewer import CodeReviwer
from platform_integration.constants import AZURE_DEVOPS_PLATFORM_NAME
from core.code_review.syntax import SyntaxProvider

DEFAULT_TEMPERATURE = 0.0


def review_pull_request(
    pull_request_id: str,
    env_config_file: str = ".env",
    temperature: float = DEFAULT_TEMPERATURE,
    platform: str = AZURE_DEVOPS_PLATFORM_NAME,
):
    openai_type = os.getenv("OPENAI_API_TYPE")
    if openai_type is not None and openai_type == "azure":
        model_name = os.getenv("MODEL_NAME")
        deployment_name = os.getenv("DEPLOYMENT_NAME")
        llm_model = AzureChatOpenAI(
            model_name=model_name,
            deployment_name=deployment_name,
            temperature=temperature,
        )
    else:
        model_name = os.getenv("MODEL_NAME")
        llm_model = ChatOpenAI(model_name=model_name, temperature=temperature)

    syntax_provider = SyntaxProvider(llm_model, verbose=True)
    changeset_provider = ChangesetProvider()
    review_provider = ReviewCommentProvider(llm_model, verbose=True)
    lang_detector = LanguageDetector()

    if platform == AZURE_DEVOPS_PLATFORM_NAME:
        org = os.getenv("AZURE_DEVOPS_ORG")
        project = os.getenv("AZURE_DEVOPS_PROJECT")
        azure_devops_client = AzureDevOpsApiClient(org, project)
        pr_data_provider = PullRequestDataProvider(azure_devops_client)
        pr_decorator_svc = PullRequestDecoratorService(azure_devops_client)
    else:
        raise Exception(f"Unsupported platform: {platform}")

    code_reviewer = CodeReviwer(
        pr_data_provider,
        pr_decorator_svc,
        changeset_provider,
        syntax_provider,
        lang_detector,
        review_provider,
    )
    code_reviewer.review_pull_request(pull_request_id)
