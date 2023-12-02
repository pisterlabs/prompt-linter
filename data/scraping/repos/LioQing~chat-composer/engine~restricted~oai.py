import logging
from typing import Any, Dict, Optional

import openai

from config.logger import logger_config
from config.openai import openai_config
from core import models as core_models
from engine.oai.models import Chatcmpl
from oai import models

openai.api_key = openai_config.key
openai.api_base = openai_config.url
openai.api_type = openai_config.api_type
openai.api_version = openai_config.version


logger = logging.getLogger(__name__)
logger.setLevel(logger_config.level)


current_component: Optional[core_models.Component] = None


openai_chatcmpl = openai.ChatCompletion.create


def create_chatcmpl_models(request: Dict[str, Any], response: Chatcmpl):
    """Create a Chatcmpl models from the request and response"""
    if current_component is None:
        raise ValueError("current_component is None")

    chatcmpl_request = models.ChatcmplRequest.objects.create(
        response=models.Chatcmpl.objects.create(
            id=response.id,
            created=response.created,
            model=response.model,
            object=response.object,
            usage=models.Usage.objects.create(
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        ),
        request=request,
        component=current_component,
    )

    chatcmpl = chatcmpl_request.response

    for choice in response.choices:
        models.Choice.objects.create(
            chatcmpl=chatcmpl,
            finish_reason=choice.finish_reason,
            index=choice.index,
            message=models.Message.objects.create(
                content=choice.message.content,
                name=choice.message.name,
                function_call=models.FunctionCall.objects.create(
                    arguments=choice.message.function_call.arguments,
                    name=choice.message.function_call.name,
                )
                if choice.message.function_call is not None
                else None,
                role=choice.message.role,
            ),
        )
