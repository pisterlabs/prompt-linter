import json
import logging

from django.conf import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)

logger = logging.getLogger(__name__)


def get_openai_json_response(
    system_prompt: str, user_prompt: str, model: str = "gpt-3.5-turbo-1106", temperature: float = 0.0
) -> dict:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    json_content = json.loads(response.choices[0].message.content)
    logger.debug("OpenAI responded", extra={"response": json_content})
    return json_content
