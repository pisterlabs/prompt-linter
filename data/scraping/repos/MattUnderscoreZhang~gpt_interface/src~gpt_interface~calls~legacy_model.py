from openai import OpenAI

from gpt_interface.log import Log
from gpt_interface.options.system_message import SystemMessageOptions


def call_legacy_model(
    interface: OpenAI,
    model: str,
    log: Log,
    temperature: float,
    system_message_options: SystemMessageOptions,
) -> str:
    prompt = "\n".join([
        f"{message.role}: {message.content}"
        for message in log.messages
    ])
    if system_message_options.use_system_message:
        if system_message_options.message_at_end:
            prompt += "\nsystem: " + system_message_options.system_message
        else:
            prompt = "system: " + system_message_options.system_message + "\n" + prompt
    prompt += "\nassistant: "
    response = interface.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].text
