"""
Demonstrates how to use the `ChatInterface` to create a chatbot using
[Mistral](https://docs.mistral.ai) through
[CTransformers](https://github.com/marella/ctransformers). The chatbot includes a
memory of the conversation history.
"""

import panel as pn
from ctransformers import AutoConfig, AutoModelForCausalLM, Config

pn.extension(design="material")

SYSTEM_INSTRUCTIONS = "Do what the user requests."


def apply_template(history):
    history = [message for message in history if message.user != "System"]
    prompt = ""
    for i, message in enumerate(history):
        if i == 0:
            prompt += f"<s>[INST]{SYSTEM_INSTRUCTIONS} {message.object}[/INST]"
        else:
            if message.user == "Mistral":
                prompt += f"{message.object}</s>"
            else:
                prompt += f"""[INST]{message.object}[/INST]"""
    return prompt


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if "mistral" not in llms:
        instance.placeholder_text = "Downloading model; please wait..."
        config = AutoConfig(
            config=Config(
                temperature=0.5, max_new_tokens=2048, context_length=2048, gpu_layers=1
            ),
        )
        llms["mistral"] = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            config=config,
        )

    llm = llms["mistral"]
    history = [message for message in instance.objects]
    prompt = apply_template(history)
    response = llm(prompt, stream=True)
    message = ""
    for token in response:
        message += token
        yield message


llms = {}
chat_interface = pn.chat.ChatInterface(
    callback=callback,
    callback_user="Mistral",
)
chat_interface.send(
    "Send a message to get a reply from Mistral!", user="System", respond=False
)
chat_interface.servable()
