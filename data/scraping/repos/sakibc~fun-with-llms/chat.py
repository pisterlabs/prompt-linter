from ui.chatbot_cmd import ChatbotCmd
from ui.chatbot_gradio import ChatbotGradio
from chatbot.basic_chatbot import BasicChatbot
from langchain.llms import OpenAI
from llm.langchain_wrapper import LangChainWrapper
from dotenv import load_dotenv
from ui.chatbot_server import ChatbotServer
import os
import json
from vectorstores import load_vectorstores
import langchain

import argparse


def main():
    load_dotenv()

    models = [name.split(".")[0] for name in os.listdir("models")] + ["openai"]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_name",
        help="Name of the model to use",
        choices=models,
    )
    parser.add_argument(
        "--size",
        help="Size of the model to use, depends on model chosen",
    )
    parser.add_argument(
        "--hosted", action="store_true", help="Use the model hosted in the cloud"
    )
    parser.add_argument(
        "--backend",
        help="Backend to use for model",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    parser.add_argument(
        "--quantization",
        help="Quantization to use for model",
        choices=["none", "int8", "nf4"],
        default="none",
    )
    parser.add_argument(
        "ui_type", help="Type of UI to use", choices=["cmd", "gradio", "server"]
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    model_name = args.model_name
    size = args.size
    backend = args.backend
    ui_type = args.ui_type
    verbose = args.verbose
    quantization = args.quantization

    langchain.verbose = verbose

    if model_name == "openai":
        llm = OpenAI(temperature=0.2)

    else:
        if args.hosted:
            model_type = "hosted"
        else:
            model_type = "local"

        with open(os.path.join("models", f"{model_name}.json")) as f:
            model_config = json.load(f)

        if model_type == "local":
            from llm.model import Model

            model = Model(
                model_name=model_name,
                model_config=model_config,
                size=size,
                backend=backend,
                quantization=quantization,
                verbose=verbose,
            )
        elif model_type == "hosted":
            from llm.hosted_model import HostedModel

            url = f"http://{os.getenv('REMOTE_HOST')}:8000"
            token = os.getenv("HOSTED_MODEL_TOKEN")

            model = HostedModel(model_name, model_config, url=url, token=token)

        llm = LangChainWrapper(model=model)

    bot = BasicChatbot(
        llm=llm,
    )

    if ui_type == "cmd":
        server = ChatbotCmd(bot)
    elif ui_type == "gradio":
        server = ChatbotGradio(bot)
    elif ui_type == "server":
        server = ChatbotServer(bot)

    server.run()


if __name__ == "__main__":
    main()
