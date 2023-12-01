import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def init_kernel():
    kernel = sk.Kernel()
    kernel.add_chat_service(
        "gpt-3.5-turbo",
        OpenAIChatCompletion(
            "gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )
    skills = {}
    skills["AnswerSkill"] = kernel.import_semantic_skill_from_directory(
        CUR_DIR, "AnswerSkill"
    )
    skills["IntentSkill"] = kernel.import_semantic_skill_from_directory(
        CUR_DIR, "IntentSkill"
    )

    return kernel, skills
