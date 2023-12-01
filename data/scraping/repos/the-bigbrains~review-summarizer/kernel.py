from dotenv import load_dotenv
import semantic_kernel as sk
import os
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")

kernel = sk.Kernel()

if not api_key:
    api_key = ""

kernel.add_chat_service(
    "chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo-16k", api_key, org_id))
