from langchain.agents import tool
import requests

API_URL = "https://api-inference.huggingface.co/models/CreatorPhan/AloBacSi"
headers = {"Authorization": "Bearer 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"}

@tool
def vn_healthcare_bot(question: str) -> str:
    """Chatbot for Vietnamese Healthcare Questions only."""
    # try:
    payload = {"inputs": question}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text']
        # return "Sorry! HuggingFace is down, please try again later."