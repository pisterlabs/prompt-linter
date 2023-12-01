from config import GPTModels, get_chat_client
import json
import logging


def create_prompt(text: str) -> str:
    return """Create a flow chart based on the following information and provide the output in JSON format compatible with GoJS, including nodes and links. The flow chart should represent key infomation in the text:

{description}

Return data in the following JSON Structure:
{{ "nodes": [...], "links": [...] }}
The flow chart should optimize learning. Keep it simple, clear and detailed. Keep colors light.""".format(
        description=text
    )


gpt = get_chat_client(GPTModels.GPT4.value, "json_object")


def flow_reponse(text):
    prompt = create_prompt(text)
    answer = str(gpt(prompt)).strip()
    try:
        response = json.loads(answer)
        logging.info("Fetched Response from OPENAI")
        if (not response["nodes"]) or (not response["links"]):
            return {"data": "Invalid Data From OpenAI", "success": False}
        return {"data": response, "success": True}
    except Exception as e:
        return {"data": f"Error Parsing Json: {str(e)}", "success": False}
