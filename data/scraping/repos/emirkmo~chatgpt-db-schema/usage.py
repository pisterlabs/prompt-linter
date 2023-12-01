from idd_ai.chatgpt import OpenAIObject


# 	$0.002 / 1K tokens
def get_cost(tokens: int, cost: float = 0.002 / 1000) -> float:
    return tokens * 0.002 / 1000


def print_useage_cost(feedback: OpenAIObject) -> None:
    print(f"Usage: {feedback.usage}")
    cost = get_cost(sum(feedback.usage.values()))
    print(f"Cost: ${cost} or SEK {feedback.cost*10} kr.")
