import cohere
import configparser

config_read = configparser.ConfigParser()
config_read.read("config.ini")

api_key = config_read.get("api_keys", "generate_answers")

co = cohere.Client(api_key)


def generate_answer(prompt):
    pre_prompt = "You are an AI chatbot that helps users navigate and use theScore Bet app. Assume that all questions are asked in the context of theScore Bet app. To start, please answer this user's question: "
    response = co.generate(
        pre_prompt + prompt,
        max_tokens=1000,
        model=config_read.get("models", "generate_answers")
    )
    return response[0].text

if __name__ == "__main__":
    prompt = input("Prompt: ")
    print(generate_answer(prompt))
