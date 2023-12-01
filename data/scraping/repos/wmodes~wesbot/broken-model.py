
import openai
import config
import mysecrets

# Set your OpenAI API key and organization (if applicable)
openai.api_key = mysecrets.OPENAI_API_KEY
openai.organization = config.OPENAI_ORG

chatParams = {
    "model": "ft:gpt-3.5-turbo-1106:artist::8KAhri96",
    "temperature": 0,
    "n": 1,
    "messages": [
        {"role": "assistant", "content": "What can I help you with today?"},
        {"role": "user", "content": "yo"},
    ]
}

print("ChatCompletion results with fine-tuned model:")
print(openai.ChatCompletion.create(**chatParams))

chatParams["functions"] = [
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  ]

print("ChatCompletion results with fine-tuned model and functions:")
print(openai.ChatCompletion.create(**chatParams))