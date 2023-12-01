import openai
import config


openai.api_key = config.api_key
openai.organization = config.organization


def create_completion(messages, model="gpt-3.5-turbo", max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    return response


def main():
    messages = [{"role": "system", "content": "Debes darme ideas para hacer videos de matemáticas para YouTube."}]
    print('¡Hola! ¿Qué te puedo sugerir? (Escribe "salir" para salir del programa)')
    while True:
        prompt = input(">>> ")
        if prompt == "salir":
            break
        messages.append({"role": "user", "content": prompt})
        response = create_completion(messages)
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        print(content)


if __name__ == "__main__":
    main()
