import openai

openai.api_key = "Ваш openai токет"

def get_response(prompt: str, context:str) -> str:
    """
    Функция для запроса ответа от OpenAI.

    Args:
        prompt (str): Текст от пользователя для передачи в модель.
        context (str): Текущий контекст для передачи в модель.

    Returns:
        str: Ответ от модели.

    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Продолжи общение: {context}Пользователь: {prompt}\n",
        temperature=0,
        max_tokens=3000,
        top_p=1.0,
        frequency_penalty=0.2,
        presence_penalty=0.0,
        stop=None
    )
    return response.choices[0].text.lstrip()

def main():
    """
    Основная функция, которая обрабатывает ввод пользователя и выводит ответы модели.

    """
    context = "" # Начальный контекст - пустая строка
    while True:
        prompt = input("Вы: ")
        if not prompt.strip(): # Валидация, если строка пустая, то запросить ввод еще раз
            print("Пожалуйста, введите текст")
            continue
        context += f"Пользователь: {prompt}\n" # Добавление текста пользователя в контекст
        ans = get_response(prompt, context) # Получение ответа от модели
        context += f"{ans}\n" # Добавление ответа модели в контекст
        print(ans)

if __name__ == '__main__':
    main()
