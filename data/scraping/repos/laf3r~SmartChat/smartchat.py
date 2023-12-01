import tkinter as tk
import openai

openai.api_key = str(input("Введите ваш api-ключ к OpenAI: "))
ai_model = ["text-davinci-002", "text-davinci-003"]
ai_n = int(input("Программа поддерживает работу с двумя моделями OpenAI:\n[1]text-davinci-002\n[2]text-davinci-003\nВыберите нейронную сеть:"))
root = tk.Tk()
root.title("SmartChat GPT-2")
root.geometry("420x395")
root.minsize(420, 395)  # минимальный размер окна
#root.maxsize(425, 390)  # максимальный размер окна

chat_log = tk.Text(root, width=50, height=20)
chat_log.grid(row=0, column=0, columnspan=2)

input_field = tk.Entry(root)
input_field.grid(row=1, column=0, sticky="ew")



#Для отладки. Отображает разрешение окна
#def on_window_resize(event):
 #   width = event.width
   # height = event.height
   # print("Window size: {}x{}".format(width, height))

def get_gpt3_response(prompt):
    # Задайте заголовки для запроса к API GPT-2
    tokens=3000
    #Максимальное число токенов 4,097. Токены влияют на длинну ответа ИИ. Также, токены считываются как количество символов в вашем сообщении.
    response = openai.Completion.create(
       engine = ai_model[ai_n-1],
       prompt = prompt,
       max_tokens = tokens,
       n = 1,
       stop = None, #Останавливает генерацию своего ответа при достижении этих символов ["!", "?"],
       temperature = 0.5,
    )

    message = response.choices[0].text.strip()
    return message


# Добавляем тег для имени пользователя
chat_log.tag_configure("username", foreground="#66ff33")

# Добавляем тег для ИИ
chat_log.tag_configure("botname", foreground="#66ffff")

# Добавляем тег для сообщений
chat_log.tag_configure("message", foreground="white")

def send_message(event=None):
    message = input_field.get()
    # Добавляем имя пользователя с тегом "username"
    chat_log.insert(tk.END, 'You: ', "username")
    # Добавляем сообщение с тегом "message"
    chat_log.insert(tk.END, '{}\n'.format(message), "message")
    response = get_gpt3_response(message)
    # Добавляем имя ИИ с тегом "botname"
    chat_log.insert(tk.END, f'{ai_model[ai_n-1]}: ', "botname")
    # Добавляем сообщение от ИИ без тега
    chat_log.insert(tk.END, '{}\n'.format(response), "message")
    input_field.delete(0, tk.END)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1)

root.bind("<Return>", send_message)

def save_chat():
    with open("chat_log.txt", "w") as f:
        f.write(chat_log.get("1.0", tk.END))

save_button = tk.Button(root, text="Save Chat", command=save_chat)
save_button.grid(row=0, column=1, sticky="ne")

chat_log.configure(state='normal')

# настройка ширины второго столбца
root.columnconfigure(1, weight=0)

# настройка адаптивной ширины поля ввода текста
root.columnconfigure(0, weight=1)


root.configure(bg="#333333")  # тёмно-серый фон
chat_log.configure(fg="#f252e6")  # фиолетово-неоновые буквы
chat_log.configure(bg="#222222")  # тёмно-серый фон для текстового поля
input_field.configure(fg="#f252e6",font=("Helvetica", 12))  # фиолетово-неоновые буквы для поля ввода
input_field.configure(bg="#444444")  # тёмно-серый фон для поля ввода
send_button.configure(bg="#444444", activebackground="#555555", fg="#f252e6", font=("Helvetica", 11))  # настройки кнопки "Send"
save_button.configure(bg="#444444", activebackground="#555555", fg="#f252e6", font=("Helvetica", 11))  # настройки кнопки "Save Chat"


chat_log.configure(font=("Helvetica", 12))
root.mainloop()

