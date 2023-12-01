import openai

openai.api_key = "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"  # aquí vuestra clave api
# Generar texto
exit_program = "Nada"
print("Bienvenido a ChatGPT, a continuación puedes escribir tu prompt y él te responderá. Para salir del programa "
      "escribe exit. ¡Disfruta!")
while exit_program != 1:
    prompt = input("Escribe tu prompt: ")

    if prompt == "exit":
        exit()
    else:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        print(completion.choices[0].message.content)

