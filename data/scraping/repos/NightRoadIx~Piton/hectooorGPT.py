import openai

# Colocar el secreto (llave)
# 1. Ingresar a la cuenta de OpenAI
# 2. Ingresar al sitio: https://platform.openai.com/account/api-keys
# 3. Dar clic en el botón "Create new secret key"
# 4. Dar nombre al proyecto y dar clic en OK
# 5. Copiar la llave generada
openai.api_key = 'SECRETO'

# La función para pregunta del chatGPT
def hectorGPT(prompt, model='gpt-3.5-turbo'):
    mensaje = [{"role": "user", "content": prompt}]
    respuesta = openai.ChatCompletion.create(
        model=model,
        messages=mensaje,
        temperature=0,
    )
    return respuesta.choices[0].message["content"]

prompt = "Say Hello to my dear friend Hector"
respuesta = hectorGPT(prompt)
print(respuesta)

prompt = "How can you add chatGPT with an API to Python program"
respuesta = hectorGPT(prompt)
print(respuesta)
