import openai

# Establece las credenciales de OpenAI API
openai.api_key = "TU_API_KEY"

# Envia el prompt(mensaje de usuario) y la entrada del usuario a la API
response = openai.Completion.create(
    engine='davinci',
    prompt="Hola.",
    temperature=1.0,
    max_tokens=100,
    stop="\n"
)

# Obtiene la respuesta del chatbot desde la API de OpenAI
chatbot_response = response.choices[0].text.strip()

# Imprime la respuesta del chatbot
print(chatbot_response)