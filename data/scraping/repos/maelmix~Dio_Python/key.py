import openai

# Substitua 'sua-api-key-aqui' pela sua chave da API
openai.api_key = 'api'

def chat_with_gpt4(prompt):
    try:
        # Inicializando o histórico da conversa
        conversation_history = []

        while True:
            # Obtendo a entrada do usuário
            user_input = input("Você: ")

            # Atualizando o histórico da conversa
            conversation_history.append(f"Você: {user_input}")
            
            # Combinando o histórico da conversa com a entrada do usuário para criar o prompt
            gpt4_prompt = "\n".join(conversation_history)
            
            # Enviando o prompt para a API do OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Assumindo que gpt-4 é o nome do modelo, substitua se necessário
                messages=[
                    {"role": "system", "content": "Você é um assistente virtual."},
                    {"role": "user", "content": gpt4_prompt}
                ]
            )
            
            # Extraindo a resposta do modelo
            gpt4_response = response['choices'][0]['message']['content'].strip()
            
            # Atualizando o histórico da conversa
            conversation_history.append(f"GPT-4: {gpt4_response}")
            
            # Exibindo a resposta do modelo
            print(f"GPT-4: {gpt4_response}")

    except Exception as e:
        print(f"Erro: {e}")

# Inicializando o chatbot
chat_with_gpt4("Olá!")







