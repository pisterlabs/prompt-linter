import pandas as pd  # Biblioteca pandas e a renomeia como 'pd' para facilitar o uso.
import requests  # Usada para fazer solicitações HTTP, como GET ou POST.
import json  # Permite trabalhar com dados no formato JSON.
import openai  # Usada para interagir com o GPT-4 da OpenAI.
import logging  # Usada para registrar informações de log durante a execução do programa.


# Configuração de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Função para buscar um usuário e lidar com erros de conexão
def get_user(id):
    try:
        response = requests.get(f"{SDW2023_API_URL}/users/{id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao buscar usuário {id}: {str(e)}")
        return None


# Leitura do arquivo CSV
ARQUIVO = "C:\\Users\\wendh\\projects\\bootcamp_santander_python\\02_trilha-python\\SDW2023.csv"

# Configuração da API
SDW2023_API_URL = "https://sdw-2023-prd.up.railway.app"

# Leitura do arquivo CSV
df = pd.read_csv(ARQUIVO)

# Extração dos IDs dos usuários
user_ids = df["UserID"].tolist()
logging.info(f"Número total de IDs de usuários: {len(user_ids)}")

# Transformação: Buscar e filtrar usuários válidos
users = [user for id in user_ids if (user := get_user(id)) is not None]
logging.info(f"Número de usuários válidos: {len(users)}")

# Documentação Oficial da API OpenAI: https://platform.openai.com/docs/api-reference/introduction
# Informações sobre o Período Gratuito: https://help.openai.com/en/articles/4936830

# Substitua o texto TODO por sua API Key da OpenAI, ela será salva como uma variável de ambiente
OPENAI_API_KEY = (
    "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"  # virouviral.ofc@gmail.com
)

openai.api_key = OPENAI_API_KEY


# Função para gerar notícias com tratamento de erros
def generate_ai_news(user):
    try:
        prompt = f"Você é um especialista em marketing bancário.\nCrie uma mensagem completa para o usuário '{user['name']}' sobre a importância dos investimentos (máximo de 100 caracteres)"
        completion = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50,
        )
        return completion.choices[0].text.strip()
    except Exception as e:
        logging.error(
            f"Erro ao gerar notícia para o usuário '{user['name']}': {str(e)}"
        )
        return ""


# Loop para gerar notícias e atualizar os usuários
for user in users:
    news = generate_ai_news(user)
    logging.info(f"Notícia gerada para o usuário '{user['name']}'!")

    # Verifica se 'news' não está vazia antes de adicionar ao usuário
    if news:
        if "news" not in user:
            user["news"] = []
        user["news"].append(
            {
                "icon": "https://digitalinnovation.one.github.io/santander-dev-week-2023-api/icons/credit.svg",
                "description": news,
            }
        )

        # Função para atualizar um usuário com tratamento de erros
        def update_user(user):
            try:
                response = requests.put(
                    f"{SDW2023_API_URL}/users/{user['id']}", json=user
                )
                response.raise_for_status()
                return True
            except requests.exceptions.RequestException as e:
                logging.error(f"Erro ao atualizar o usuário '{user['name']}': {str(e)}")
                return False

        success = update_user(user)
        if success:
            logging.info(f"Usuário '{user['name']}' atualizado com sucesso!")
        else:
            logging.warning(f"Falha ao atualizar o usuário '{user['name']}'!")

logging.info("Processo de ETL concluído.")
