#Fazendo as importações necessárias
import pandas as pd 
import openai
#with open('C:\\Users\\Usuário\\Desktop\\desafio-dio-ETL\\txt\\key.txt') as f:
with open('C:\\Users\\Usuário\\Desktop\\desafio-dio-ETL\\txt\\key.txt', 'r') as file:
    openai_api_key = file.read().strip()  

# Configure a chave da API OpenAI
openai.api_key = openai_api_key

#Extraindo os dados do arquivo CSV de clientes 
df = pd.read_csv(r"C:\Users\Usuário\Desktop\desafio-dio-ETL\csv\clientes.csv")
clientes_id = df['id_cliente'].tolist() 
nome_cliente_antes = df['nome_cliente'].tolist()
numero_cliente = df['numero_telefone'].tolist()
total_consumido = df['total_consumido'].tolist()
produto_mais_comprado = df['produto_mais_comprado'].tolist()
data_ultima_compra = df['data_ultima_compra'].tolist()

#Transformando os dados do nome dos clientes
nome_clientes = [nome_cliente_antes[i].title() for i in range(len(nome_cliente_antes))]
#print(nome_clientes)

# Função para gerar mensagem personalizada
def generate_ai_msg(nome_cliente, produto_mais_comprado, data_ultima_compra):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um especialista em marketing"},
                {"role": "user", "content": f"Crie uma mensagem para {nome_cliente} sobre seu item mais comprado {produto_mais_comprado} suas compras e a data da sua última compra {data_ultima_compra} (máximo 280 caracteres):"},
            ]
        )
        return completion.choices[0].message.content.strip('"')
    except Exception as e:
        print(f"Ocorreu um erro na chamada da API: {e}")
        return ''

# Lista para armazenar as mensagens personalizadas
mensagens_personalizadas = []

# Loop para criar as mensagens personalizadas
for nome_cliente in nome_clientes:
    mensagem_personalizada = generate_ai_msg(nome_cliente, produto_mais_comprado, data_ultima_compra)
    mensagens_personalizadas.append(mensagem_personalizada)

# Colocando os dados em um DataFrame
try:
    df = pd.DataFrame({
        'id_cliente': clientes_id,
        'nome_cliente': nome_clientes,
        'numero_telefone': numero_cliente,
        'total_consumido': total_consumido,
        'produto_mais_comprado': produto_mais_comprado,
        'data_ultima_compra': data_ultima_compra,
        'mensagem_personalizada': mensagens_personalizadas
    })
    df.to_csv(r"C:\Users\Usuário\Desktop\desafio-dio-ETL\csv\clientes_final.csv", index=False)
except Exception as e:
    print(f'Erro na criação do CSV: {e}')








