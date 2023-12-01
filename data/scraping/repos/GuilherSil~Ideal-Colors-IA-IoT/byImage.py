import cv2
import numpy as np
import openai

# Configure sua chave de API da OpenAI
openai.api_key = "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"

# Função para obter uma resposta do chat GPT
def obter_resposta_gpt(pergunta):
    resposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Suponha que você é um instrutor de Python."},
            {"role": "user", "content": pergunta},
        ],
        temperature=0.7,
    )
    return resposta['choices'][0]['message']['content']

# Tons de pele em dicionario conforme valores utilizados na escala de Taylor Hyperpigmentation
dicionario_cores = {
    'deep_1': np.array([75, 57, 50], dtype=np.uint8),
    'deep_2': np.array([90, 69, 60], dtype=np.uint8),
    'deep_3': np.array([105, 80, 70], dtype=np.uint8),
    'medium_deep_1': np.array([120, 92, 80], dtype=np.uint8),
    'medium_deep_2': np.array([135, 103, 90], dtype=np.uint8),
    'medium_deep_3': np.array([150, 114, 100], dtype=np.uint8),
    'medium_1': np.array([165, 126, 110], dtype=np.uint8),
    'medium_2': np.array([180, 138, 120], dtype=np.uint8),
    'medium_3': np.array([195, 149, 130], dtype=np.uint8),
    'light_medium_1': np.array([210, 161, 140], dtype=np.uint8),
    'light_medium_2': np.array([225, 172, 150], dtype=np.uint8),
    'light_medium_3': np.array([240, 184, 160], dtype=np.uint8),
    'light_1': np.array([255, 195, 170], dtype=np.uint8),
    'light_2': np.array([255, 206, 180], dtype=np.uint8),
    'light_3': np.array([255, 218, 190], dtype=np.uint8)
}

# Função para determinar o tom de pele
def encontrar_tom_de_pele(imagem):
    # Converte a imagem para o espaço de cor HSV
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Define a faixa de cores de pele no espaço HSV
    cor_min = np.array([0, 48, 80], dtype=np.uint8)
    cor_max = np.array([20, 255, 255], dtype=np.uint8)

    # Filtra a imagem para encontrar pixels de cor de pele
    mascara = cv2.inRange(imagem_hsv, cor_min, cor_max)

    # Calcula a média dos valores de pixel da região de pele
    media_pele = cv2.mean(imagem, mask=mascara)[:3]

    # Calcula a distância euclidiana entre a média da pele e as cores do dicionário
    distancias = [np.linalg.norm(media_pele - cor) for cor in dicionario_cores.values()]

    # Encontra o índice da cor mais próxima
    indice_tom_pele = np.argmin(distancias)

    # Obtém o valor RGB do tom de pele correspondente
    valor_rgb = list(dicionario_cores.values())[indice_tom_pele]

    return valor_rgb

# Carregar imagem
imagem = cv2.imread('teste_3.jpeg')

# Converter a imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Carregar classificador Haar Cascade para detecção de faces
classificador_rosto = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detectar rostos na imagem
rostos = classificador_rosto.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Processar cada rosto encontrado
for (x, y, w, h) in rostos:
    # Desenhar retângulo ao redor do rosto
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibir imagem com os rostos detectados
cv2.imshow('Rostos detectados', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Encontra o tom de pele da região de pele
valor_rgb_tom_pele = encontrar_tom_de_pele(imagem)

print('Valor RGB do tom de pele:', valor_rgb_tom_pele)


pergunta = f"Com base no tom de pele {valor_rgb_tom_pele}, responda com a seguinte estrutura:" \
            \
            "Tom de pele: [nome do tom de pele]" \
            "Com base na estação atual [estação atual do brasil], as cores que podem melhor funcionar bem em uma paleta de cores são: "
           


resposta_gpt = obter_resposta_gpt(pergunta)

print("Resposta do GPT:", resposta_gpt)