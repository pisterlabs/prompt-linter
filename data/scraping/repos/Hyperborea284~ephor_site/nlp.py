from summarizer import Summarizer
from spacy import displacy
import spacy
from polyglot.detect import Detector
import subprocess
import os
import re
import pandas as pd
import string
import nltk
from nltk import *
from nltk.tokenize import word_tokenize
from tabulate import tabulate
from operator import itemgetter

from langdetect import detect
#from ftlangdetect import detect

import string
from collections import Counter
from datetime import datetime
from IPython.core.display import HTML

import numpy as np
import openai
import speech_recognition as sr
from dotenv import load_dotenv


from googletrans import Translator
from gtts import gTTS
from llama_index import Document, SimpleDirectoryReader, VectorStoreIndex, LLMPredictor, ServiceContext
from typing import List, Dict, Tuple
import sentencepiece
import codecs





class Func:
    def __init__(self):
        """
        Inicializa a classe Func e limpa a tela do terminal.
        """
        os.system('clear')

    @staticmethod       # Esta função foi substituída por uma varável local salva em ~/.bashrc ; remover e refatorar
    def api_loader():
        """
        Carrega a chave de API do OpenAI a partir do arquivo .env.

        Argumentos:
        None

        Retorna:
        str: chave de API do OpenAI
        """
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None:
            raise ValueError("Did not find openai_api_key, please add an environment variable `OPENAI_API_KEY` which contains it, or pass  `openai_api_key` as a named parameter.")
        return openai_api_key

    @staticmethod
    def import_text_from_folder(folder_path):
        """
        Esta função recebe um caminho de pasta como entrada e retorna uma única string contendo todo o texto dos
        arquivos txt encontrados nessa pasta.
    
        Argumentos:
        folder_path -- O caminho da pasta que contém os arquivos txt a serem importados
    
        Retorna:
        text -- A única string contendo todo o texto dos arquivos txt encontrados na pasta especificada por folder_path
    
        Raises:
        FileNotFoundError -- Se a pasta especificada por folder_path não existir
    
        Esta função percorre todos os arquivos txt encontrados na pasta especificada por folder_path, lê o conteúdo
        de cada um deles e adiciona seu texto a uma única string. A string contendo todo o texto é então retornada
        pela função.
    
        Se a pasta especificada por folder_path não existir, a função levanta a exceção FileNotFoundError.
        """

        # Verifica se a pasta existe
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"A pasta {folder_path} não existe.")

        # Cria uma lista vazia para armazenar o texto de todos os arquivos
        text_list = []

        # Percorre todos os arquivos da pasta especificada por folder_path
        for file_name in os.listdir(folder_path):
            # Verifica se o arquivo é um arquivo txt
            if file_name.endswith('.txt'):
                # Abre o arquivo em modo de leitura ('r') e armazena seu texto na lista text_list
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    text_list.append(file.read())

        # Combina todo o texto da lista text_list em uma única string
        text = ''.join(text_list)
        return text

    @staticmethod
    def convert_files_to_utf8(folder_path):
        """
        Converte os arquivos em uma pasta para o formato de codificação UTF-8.
    
        :param folder_path: O caminho da pasta contendo os arquivos.
        :type folder_path: str
        """
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
    
            # Verifica se o caminho é um arquivo
            if os.path.isfile(file_path):
                # Obtém a extensão do arquivo
                file_extension = os.path.splitext(file_name)[1]
    
                # Processa apenas arquivos não-texto
                if file_extension != '.txt':
                    # Lê o arquivo com sua codificação original
                    with open(file_path, 'rb') as file:
                        content = file.read()
    
                    try:
                        # Tenta decodificar o conteúdo como UTF-8
                        decoded_content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        # Lida com arquivos que não podem ser decodificados como UTF-8
                        print(f"Arquivo '{file_name}' ignorado devido a erro de decodificação.")
                        continue
    
                    # Cria um novo caminho de arquivo com a extensão '.txt'
                    new_file_path = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}.txt")
    
                    # Salva o conteúdo como um arquivo de texto codificado em UTF-8
                    with codecs.open(new_file_path, 'w', 'utf-8') as new_file:
                        new_file.write(decoded_content)
    
                    print(f"Arquivo '{file_name}' convertido para UTF-8 e salvo como '{new_file_path}'.")

    @staticmethod
    def criar_indice_sql(host, user, password, database_sql, tabela):
        """
        Cria um índice de pesquisa usando os dados de uma tabela específica em um banco de dados SQL.
        Argumentos:
        host (str): endereço do servidor de banco de dados
        user (str): nome de usuário do banco de dados
        password (str): senha do usuário do banco de dados
        database_sql (str): nome do banco de dados a ser conectado
        tabela (str): nome da tabela que contém os dados a serem indexados
        
        Retorna:
        None
        """
        # Conectar ao banco de dados
        conn = pymysql.connect(host=host, user=user, password=password, database=database_sql)
        cursor = conn.cursor()

        try:
            # Selecionar os dados a serem indexados
            cursor.execute(f"SELECT id_sql, texto FROM {tabela}")

            # Criar os documentos e adicionar ao índice
            indice = GPTSimpleVectorIndex()
            for id_sql, texto in cursor:
                documento = Document(texto, doc_id=str(id_sql))
                indice.add_document(documento)

            # Salvar o índice em disco
            reader = SimpleDirectoryReader("indice")
            indice.write(reader)

        except Exception as e:
            print(f"Error: {e}")
            conn.rollback()
        else:
            conn.commit()

        finally:
            # Fechar a conexão com o banco de dados
            cursor.close()
            conn.close()

        return None

    @staticmethod
    def llama_index_texts(folder_path):
        """
        Cria um índice de vetores a partir dos textos presentes em uma pasta.
    
        :param folder_path: O caminho da pasta contendo os textos.
        :type folder_path: str
        :return: O índice de vetores criado a partir dos textos.
        :rtype: VectorStoreIndex
        """
        documents = SimpleDirectoryReader(folder_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index
    
    @staticmethod
    def add_to_index(index, file_path):
        """
        Esta função adiciona um documento de texto contido em um arquivo ao índice Llama existente.
        
        Argumentos:
        index -- O índice Llama atual.
        file_path -- Caminho absoluto ou relativo do arquivo contendo o documento a ser adicionado ao índice.
        
        Retorna:
        index -- O índice Llama atualizado com o novo documento de texto adicionado.
        
        Esta função carrega o documento de texto do arquivo especificado e o adiciona ao índice Llama existente.
        O índice é atualizado com o novo documento de texto adicionado e a função retorna o índice atualizado.
        """
        # Carregando documento a partir do arquivo
        with open(file_path, 'r') as f:
            text = f.read()
        doc = Document(file_path, text)

        # Adicionando documento ao índice Llama existente
        index.add_documents([doc])

        # Retornando o índice atualizado
        return index

    @staticmethod
    def saver(index, path_folder):
        """
        Salva o índice do Llama em disco como um arquivo json.
    
        Args:
            index: O índice do Llama a ser salvo.
            path_folder: O caminho para o diretório onde o arquivo json será salvo.
    
        Returns:
            Nada. A função salva o índice do Llama como um arquivo json.
        """
        # Salva o índice em um arquivo json
        index.save_to_disk(f'{path_folder}/index.json')

    @staticmethod
    def loader(path_folder):
        """
        Carrega o índice do Llama a partir de um arquivo json em disco.
    
        Args:
            path_folder: O caminho para o diretório onde o arquivo json do índice está salvo.
    
        Returns:
            O índice do Llama carregado a partir do arquivo json.
        """
        # Carrega o índice a partir de um arquivo json
        index = GPTSimpleVectorIndex.load_from_disk(f'{path_folder}/index.json')
        return index

    @staticmethod
    def response(index, question):
        """
        Realiza uma consulta ao índice de vetores com uma determinada pergunta e retorna a resposta.
    
        :param index: O índice de vetores utilizado para a consulta.
        :type index: VectorStoreIndex
        :param question: A pergunta a ser consultada.
        :type question: str
        :return: A resposta obtida da consulta.
        :rtype: str
        """
        # Cria um mecanismo de consulta
        os.environ['OPENAI_API_KEY'] = Func.api_loader()
        query_engine = index.as_query_engine()
    
        # Realiza a consulta
        response = query_engine.query(question)
        print(response)
        return response

    @staticmethod
    def summarize_text_simple(text):
        """
        Função que usa a API gratuita do ChatGPT para resumir um texto.
    
        Args:
        text (str): O texto a ser resumido.
    
        Returns:
        O resumo do texto, gerado pela API do ChatGPT.
        """
        # Configura o modelo e o prompt de entrada para a API
        model_engine = "text-davinci-002"
        prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"

        # Gera a saída usando a API do ChatGPT
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # Extrai o resumo da resposta da API
        summary = response.choices[0].text.strip()

        return summary

    @staticmethod
    def summarize_text_full(text, engine="text-davinci-002", prompt_prefix="Please summarize the following text:\n\n",
                            prompt_suffix="\n\nSummary:", max_tokens=1024, n=1, stop=None, temperature=0.5):
        """
        Função que usa a API gratuita do ChatGPT para resumir um texto.
    
        Args:
        text (str): O texto a ser resumido.
        engine (str): O mecanismo do modelo a ser usado na API do ChatGPT (padrão: "text-davinci-002").
        prompt_prefix (str): O prefixo do prompt de entrada para a API (padrão: "Please summarize the following text:\n\n").
        prompt_suffix (str): O sufixo do prompt de entrada para a API (padrão: "\n\nSummary:").
        max_tokens (int): O número máximo de tokens permitidos na resposta (padrão: 1024).
        n (int): O número de respostas a serem geradas pela API (padrão: 1).
        stop (str or List[str]): Uma string ou lista de strings que indica onde parar a geração da resposta (padrão: None).
        temperature (float): Controla a aleatoriedade da geração de texto pela API (padrão: 0.5).
    
        Returns:
        O resumo do texto, gerado pela API do ChatGPT.
        """

        # Monta o prompt de entrada para a API, com o texto fornecido
        prompt = f"{prompt_prefix}{text}{prompt_suffix}"

        # Gera a saída usando a API do ChatGPT
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
        )

        # Extrai o resumo da resposta da API
        summary = response.choices[0].text.strip()

        return summary, response

    @staticmethod
    def cleaner(text, lang_code_short):
        """
        Recebe um texto e um código de idioma, realiza a limpeza do texto e retorna trigramas, bigramas,
        lista de palavras filtradas, lista de hapaxes e uma segunda versão da lista filtrada sem hapaxes.

        Args:
            text (str): Texto a ser limpo.
            lang_code_short (str): Código de idioma de duas letras.

        Returns:
            tuple: Uma tupla contendo uma lista de trigramas, uma lista de bigramas, uma lista de palavras
            filtradas, uma lista de hapaxes e uma lista de palavras filtradas sem hapaxes.
        """
        # Tokenização do texto
        text_tokens = nltk.tokenize.word_tokenize(text, language=lang_code_short)

        # Definição das stopwords
        stopwords = set(nltk.corpus.stopwords.words(lang_code_short))

        # Definição dos caracteres especiais que serão removidos
        special_chars = string.punctuation + '`”\'\'``). .). .: .) ]: [: :] :[ ©\'–// /\'- ! ? ., . “'

        # Remoção das stopwords e dos caracteres especiais
        filtered_word_0 = [word.lower() for word in text_tokens if
                           word.lower() not in stopwords and word.lower() not in special_chars]

        # Remoção dos número
        filtered_word_1 = [re.sub('[0-9]', '', word) for word in filtered_word_0]

        # Remoção de palavras com menos de dois caracteres
        filtered_word = [word for word in filtered_word_1 if len(word) >= 2]

        # Criação das listas de trigramas e bigramas
        output_tri = list(nltk.trigrams(filtered_word))
        output_bi = list(nltk.bigrams(filtered_word))

        # Criação das listas de palavras únicas e palavras sem hapaxes
        word_counts = Counter(filtered_word)
        hapaxes = [word for word in word_counts if word_counts[word] == 1]
        filtered_word_no_hap = [word for word in filtered_word if word not in hapaxes]

        return output_tri, output_bi, filtered_word, hapaxes, filtered_word_no_hap

    @staticmethod
    def entities(texto):
        """
        Função que extrai entidades nomeadas de um texto e identifica o idioma do texto.

        Args:
        texto (str): O texto para extrair entidades nomeadas e identificar idioma.

        Returns:
        tuple: Uma tupla contendo uma lista de entidades nomeadas, o código ISO do idioma, 
        o nome do idioma em sua forma reduzida e o nome do modelo do Spacy usado para processar o texto.
        """
        # Remover quebras de linha do texto
        text = re.sub(r'\n', '', texto)
        texto = texto.replace('\n', ' ')

        # Detectar idioma do texto
        lang_code = detect(text=text)

        # Identificar nome do idioma em sua forma reduzida e nome do modelo Spacy correspondente
        if lang_code == 'en':
            lang_code_short = 'english'
            lang_code_full = 'en_core_web_sm'
        elif lang_code == 'pt':
            lang_code_short = 'portuguese'
            lang_code_full = 'pt_core_news_sm'
        else:
            lang_code_short = lang_code
            lang_code_full = ''

        # Extrair entidades nomeadas do texto usando modelo Spacy e adicioná-las à lista
        ent_list = []
        if lang_code_full:
            pln = spacy.load(lang_code_full)
            documento = pln(texto)
            for entidade in documento.ents:
                if [entidade.text, entidade.label_] not in ent_list:
                    ent_list.append([entidade.text, entidade.label_])

        # Retornar lista de entidades nomeadas, código ISO do idioma, nome do idioma em sua forma reduzida
        # e nome do modelo do Spacy usado para processar o texto
        return list(ent_list), lang_code, lang_code_short, lang_code_full

    @staticmethod
    def calcula_ome(texto: str) -> Tuple[List[str], Dict[str, float]]:
        """Calcula a ordem média de evocação (OME) para cada palavra em um texto e retorna as
        palavras ordenadas pela OME, do maior para o menor, juntamente com um dicionário
        contendo as OMEs calculadas para cada palavra.

        Argumentos:
        texto -- O texto a ser analisado.

        Retorna:
        Uma tupla contendo uma lista de strings com as palavras ordenadas pela OME e um
        dicionário onde cada chave é uma palavra do texto e cada valor é a OME calculada para essa palavra.
        """
        # Quebra o texto em uma lista de palavras
        palavras = texto.split()

        # Calcular a frequência de cada palavra
        frequencias = Counter(palavras)

        # Calcular a ordem de cada palavra na lista
        ordens = {}
        for i, palavra in enumerate(palavras):
            if palavra not in ordens:
                ordens[palavra] = []
            ordens[palavra].append(i + 1)  # adiciona a posição da palavra na lista

        # Calcular a OME para cada palavra
        ome = {}
        for palavra in frequencias:
            # soma as ordens de cada ocorrência da palavra na lista
            soma_ordens = sum([ordem for ordem in ordens[palavra]])
            # calcula a OME da palavra dividindo a soma das ordens pelo número de ocorrências
            ome_palavra = soma_ordens / frequencias[palavra]
            # adiciona a OME calculada ao dicionário ome, com a palavra como chave
            ome[palavra] = ome_palavra

        # Ordenar as palavras pela OME (maior OME primeiro)
        palavras_ordenadas = sorted(ome, key=ome.get, reverse=True)

        # retorna as palavras ordenadas pela OME e o dicionário com as OMEs calculadas para cada palavra
        return palavras_ordenadas, ome

    @staticmethod
    def analise_prototipica(palavras_ordenadas: list, ome: dict) -> Tuple[
        List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        A função analise_prototipica analisa a distribuição das palavras em um texto de acordo com a teoria do
        protótipo, que divide as palavras em quatro categorias: núcleo central, zona periférica 1, zona periférica 2
        e zona periférica 3. A função recebe uma lista de palavras ordenadas pela OME (do maior para o menor) e um
        dicionário contendo as OMEs calculadas para cada palavra. Retorna uma tupla contendo as quatro listas de
        acordo com a análise prototípica.
        """

        # Verifica se as entradas são válidas
        if not isinstance(palavras_ordenadas, list) or not isinstance(ome, dict):
            raise TypeError("palavras_ordenadas deve ser uma lista e ome deve ser um dicionário")

        # Verifica se as entradas têm o mesmo tamanho
        if len(palavras_ordenadas) != len(ome):
            raise ValueError("palavras_ordenadas e ome devem ter o mesmo tamanho")

        # Cria uma lista de tuplas contendo as palavras e suas ome
        palavras_omes = [(palavra, ome.get(palavra, 0)) for palavra in palavras_ordenadas]

        # Obtém a média da OME para a normalização
        media_ome = np.mean(list(ome.values()))

        # Cria as quatro listas de acordo com a análise prototípica
        nucleo_central = [(palavra, ome / media_ome) for palavra, ome in palavras_omes if ome > media_ome]

        zona_periferica_1 = []
        zona_periferica_2 = []
        zona_periferica_3 = []

        for i, (palavra, ome) in enumerate(palavras_omes):
            ome_norm = ome / media_ome
            if ome_norm <= 1:
                if i < len(palavras_omes) * 0.25:
                    zona_periferica_1.append((palavra, ome_norm))
                elif i >= len(palavras_omes) * 0.25 and i < len(palavras_omes) * 0.75:
                    zona_periferica_2.append((palavra, ome_norm))
                    # Adiciona a palavra em zona_periferica_1 se já não estiver presente
                    if palavra not in [p[0] for p in zona_periferica_1]:
                        zona_periferica_1.append((palavra, ome_norm))
                else:
                    zona_periferica_3.append((palavra, ome_norm))
                    # Adiciona a palavra em zona_periferica_1 ou zona_periferica_2 se já não estiver presente
                    if palavra not in [p[0] for p in zona_periferica_1]:
                        if palavra not in [p[0] for p in zona_periferica_2]:
                            zona_periferica_1.append((palavra, ome_norm))
                        else:
                            zona_periferica_2.append((palavra, ome_norm))
                    elif palavra not in [p[0] for p in zona_periferica_2]:
                        zona_periferica_2.append((palavra, ome_norm))

        # Retorna as quatro listas de acordo com a análise prototípica
        return nucleo_central, zona_periferica_1, zona_periferica_2, zona_periferica_3

    @staticmethod
    def teste(path_folder, hap_or_no_hap):
        """Executa o teste de análise prototípica no corpus especificado.
    
        Args:
            path_folder: O caminho da pasta que contém o corpus a ser utilizado.
            hap_or_no_hap: Um parâmetro que determina se a análise prototípica deve ser realizada com palavras hapax
            ou sem palavras hapax.
    
        Returns:
            Uma tupla contendo todos os resultados relevantes da análise.
    
        Raises:
            ValueError: Se hap_or_no_hap não for 'hap' ou 'no_hap'.
        """
        # Importa o texto do arquivo especificado
        texto = Func.import_text_from_folder(f"{path_folder}")

        # Identifica as entidades do texto e seu idioma
        ent_list, lang_code, lang_code_short, lang_code_full = Func.entities(texto)

        # Limpa o texto e obtém as listas filtradas de palavras
        output_tri, output_bi, filtered_word, hapaxes, filtered_word_no_hap = Func.cleaner(texto, lang_code_short)

        # Calcula a OME das palavras filtradas de acordo com o parâmetro hap_or_no_hap
        if hap_or_no_hap == 'hap':
            palavras_ordenadas, omes = Func.calcula_ome(' '.join(filtered_word))
        elif hap_or_no_hap == 'no_hap':
            palavras_ordenadas, omes = Func.calcula_ome(' '.join(filtered_word_no_hap))
        else:
            raise ValueError("hap_or_no_hap precisa ser 'hap' ou 'no_hap'")

        # Realiza a análise prototípica das palavras filtradas
        nucleo_central, zona_periferica_1, zona_periferica_2, zona_periferica_3 = Func.analise_prototipica(
            palavras_ordenadas, omes)

        # Formata os dados da análise para exibição em tabela
        data = [["Núcleo Central", len(nucleo_central), "Zona Periférica 1", len(zona_periferica_1),
                 "Zona Periférica 2", len(zona_periferica_2), "Zona Periférica 3", len(zona_periferica_3)]]

        # Imprime a tabela dos resultados da análise
        print(tabulate(data, headers=["", "Nº de Palavras", "", "Nº de Palavras", "", "Nº de Palavras", "",
                                      "Nº de Palavras"]))

        # Retorna todos os resultados relevantes
        return texto, ent_list, lang_code, lang_code_short, lang_code_full, output_tri, data, \
            output_bi, filtered_word, hapaxes, filtered_word_no_hap, palavras_ordenadas, omes, \
            nucleo_central, zona_periferica_1, zona_periferica_2, zona_periferica_3

    @staticmethod
    def translate_text(text, target_language):
        """
        Função para traduzir um texto para outro idioma usando a biblioteca googletrans.
    
        Args:
        text (str): O texto a ser traduzido.
        target_language (str): O código ISO do idioma de destino para a tradução, como 'pt' para português.
    
        Returns:
        O texto traduzido para o idioma de destino.
        """
        translator = Translator()
        translated_text = translator.translate(text, dest=target_language)
        return translated_text.text

    @staticmethod
    def text_to_audio(text: str, save_file: bool = False) -> None:
        """
        Função que gera um arquivo de áudio com um texto fornecido e reproduz o áudio.
    
        Args: text (str): Texto a ser convertido em áudio. save_file (bool): Indica se o arquivo de áudio deve ser 
        salvo. Se `False`, o arquivo é salvo como "audio.mp3". Se `True`, o arquivo é salvo com o nome no formato 
        "YYYY-MM-DD_HH-MM-SS.mp3".
        """
        # Gerar arquivo de áudio com o texto fornecido
        tts = gTTS(text=text, lang='pt-br')
        audio_file = "audio.mp3" if not save_file else datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp3"
        tts.save(audio_file)

        # Reproduzir áudio
        os.system(f"start {audio_file}")

    @staticmethod
    def audio_to_text(audio_file):
        """
        Função que recebe um arquivo de áudio e retorna seu conteúdo como texto.

        Args:
        audio_file (str): O caminho para o arquivo de áudio.

        Returns:
        str: O conteúdo do áudio como texto.
        """

        # Inicializar o reconhecedor de fala
        r = sr.Recognizer()

        # Ler arquivo de áudio
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)

        # Reconhecer texto do áudio usando o serviço da Google
        text = r.recognize_google(audio_data, language='pt-BR')

        # Retorna o texto reconhecido
        return text

    @staticmethod
    def bert_sumarizar_text(text, num):
        # Cria uma instância de Summarizer
        sumarizador = Summarizer()

        # Chama o método summarize da instância de Summarizer, passando o texto e o número de sentenças a serem 
        # resumidas
        resumo = sumarizador(text, num_sentences=num)

        # Retorna o resumo gerado pelo Summarizer
        return resumo

    @staticmethod
    def visualiza_resumo(titulo, lista_sentencas, melhores_sentencas):
        # Cria uma string vazia
        texto = ""

        # Cria um título para o resumo do texto com base no parâmetro "titulo" recebido pela função
        display(HTML(f'<h1>Resumo do texto - {titulo}</h1>'))

        # Itera sobre a lista de sentenças recebida pela função
        for i in lista_sentencas:
            # Se a sentença atual estiver entre as melhores sentenças, adiciona uma marcação de destaque HTML
            if i in melhores_sentencas:
                texto += str(i).replace(i, f"<mark>{i}</mark>")
            else:
                texto += i
        # Exibe o resumo do texto com as sentenças marcadas em destaque (se aplicável)
        display(HTML(f""" {texto} """))

    @staticmethod
    def idiom(form, name):
        with open(f"{name}.txt", "w") as text_file:
            text_file.write(f"{form.cleaned_data['content']}\n\n")
            text_file.close()
            content = form.cleaned_data['content'].replace('\n', ' ')
            ent_list, lang_code, lang_code_short, lang_code_full = Func.entities(content)
            barplot_size = form.cleaned_data["barplot_size"]
            plot_min_freq = form.cleaned_data["plot_min_freq"]
            reinert_segm = form.cleaned_data["reinert_segm"]
            reinert_min_freq = form.cleaned_data["reinert_min_freq"]
            reinert_k = form.cleaned_data["reinert_k"]
            reinert_min_seg = form.cleaned_data["reinert_min_seg"]
    
            subprocess.call (f"/usr/bin/Rscript nlp.R {name}.txt {lang_code_short} {barplot_size} {plot_min_freq} {reinert_segm} {lang_code} {reinert_min_freq} {reinert_k} {reinert_min_seg}", shell=True)
        os.remove(f"{name}.txt")  

    @staticmethod
    def bert_sumarizar(form):
        sumarizador = Summarizer()
        resumo = sumarizador(form.cleaned_data['content'])
        return resumo
    
    @staticmethod
    def cleaner(text, lang_code_short):
        text_0 = word_tokenize(text)
        stopwords = set(nltk.corpus.stopwords.words(lang_code_short))
        string_better = string.punctuation + '`' + '”' + "'" + " '" + "''" + "``" + ")." + ".)." + ".:" + ".)" \
                                           + "]:" + "[:" + ":]" + ":[" + '©' + "'–" + '//' + '/' + "'-" + "!" \
                                           + "?" + ".," + "." + "," + "“"
    
        filtered_word_0 = [word.lower() for word in text_0 if (word.lower() not in stopwords) and (word.lower() not in string_better)]
        filtered_word_1 = [re.sub('[0-9]', '', i) for i in filtered_word_0]
    
        for i in filtered_word_1:
            if(len(i) >= 2):
                continue
            elif(len(i) == 0):
                filtered_word_1.remove(i)
            else:
                filtered_word_1.remove(i)
    
        output_bi = list(nltk.bigrams(filtered_word_1))
        output_tri = list(nltk.trigrams(filtered_word_1))
    
        return output_tri, output_bi, filtered_word_1    

    @staticmethod
    def proto(filtered_word_1):
        # constrói dicionário beta contendo o termo e a frequencia
        beta = {}
        for i in filtered_word_1:
            if i not in beta:
                beta[i] = (1)
            else:
                beta[i] += (1)
        
        # constrói dicionário charlie contendo o termo e a ordenação
        charlie = {}
        for i in filtered_word_1:
            charlie[i] = [o for o, x in enumerate(filtered_word_1) if x == i]
        
        delta = sorted(beta.items(), key=lambda item: item[1], reverse=True)
        echo = sorted(charlie.items(), key=lambda item: item[1], reverse=True)
        
        alt_freq_bai_ord = (pd.concat([pd.Series(dict(delta)),pd.Series(dict(echo))],axis=1).reset_index().values.tolist())
    
        for i in range(len(alt_freq_bai_ord)):
            ome = sum(alt_freq_bai_ord[i][2])/(len(alt_freq_bai_ord[i][2]))
            alt_freq_bai_ord[i].insert(3, ome)
    
        bai_freq_alt_ord = sorted(sorted(alt_freq_bai_ord, key=lambda x: x[2][0], reverse=True), key = lambda x: x[1])
        alt_freq_alt_ord = sorted(sorted(alt_freq_bai_ord, key=lambda x: x[2][0], reverse=True), key = lambda x: x[1], reverse=True)
        bai_freq_bai_ord = sorted(sorted(alt_freq_bai_ord, key=lambda x: x[2][0]), key = lambda x: x[1])
    
        return alt_freq_bai_ord, bai_freq_alt_ord, alt_freq_alt_ord, bai_freq_bai_ord
    
