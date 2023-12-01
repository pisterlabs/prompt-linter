"""
Este archivo contiene clases y funciones para cargar, procesar documentos de texto, vectorizarlos y construir respuestas en base a ellos.
Fue programado teniendo en mente su utilización en aplicaciones de IA.

- Patrick Vásquez <pvasquezs@fen.uchile.cl>
  Ultima actualización: 23/10/2023
"""

from abc import ABC, abstractmethod

# Text loaders

class ExtractDocumentFormat:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_extension(file_path):
        """
        Extrae la extensión del archivo.
        :return: La extensión del archivo.
        """
        import os
        name, extension = os.path.splitext(file_path)
        return extension
    
class TextDocumentLoader(ABC):
    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def load_document(self):
        """
        Carga un documento de texto.
        Debe ser implementado por las subclases con los métodos para cada extensión.
        """
        pass

class PdfLoader(TextDocumentLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def load_document(file_path):
        """
        Carga un documento PDF.
        :param file_path: La ruta del archivo PDF.
        :return: Los datos cargados desde el PDF.
        """
        from langchain.document_loaders import PyPDFLoader
        pdfloader = PyPDFLoader(file_path)
        print(f"Loading a PDF document from {file_path}")
        data = pdfloader.load(file_path)
        return data

class DocxLoader(TextDocumentLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def load_document(file_path):
        """
        Carga un documento de texto (DOCX).
        :param file_path: La ruta del archivo DOCX.
        :return: Los datos cargados desde el DOCX.
        """
        from langchain.document_loaders import Docx2txtLoader
        docxloader = Docx2txtLoader(file_path)
        print(f"Loading a DOCX document from {file_path}")
        data = docxloader.load(file_path)
        return data

class TxtLoader(TextDocumentLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def load_document(file_path):
        """
        Carga un documento de texto (TXT).
        :param file_path: La ruta del archivo TXT.
        :return: Los datos cargados desde el TXT.
        """
        from langchain.document_loaders import TextLoader
        textloader = TextLoader(file_path)
        print(f"Loading a TXT document from {file_path}")
        data = textloader.load()
        return data

class LoadDocument:
    def __init__(self, file_path, file_extension):
        self.file_path = file_path
        self.file_extension = file_extension

    def load_document(file_path, file_extension):
        """
        Crea y carga un documento basado en su extensión.
        :param file_path: La ruta del archivo.
        :param file_extension: La extensión del archivo.
        :return: Los datos cargados desde el archivo.
        :raises: Exception si la extensión del archivo no es compatible.
        """
        if file_extension == ".pdf":
            data = PdfLoader.load_document(file_path)
        elif file_extension == ".docx":
            data = DocxLoader.load_document(file_path)
        elif file_extension == ".txt":
            data = TxtLoader.load_document(file_path)
        else:
            raise Exception("File extension not supported")
        return data
    
# Text processors

class ChunkData:
    def __init__(self, data, chunk_size, chunk_overlap):
        self.data = data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_data(data, chunk_size=256, chunk_overlap=20):
        """
        Divide los datos de entrada en fragmentos de tamaño fijo con un solapamiento especificado.
        
        Args:
        data (str): Los datos de entrada que se dividirán en fragmentos.
        chunk_size (int): El tamaño de cada fragmento.
        chunk_overlap (int): El solapamiento entre fragmentos adyacentes.
        
        Returns:
        list: Una lista de fragmentos de texto.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(data)
        return chunks

class TextPreprocessor:
    def __init__(self, data, chunk_size=256, chunk_overlap=20):
        self.data = data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def preprocess_data(data, chunk_size, chunk_overlap):
        """
        Preprocesa los datos de entrada.
        
        Args:
        data (str): Los datos de entrada que se preprocesarán.
        chunk_size (int): El tamaño de cada fragmento.
        chunk_overlap (int): El solapamiento entre fragmentos adyacentes.
        
        Returns:
        list: Una lista de fragmentos de texto preprocesados.
        """
        from langchain.text_preprocessor import TextPreprocessor
        text_preprocessor = TextPreprocessor()
        preprocessed_data = text_preprocessor.preprocess(data)
        return preprocessed_data
    
class CalculateEmbeddingCost:
    def __init__(self, chunks):
        self.chunks = chunks

    def __call__(self, chunks):
        """
        Calcula el costo de embedding para un conjunto de textos utilizando el modelo 'text-embedding-ada-002'.
        
        Args:
        chunk: lista de objetos que contienen el contenido de las páginas a ser procesadas.
        
        Returns:
        total_tokens: número total de tokens en el conjunto de textos.
        embedding_cost: costo de embedding en dólares estadounidenses.
        """
        
        import tiktoken
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
        embedding_cost = (total_tokens / 1000 * 0.0001)
        print(f'Costo de embedding, ada-002: ${embedding_cost:.4f}')
        return total_tokens, embedding_cost

# Text vectorizers

class TextVectorizer(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def create_embedding_model(self):
        """
        Vectoriza los datos de entrada.
        Debe ser implementado por las subclases con los métodos para cada modelo.
        """
        pass

class OpenAIEmbeddings(TextVectorizer):
    def __init__(self, data):
        super().__init__(data)
    
    def create_embedding_model():
        from langchain.embeddings.openai import OpenAIEmbeddings
        from dotenv import load_dotenv
        import os
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        return embedding_model

# Vector stores

class VectorStore(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def create_vector_store():
        """
        Crea un vector store.
        Debe ser implementado por las subclases con los métodos para cada modelo.
        """
        pass

class VectorStoreChroma(VectorStore):
    def __init__(self, data):
        super().__init__(data)
    
    def create_vector_store(chunks, embedding_model, persist_directory='db'):
        """
        Crea un vector store para un conjunto de textos usando Chroma.
        
        Args:
        chunks: lista de objetos que contienen el contenido de las páginas a ser procesadas.
        embeddings: objeto que contiene el modelo de embeddings de los textos.
        persist_directory: la carpeta donde se guarda el vector db.
        
        Returns:
        vector_store: vector store para el conjunto de textos.
        """
        from langchain.vectorstores import Chroma
        vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory)
        vector_store.persist()
        return vector_store
    
# Chat models

class ChatModel(ABC):

    @abstractmethod
    def create_chat_model():
        """
        Crea un modelo de chat.
        Debe ser implementado por las subclases con los métodos para cada modelo.
        """
        pass

class OpenAIChat(ChatModel):
    def __init__(self, data):
        self.data = data
    
    def create_chat_model(model='gpt-3.5-turbo', system_message=None, temperature=1):
        """
        Crea un modelo de chat utilizando la biblioteca langchain. El modelo se entrena con el modelo especificado y utiliza la API de OpenAI para generar respuestas. La temperatura controla la creatividad de las respuestas generadas. Los mensajes del sistema y del usuario se proporcionan como entrada para el modelo.

        Args:
            model (str, optional): El modelo que se utilizará para entrenar el modelo de chat. Por defecto, se utiliza 'gpt-3.5-turbo'.
            temperature (int, optional): La temperatura que controla la creatividad de las respuestas generadas. Por defecto, se utiliza 1.
            system_message (str, optional): El mensaje del sistema que se proporciona como entrada para el modelo. Por defecto, es None.
            human_message (str, optional): El mensaje del usuario que se proporciona como entrada para el modelo. Por defecto, es None.

        Returns:
            None
        """
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import(SystemMessage)
        from dotenv import load_dotenv
        import os
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        llm = ChatOpenAI(model='gpt-3.5-turbo',
                        temperature=temperature,
                        openai_api_key=api_key)
        llm(messages = [
            SystemMessage(content=system_message)
            ])
        return llm
    
# QA models

class StandardQA(ABC):

    @abstractmethod
    def ask_and_get_answer():
        """
        Crea un modelo de pregunta-respuesta.
        Debe ser implementado por las subclases con los métodos para cada modelo.
        """
        pass

class SimpleQuestionAnswer(StandardQA):
    def __init__(self, query, vector_store, llm, k):
        self.query = query
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
 
    def ask_and_get_answer(query, vector_store, llm, k):
        """
        Realiza una pregunta a un modelo de recuperación de información y devuelve la respuesta más relevante.

        Args:
            query (str): La pregunta que se desea hacer al modelo.
            vector_store (obj): Objeto que contiene los vectores de las preguntas y respuestas.
            llm (str): Modelo de lenguaje que se utilizará para la generación de respuestas.
            k (int): Número de respuestas candidatas que se considerarán.

        Returns:
            str: La respuesta más relevante a la pregunta realizada.
        """
        from langchain.chains import RetrievalQA
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type='stuff')
        return chain.run(query)
