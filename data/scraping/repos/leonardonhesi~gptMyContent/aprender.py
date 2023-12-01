import os
from langchain.text_splitter     import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores      import Chroma
from langchain.document_loaders  import DirectoryLoader
from langchain.document_loaders  import TextLoader
from langchain.document_loaders  import UnstructuredURLLoader
from langchain.document_loaders  import PyPDFLoader
from langchain                   import OpenAI, PromptTemplate
from langchain.chains.summarize  import load_summarize_chain
from youtube                     import vid_toText

# Criar os diretorios caso nao existam
for f in ['./pdf','./textos','./videos', './resumos', './modelo_whisper']:
  if os.path.isdir(f) == False:
    os.mkdir(f)

# Criar um resumo de todos os documentos
def sumarize(doc):
   prompt_template = """
    Write a concise summary of the following::
    {text} TEXT IN pt-BR:"""
   PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
   chain = load_summarize_chain(OpenAI(temperature=0, max_tokens=200),
                                chain_type="map_reduce",return_intermediate_steps=False,
                                map_prompt=PROMPT,
                                combine_prompt=PROMPT)
   return chain({"input_documents": doc}, return_only_outputs=True)


# Transformar os textos em embeddings, vetor de representacao numerica do texto 
# facilitando os processamentos com IA, esse vetor e salvo em um banco especifico de vetores Chromadb
# Esta tranformação e realizada com o modelo (text-embedding-ada-002 da openAI).

def addDBList(worklist, collection_name, persist_directory):
  # Divide de acordo com limite 1000 do modelo da openai
  text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)  
  nbr_doc = text_splitter.split_documents(worklist)
  # Salva os vetores no banco
  embeddings = OpenAIEmbeddings()
  vectordb = Chroma.from_documents(nbr_doc,
                                   embeddings,
                                   collection_name=collection_name,
                                   persist_directory=persist_directory)
  vectordb.persist()
  return nbr_doc


# Carrega os documentos do diretorio respeitando filtro de exttensao
# carregando conteudo para langchain neste utilizando TextLoader
def get_dir(dir):
  loader = DirectoryLoader(dir, glob="**/*.txt", loader_cls=TextLoader)
  docs = loader.load()
  return docs


# Carrega os documentos com origem nas URLs fornecidas
def get_from_urls(urls):
  data = list()
  if urls:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
  return data 


# Carrega os documentos com origem nos PDF enviados
def get_pdf(dir_pdf):
  retorno = list()
  pdfs = os.listdir(dir_pdf)
  for pdf in pdfs:
      if pdf.endswith(".pdf"):
        loader = PyPDFLoader(dir_pdf + '/' +pdf)
        pages = loader.load_and_split()
        for p in pages:
            retorno.append(p)
  return retorno


# Salva os arquivos em PDF selecionados
def save_pdf(pdfs_files):
    for uploaded_file in pdfs_files:
        bytes_data = uploaded_file.read()
        with open('./pdf/'+ uploaded_file.name ,'wb') as the_file:
          the_file.write(bytes_data)


# Trancreve o audio dos videos em texto
def yt_to_text(videos_yt):
    video_id    = []
    nome        = []
    for v in videos_yt:
        video_id.append(v['videos'])
        nome.append(v['titulo'])  
    vid_toText(video_id, nome, './videos/', './textos/')

   
# Apagar os arquivos contidos nos diretorios recebidos como lista
def apagar_arquivos(lista_dir):
    for lim in lista_dir:
      for f in os.listdir(lim):
         os.remove(os.path.join(lim, f))


# Funcao principal que realiza o aprendizado
def ai_aprender(collection_name, videos_yt, pdfs_files, urls):
    
    # Diretorio onde fica a tabela de vetores do conhecimento
    persist_directory="./db/chroma/" + collection_name

    # Salvar os PDF's em disco
    save_pdf(pdfs_files)
    
    # Converter videos youtube em texto
    yt_to_text(videos_yt)

    # Lista para Carregar os conteudos de origem  
    from_list = []    
    
    # Carrega todos os arquivos de texto no diretorio
    conteudo_text = get_dir('./textos/') 
    
    # Carrega o conteudo das urls informada
    conteudo_url = get_from_urls(urls)
    
    # Carrega todos os arquivos PDF do diretorio
    conteudo_pdf = get_pdf('./pdf')

    # Junta todas as origens em apenas uma lista (para salvar no banco)
    for conteudo in [conteudo_text, conteudo_url, conteudo_pdf] :
      for d in conteudo:
        from_list.append(d)
      
    # Processa os arquivos Embeddings salvando no banco de vetor 
    docs = addDBList(from_list, collection_name, persist_directory)
    
    # Gravar um resumo do conteudo
    resumo = sumarize(docs)
    with open('./resumos/'+ collection_name +'.txt','w', encoding='utf-8') as the_file:
       the_file.write(resumo['output_text'])

    # Remover os arquivos
    apagar_arquivos(['./videos/', './textos/', './pdf' ])

    return True

