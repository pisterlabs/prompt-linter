# %%
# Librerías
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models import LdaMulticore
import pandas as pd
import spacy
import nltk
import json
import os
# %% Funciones


def procesamiento(columna: str, df: pd.DataFrame):
    """Función para procesar el texto y obtener columnas tokenizada y radicalizada

    Args:
        columna (str): columna a procesar
        df (pd.DataFrame): dataframe en donde se encuentra el texto a procesar

    Returns:
        pd.DataFrame: Data Frame con columna procesada
    """
    # Modelo de spacy que se utilizará
    # spacy.cli.download('es_core_news_md')
    es = spacy.load('es_core_news_md')

    # Etiquetas a remover del texto lematizado
    removal = ['ADV', 'PRON', 'CCONJ', 'PUNCT',
               'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']

    # Convertir a objeto spaCy
    aux = df[columna].str.lower().apply(es)

    # Tokenización
    df[f'{columna} procesado'] = aux.apply(
        lambda x: [token for token in x])
    # Normalización (minuscula, tamaño > 3 y solo letras)
    df[f'{columna} procesado'] = df[f'{columna} procesado'].apply(
        lambda x: [token for token in x if len(token) > 3 and token.is_alpha])
    # Remover stopwords (combinación de contexto y spacy).
    # Convertir Token a str
    with open('/Users/'+os.getlogin()+'/OneDrive - Accenture/ACNCustomerAnalysis/NLP_Analitycs/Scripts/sw_es.txt', 'r', encoding='utf-8') as file:
        stop_words_contexto = {line.split(None, 1)[0] for line in file}
    es.Defaults.stop_words |= stop_words_contexto
    df[f'{columna} procesado'] = df[f'{columna} procesado'].apply(
        lambda x: [token for token in x if not token.is_stop])

    # Segmentación en oraciones
    df[f'{columna} segmentado'] = aux.apply(
        lambda x: ", ".join([segment.orth_ for segment in x.sents]))

    # Extracción de entidades
    df[f'Entidades de {columna}'] = aux.apply(
        lambda x: ", ".join([ent.text for ent in x.ents]))

    # Radicalización (stemming)
    stemmer = nltk.SnowballStemmer('spanish')
    df[f'{columna} radicalizado'] = df[f'{columna} procesado'].apply(
        lambda x: ", ".join([stemmer.stem(token.orth_) for token in x]))

    # Lemmatization
    df[f'{columna} lematizado'] = df[f'{columna} procesado'].apply(
        lambda x: ", ".join([token.lemma_ for token in x if token.pos_ not in removal]))

    # Procesado a string
    df[f'{columna} procesado'] = df[f'{columna} procesado'].apply(
        lambda x: ", ".join([token.orth_ for token in x]))


def lista_ngramas(val_ent: str, val_pal: str, indice: int, n: int):
    """Función que genera la lista de todas las palabras del conjunto
    de datos con la frecuencia de cada una por artículo, 
    especifica a que artículo pertenece y si es una entidad (1) o no
    (0).

    Args:
        val_ent (str): cadena de entidades obtenida en el procesamiento
        val_pal (str): cadena de palabras obtenida en el procesamiento
        indice (int): indice del artículo al que corresponden las cadenas
        n (int): tamaño de la subsecuencia del n-grama

    Returns:
        pd.DataFrame: DataFrame con el indice, la palabra, frecuencia de
        aparición, ID del artículo al que pertenece. Si es solo una palabra
        se incluye la columna de entidad que indica si lo es o no
    """
    # Verificar si existen entidades
    if type(val_ent) == float:
        entidades = {}
    else:
        entidades = set(val_ent.split(', '))

    palabras = val_pal.split(', ')
    ngrams = list(nltk.ngrams(palabras, n))
    freq_pal = dict(nltk.FreqDist(ngrams))

    if n == 1:
        lista = []
        for key, value in freq_pal.items():
            word = ", ".join(list(key))
            if word in entidades:
                lista.append([word, value, indice, 1])
            else:
                lista.append([word, value, indice, 0])
        df_frec = pd.DataFrame(
            lista, columns=['Palabra', 'Frecuencia', 'ID_Articulo', 'Entidad'])
    else:
        lista = []
        for key, value in freq_pal.items():
            lista.append([", ".join(list(key)), value, indice])
        df_frec = pd.DataFrame(
            lista, columns=['Palabra', 'Frecuencia', 'ID_Articulo'])

    df_frec.index.name = 'ID_Token'
    return df_frec


def asignar_industrias(df: pd.DataFrame):
    """Función para asignar las industrias a cada artículo según de la empresa
    de la que se esta hablando

    Args:
        df (pd.DataFrame): DataFrame a asignarle la columna Industria
    """
    with open('/Users/'+os.getlogin()+'/OneDrive - Accenture/ACNCustomerAnalysis/Web_Scraping/config.json', encoding='utf-8-sig') as f:
        data = json.load(f)

    dic_ind = {}
    for i in data['industrias']:
        for j in i['Empresas']:
            dic_ind[j] = i['industria']

    df['Industria'] = df['Empresa'].apply(lambda x: dic_ind[x])


# %% LECTURA Y PREPARACIÓN DE LOS DATOS
path = 'C:/Users/'+os.getlogin() + \
    '/OneDrive - Accenture/ACNCustomerAnalysis/Medios_Comunicacion'
# LEER ARCHIVOS CON DATOS
df_raw = pd.read_csv(path+'/data/raw/database.csv',
                     encoding='utf-8-sig', index_col=[0])
df_curated = pd.read_csv(
    path+'/data/curated/curated_database.csv', encoding='utf-8-sig', index_col=[0])

# Verificar cuales articulos no han sido procesados
df = df_raw[~df_raw['Titulo'].isin(df_curated['Titulo'])]
# %%
if len(df) > 0:
    # Estandarización formato fechas
    df['Contenido'] = df['Contenido'].str.replace(
        '\r|\n|\f|\v', ' ')
    df['Titulo'] = df['Titulo'].str.replace('\r|\n|\f|\v', ' ')
    df['Resumen'] = df['Resumen'].str.replace('\r|\n|\f|\v', ' ')
    df['Autor'] = df['Autor'].str.replace('\r|\n|\f|\v', ' ')
    df['Fecha Publicacion'] = pd.to_datetime(
        df['Fecha Publicacion']).dt.strftime('%d-%m-%Y')
    df['Fecha Extraccion'] = pd.to_datetime(
        df['Fecha Extraccion']).dt.strftime('%d-%m-%Y')

    # ELIMINACIÓN COLUMNAS Y FILAS NO RELEVANTES
    # Eliminar filas sin información en la columna Contenido
    df = df.drop(df[df['Contenido'] == "SIN PARRAFOS"].index).reset_index(
        drop=True)
    df = df.drop(df[df['Contenido'].isna()].index).reset_index(drop=True)
    df = df.drop(df[df.Contenido.str.len() < 500].index).reset_index(drop=True)

    # Eliminar aquellos artículos que no tengan la empresa en su contenido y reemplazar los bigramas
    empresas = df['Empresa'].unique()
    no_cont = []

    for emp in empresas:
        df_empresa = df[df['Empresa'] == emp]
        df_empresa['Contenido'] = df_empresa['Contenido'].str.lower()
        if emp == 'Grupo Exito':
            # Guardar aquellos que no contengan a la empresa en su contenido
            emp = 'Éxito'
            no_cont += df_empresa[~df_empresa['Contenido']
                                  .str.contains(emp.lower())].index.to_list()
            # Reemplazar el espacio _ para considerar el bigrama
            df['Contenido'] = df['Contenido'].str.replace(
                'Grupo Éxito', 'Grupo_Exito')
            df['Contenido'] = df['Contenido'].str.replace(
                'Éxito', 'Grupo_Exito')
        elif ' ' in emp:
            # Guardar aquellos que no contengan a la empresa en su contenido
            aux = emp.split()[1]
            no_cont += df_empresa[~df_empresa['Contenido']
                                  .str.contains(aux.lower())].index.to_list()
            # Reemplazar el espacio _ para considerar el bigrama
            df['Contenido'] = df['Contenido'].str.replace(
                emp, emp.replace(' ', '_'))
            df['Contenido'] = df['Contenido'].str.replace(
                aux, emp.replace(' ', '_'))
        else:
            # Guardar aquellos que no contengan a la empresa en su contenido
            no_cont += df_empresa[~df_empresa['Contenido']
                                  .str.contains(emp.lower())].index.to_list()

    df = df.drop(no_cont).reset_index(drop=True)

# %%
procesamiento('Contenido', df)
df_palabras = pd.DataFrame()
df_bigramas = pd.DataFrame()
df_trigramas = pd.DataFrame()

len_df = len(df)
len_curated = len(df_curated)

for i in range(len_curated, len_curated + len_df):
    aux_palabras = lista_ngramas(df.loc[i - len_curated, 'Entidades de Contenido'],
                                 df.loc[i - len_curated, 'Contenido procesado'], i, 1)
    df_palabras = pd.concat([df_palabras, aux_palabras], ignore_index=True)

    aux_bigramas = lista_ngramas(df.loc[i - len_curated, 'Entidades de Contenido'],
                                 df.loc[i - len_curated, 'Contenido procesado'], i, 2)
    df_bigramas = pd.concat([df_bigramas, aux_bigramas], ignore_index=True)

    aux_trigramas = lista_ngramas(df.loc[i - len_curated, 'Entidades de Contenido'],
                                  df.loc[i - len_curated, 'Contenido procesado'], i, 3)
    df_trigramas = pd.concat([df_trigramas, aux_trigramas], ignore_index=True)

df_curated = pd.concat([df_curated, df], ignore_index=True)
# Eliminar Aquellos con titulos repetidos
df_curated = df_curated[~df_curated['Titulo'].duplicated()
                        ].reset_index(drop=True)
df_curated = df_curated[~df_curated['Fecha Publicacion'].isna()].reset_index(
    drop=True)
df_curated.index.name = 'ID_Articulo'
df_curated.to_csv(path+'/data/curated/curated_database.csv',
                  encoding='utf-8-sig')

palabras_csv = pd.read_csv(
    path+'/data/curated/palabras.csv', encoding='utf-8-sig', index_col=[0])
df_palabras = pd.concat([palabras_csv, df_palabras], ignore_index=True)
df_palabras.to_csv(path+'/data/curated/palabras.csv', encoding='utf-8-sig')

bigramas_csv = pd.read_csv(
    path+'/data/curated/bigramas.csv', encoding='utf-8-sig', index_col=[0])
df_bigramas = pd.concat([bigramas_csv, df_bigramas], ignore_index=True)
df_bigramas.to_csv(path+'/data/curated/bigramas.csv', encoding='utf-8-sig')

trigramas_csv = pd.read_csv(
    path+'/data/curated/trigramas.csv', encoding='utf-8-sig', index_col=[0])
df_trigramas = pd.concat([trigramas_csv, df_trigramas], ignore_index=True)
df_trigramas.to_csv(path+'/data/curated/trigramas.csv', encoding='utf-8-sig')

# topicos(df_curated, 'Contenido')
# %%
