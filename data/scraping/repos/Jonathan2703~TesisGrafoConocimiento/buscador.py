import gradio as gr
import openai
from sklearn.metrics.pairwise import cosine_similarity
from openai.embeddings_utils import get_embedding
from pyfuseki import FusekiQuery
import os
import ast
import pandas as pd

# Configura tu API key de OpenAI
openai.api_key = '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'

# Cargar los datos de los periódicos
datos = pd.read_csv('C:/Users/LENOVO/Downloads/Tesis/buscador/periodicos.csv', encoding='utf-8')
datos['em'] = datos['em'].apply(lambda x: ast.literal_eval(x))

def buscar_entidades(frase):
    respuesta = openai.Completion.create(
        engine='text-davinci-002',
        prompt=f"Sacame las entidades de la siguiente frase: \"{frase}\"",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return respuesta.choices[0].text.strip().split("\n")

def buscar(busqueda, n_resultados=1):
    global datos  # Agregar esta línea para acceder a la variable global
    busqueda_embed = get_embedding(busqueda, model="text-embedding-ada-002")
    datos["Similitud"] = datos["em"].apply(lambda x: cosine_similarity([x], [busqueda_embed])[0][0])
    datos = datos.sort_values("Similitud", ascending=False) 
    return datos.loc[:n_resultados, ["Periodico", "url", "texto"]]


def buscar_periodicos():
    url = "http://localhost:3030"
    endpoint = "periodicos"   
    fuseki_query = FusekiQuery(url, endpoint)
    query_template = '''
    prefix rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    prefix rdfs:<http://www.w3.org/2000/01/rdf-schema#>
    prefix xsd:<http://www.w3.org/2001/XMLSchema#>
    prefix sc:<https://schema.org/>
    prefix ex:<http://newsont.com/>
    Select ?Periodico ?url ?t ?texto ?em
    WHERE { 
    ?Periodico a sc:Newspaper.
    ?Periodico sc:url ?url.
    ?Periodico <http://newsont.com/hasPage> ?Pagina.
    ?Pagina <http://newsont.com/hasText> ?t.
    ?t <http://newsont.com/text> ?texto.
    ?t <http://newsont.com/embending> ?em.
    }
    '''
    resultado = fuseki_query.run_sparql(query_template)
    resultado_json = resultado._convertJSON()
    bindings = resultado_json['results']['bindings']
    rows = [[binding['Periodico']['value'], binding['url']['value'], binding['t']['value'], binding['texto']['value'], f"[{binding['em']['value']}]"] for binding in bindings]
    return pd.DataFrame(rows, columns=["Periodico", "url", "t", "texto", "em"])

def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def load_data(path):
    df = pd.read_csv(path, encoding="iso-8859-1")
    df['em'] = df['em'].apply(lambda x: ast.literal_eval(x))
    return df

def buscar_frases(frase):
    entidades = buscar_entidades(frase)
    resultados = buscar(frase, n_resultados=1)
    return resultados

def preprocesar_datos():
    periodicos = buscar_periodicos()
    periodicos.to_csv('C:/Users/LENOVO/Downloads/Tesis/buscador/periodicos.csv', encoding='utf-8', index=False)
    print("Se ha escrito el archivo CSV")

# Comprobar si los datos ya se han preprocesado y, si no, preprocesarlos
if not os.path.isfile('C:/Users/LENOVO/Downloads/Tesis/buscador/periodicos.csv'):
    preprocesar_datos()

# Función para ejecutar la búsqueda cuando se presiona el botón
def buscar_frases_interfaz(frase):
    resultados = buscar(frase, n_resultados=1)
    return resultados


# Crear la interfaz y ejecutar la aplicación
with gr.Blocks() as demo:
    definicion_frase = gr.Textbox(label="Introduce una frase", placeholder="Escribe una frase", lines=3)
    output = gr.Dataframe(headers=['Periodico', 'URL', 'Texto'])
    boton_buscar = gr.Button("Buscar")
    boton_buscar.click(fn=buscar_frases_interfaz, inputs=[definicion_frase], outputs=output)

demo.launch()
# interfaz = gr.Interface(
#     fn=buscar_frases_interfaz, 
#     inputs=[definicion_frase, fecha_inicial, fecha_final], 
#     outputs=resultado_noticias,
#     output_below_inputs=True, 
#     layout="horizontal",
#     title="Buscador de Periodicos" )
# interfaz.launch()