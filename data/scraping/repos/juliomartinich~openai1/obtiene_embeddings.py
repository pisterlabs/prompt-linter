# obtiene embeddings de una lista de palabras (o frases)
# los salva en el archivo
# el ultimo item de cada lista guardada es la palabra
import openai, os, csv

openai.api_key = os.environ.get('OPENAIAPIKEY')

palabras = ["gato", "perro", "burro", "cisne", "zorro", "liebre", "caballo", "nutria", "coipo" ]

vectores = {}
for palabra in palabras:
  vectores[palabra] = openai.Embedding.create(
    input=palabra, model="text-embedding-ada-002"
  )["data"][0]["embedding"]

file_name = "vectores_de_animales.csv"

# Open the file in write mode
with open(file_name, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the vectors to the file
    for palabra in palabras:
        #pongo al final del vector la palabra
        vectores[palabra].append(palabra)
        writer.writerow(vectores[palabra])

