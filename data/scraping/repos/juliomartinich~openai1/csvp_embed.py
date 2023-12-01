# lee un archivo csv con textos ,calcula sus embeddings y los salva en otro csv
import openai, os, csv

openai.api_key = os.environ.get('OPENAI_API_KEY')

def read_embedding(texto):

  vector_embedding = openai.Embedding.create(
    input=texto, model="text-embedding-ada-002"
  )["data"][0]["embedding"]

  return vector_embedding

# Define the file name and path where the CSV file is located
file_name_read = "mis_word_p.csv"
file_name_save = "mis_word_p_embed.csv"

# Open the file in read mode
with open(file_name_read, mode='r') as file_read:
    # Create a CSV reader object
    reader = csv.reader(file_read, delimiter=",")

    # Open the file in write mode
    with open(file_name_save, mode='w', newline='') as file_save:
        # Create a CSV writer object
        writer = csv.writer(file_save)

        # Convert the CSV data to a Python vector
        for row in reader:
            vector = read_embedding(row[2])
            vector.append(row[0])
            vector.append(row[1])
            vector.append(row[2])
            writer.writerow(vector)
            print(row[0], row[1], vector[0])

