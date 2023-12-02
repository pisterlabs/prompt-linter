from qdrant_client import QdrantClient
import json
import openai



qdrant_client = QdrantClient(
    url="https://bceadf6b-a95a-42cc-9053-32a6bce439fa.us-east-1-0.aws.cloud.qdrant.io:6333",
    api_key="yd09n92c2R6c3HuhoVFxHm17wyRX3V_Wje2sfm-iF1FpHNS8AgvGxw",
)

print("Create collection reponse:", qdrant_client)

collection_info = qdrant_client.get_collection(collection_name="mycollection")

print("Collection info:", collection_info)

#import pdfplumber

#fulltext = ""
#with pdfplumber.open("starship.pdf") as pdf:
#    # loop over all the pages
#    for page in pdf.pages:
#        fulltext += page.extract_text()

#print(fulltext)

#text = fulltext


#if __name__ == "__main__":
#    client = SearchClient()

# import data from data.json file
with open("data.json", "r") as f:
    data = json.load(f)

index_result = qdrant_client.index(data)
print(index_result)

print("====")

search_result = qdrant_client.search(
    "Tallest animal in the world, quite long neck.",
)

print(search_result)
#

chunks = []
while len(text) > 500:
    last_period_index = text[:500].rfind('.')
    if last_period_index == -1:
        last_period_index = 500
    chunks.append(text[:last_period_index])
    text = text[last_period_index+1:]
chunks.append(text)


for chunk in chunks:
    print(chunk)
    print("---")


from qdrant_client.http.models import PointStruct

points = []
i = 1
for chunk in chunks:
    i += 1

    print("Embeddings chunk:", chunk)
    response = openai.Embedding.create(
        input=chunk,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    points.append(PointStruct(id=i, vector=embeddings, payload={"text": chunk}))


operation_info = qdrant_client.upsert(
    collection_name="mycollection",
    wait=True,
    points=points
)

print("Operation info:", operation_info)


def create_answer_with_context(query):
    response = openai.Embedding.create(
        input="What is starship?",
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    search_result = qdrant_client.search(
        collection_name="mycollection",
        query_vector=embeddings,
        limit=5
    )

    prompt = "Context:\n"
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    print("----PROMPT START----")
    print(":", prompt)
    print("----PROMPT END----")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )

    return completion.choices[0].message.content

input = "what is starship?"
answer = create_answer_with_context(input)
print(answer)

# Starship is a fully reusable transportation system designed by SpaceX to service Earth orbit needs as well as missions to the Moon and Mars. It is a two-stage vehicle composed of the Super Heavy rocket (booster) and Starship (spacecraft) powered by sub-cooled methane and oxygen, with a substantial mass-to-orbit capability. Starship can transport satellites, payloads, crew, and cargo to a variety of orbits and landing sites. It is also designed to evolve rapidly to meet near term and future customer needs while maintaining the highest level of reliability.
