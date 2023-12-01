import json
import sys
from sklearn.decomposition import PCA
import numpy
import cohere
import time

apiKey = 'API KEY'
fileName = './data.txt'

co = cohere.Client(f'{apiKey}')


def getData(texts):
    if (len(texts) <= 500):
        response = co.embed(model='large', texts=texts)
        return response.embeddings
    else:
        response = co.embed(model='large', texts=texts[:500])
        time.sleep(61)
        return response.embeddings + getData(texts[500:])


def centerVectors(npData):
    for col in range(npData.shape[1]):
        npData[:, col] = numpy.subtract(npData[:, col], numpy.average(npData[:, col]))
    return npData


def main():
    texts = open(fileName).read().split('\n')

    embeddingData = centerVectors(numpy.array(getData(texts)))
    open('embeddingData.json', 'w').write(json.dumps(embeddingData.tolist()))
    # embeddingData = numpy.array(json.load(open('embeddingData.json')))

    PCAEmbedding = PCA(n_components=3)
    PCEmbedding = PCAEmbedding.fit_transform(embeddingData)

    data = {}
    data['texts'] = texts
    data['embeddings'] = PCEmbedding.tolist()
    print(json.dumps(data))


if (__name__ == "__main__"):
    main()
