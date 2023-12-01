import numpy as np
from cohere.classify import Classifications
from cohere.classify import Example


def getTopLabels(response: Classifications, n: int) -> list:
    x = response.classifications
    dictionary = {}
    values = []
    length = len(x[0].confidence)
    for i in range(length):
        values.append(x[0].confidence[i].confidence)
        dictionary[x[0].confidence[i].confidence] = x[0].confidence[i].label

    sorted_values = np.sort(values)
    top_3_object = []

    for i in range(n):
        top_3_object.append((sorted_values[length - i - 1], dictionary[sorted_values[length - i - 1]]))

    return top_3_object

