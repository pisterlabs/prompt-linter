import featureform as ff
from featureform import local
from io import StringIO
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai


def dataProcess(className):
    client = ff.Client(local=True)

    chapters = local.register_directory(
        name=className +"-chapters",
        path="/" + className + "/data/files",
        description="Text from " + className + " Chapters",
    )

    ed_posts = local.register_directory(
        name=className + '-edstem',
        path="/"+className + '/data/edstem',
        description=className + ' Posts from edstem',
    )

    return chapters, ed_posts